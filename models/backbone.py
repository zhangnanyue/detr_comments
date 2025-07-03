# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

# 目标：用一个行为完全固定的归一化层，来替换掉 ResNet 中原本行为可变的 BatchNorm2d 层。
# 问题/原因：
    # 普通的 BN 层会根据当前训练批次（mini-batch）的数据的均值和方差来归一化特征。
    # 在 ImageNet 上预训练时，批次很大，计算出的均值方差很稳定。
    # 但在下游任务（如目标检测）中，我们的批次大小通常很小（比如每张 GPU 上只有 1-2 张图），
    # 这时计算出的均值方差波动很大，如果用它来归一化，会“污染”预训练好的高质量特征。
# 解决方法：
    # 使用 FrozenBatchNorm2d 来代替标准 BatchNorm2d，它不会根据当前批次的数据来计算均值和方差，而是使用预训练好的均值和方差。
# 这样，我们就可以在下游任务中使用更小的批次大小，同时保持预训练好的高质量特征。

# 总结：FrozenBatchNorm2d 保护了预训练模型的权重不受小批次训练的干扰，从而保留了其强大的泛化特征提取能力。
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # 参数冻结，根据train_backbone的值，决定是否冻结参数
        # 通常我们不希望在自己的小数据集上从头训练整个 ResNet，因为很容易过拟合。
        # train_backbone=False 会将所有参数的 requires_grad 设为 False，
        # 这样在训练时这些参数就不会被更新。我们只使用它预训练好的特征提取能力。
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # 根据是否需要返回中间层的特征，设置返回的层级
        # 一个完整的 ResNet 是为图像分类设计的，它最后会输出一个 1000 类的向量。
        # 但我们做目标检测，需要的是它倒数第二层的二维特征图。
        # IntermediateLayerGetter 是 PyTorch 的一个工具，它能帮我们方便地“劫持”网络的中间输出。
        # 这里 {'layer4': "0"} 的意思就是：“我们只对 ResNet 的 layer4 的输出感兴趣”。
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

# 根据指定的名字（比如resnet50），从PyTorch库torchvision中加载一个预训练好的ResNet模型
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 从torchvision.models中加载一个预训练模型，例如 “resnet50”
        # norm_layer=FrozenBatchNorm2d，将ResNet中所有的标准 BatchNorm2d 层都换成了我们自定义的 FrozenBatchNorm2d 层，
        # 主要作用是冻结BN层参数，不参与训练，具体原因可以看FrozenBatchNorm2d函数解析
        # replace_stride_with_dilation=[False, False, dilation]，将ResNet的stride设置为1，并使用dilation参数来控制膨胀卷积的膨胀率
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # 获取backbone的输出通道数，例如resnet18和resnet34的输出通道数是512，resnet50的输出通道数是2048
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential): 
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    # NestedTensor 是 PyTorch 中的一个数据结构，用于表示嵌套的 Tensor 列表。
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

# 一个完整的BackBone由三部分构成：
    # 1. 位置编码模块
    # 2. 真正的Backbone（例如ResNet）
    # 3. 将backbone和position embedding拼接起来，打包成一个模块
def build_backbone(args):
    # 1. 构建position embedding位置编码模块
    position_embedding = build_position_encoding(args)
    # 2. 判断是否要训练backbone的参数
    train_backbone = args.lr_backbone > 0
    # 3. 判断是否需要返回中间层的特征（用于分割任务）
    return_interm_layers = args.masks
    # 4. 创建真正的 Backbone（例如ResNet）
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 5. 将backbone和position embedding拼接起来，打包成一个模块
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
