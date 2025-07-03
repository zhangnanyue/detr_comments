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
    # 使用 FrozenBatchNorm2d 来代替标准 BatchNorm2d，它不会根据当前批次的数据来计算均值和方差，
    # 而是使用预训练好的均值和方差。这样，我们就可以在下游任务中使用更小的批次大小，同时保持预训练好的高质量特征。

# 总结：FrozenBatchNorm2d 保护了预训练模型的权重不受小批次训练的干扰，从而保留了其强大的泛化特征提取能力。
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    # n 代表输入特征图的通道数 (number of channels)。BatchNorm 是对每个通道独立进行操作的，所以需要知道有多少个通道。
    # 该如何理解BatchNorm 是对每个通道独立进行操作的？
        # BatchNorm 对每个通道的归一化，使用的均值、方差、以及后续缩放（γ）和偏移（β）参数，都是这个通道专属的。通道与通道之间互不干扰。
        # 假设输入的特征图X的形状是（B，C，H，W），例如(4, 3, 10, 10)，表示：
            # B = 4：表示有4个样本
            # C = 3：表示有3个通道
            # H = 10：表示每个通道的特征图高度为10
            # W = 10：表示每个通道的特征图宽度为10
        # BatchNorm的计算步骤如下：
            # 1. 计算均值（Mean）：
                # 对于通道0 (R): 它会收集所有4张图里，所有 10x10 个像素上属于通道0的值，总共 4 * 10 * 10 = 400 个数值。
                # 然后计算这400个数的平均值。得到一个标量 mean_channel_0。
                # 对于通道1（G）和通道2（B），独立重复上述过程。获得 mean_channel_1 和 mean_channel_2。
                # 最终我们会得到一个形状为（C, ），即（3, ）的均值向量[mean_channel_0, mean_channel_1, mean_channel_2]
            # 2. 计算方差（Variance）：
                # 过程与计算均值完全一样，只是计算的是方差。最终也得到一个形状为 (C,) 的方差向量。
            # 3. 归一化：
                # 用每个通道各自的均值和方差，去归一化这个通道上的所有数值（N*H*W 个）。
            # 4. 缩放和偏移 (weight and bias):
                # 模型还有两个可学习的参数 gamma（weight） 和 beta（bias），它们的形状也都是 (C,)。
                # 通道0上的所有数据，都会乘以 gamma_0，加上 beta_0。
                # 通道1上的所有数据，都会乘以 gamma_1，加上 beta_1。
                # 以此类推。
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # self.register_buffer(...)这是理解此类的关键。
        # register_buffer 是 PyTorch 的一个函数，用于注册一个不被视为模型参数的状态。
        # Buffer vs. Parameter:
            # nn.Parameter: 会被 model.parameters() 收集，从而被优化器（如Adam, SGD）更新。
            # buffer: 也是模型状态的一部分（会被保存在 state_dict 中），但不会被优化器更新。
        # 通过使用 register_buffer，确保了weight, bias, running_mean, running_var这些值在模型加载后，在训练过程中不会被改变，从而实现了“冻结”。

        # 注册一个名为 weight 的缓冲区。这是 BatchNorm 中的可学习仿射参数 γ (gamma)。
        # 这里初始化为全1向量。当从预训练模型加载权重时，这些值会被覆盖。
        self.register_buffer("weight", torch.ones(n))
        # 注册一个名为 bias 的缓冲区。这是 BatchNorm 中的可学习仿射参数 β (beta)。
        self.register_buffer("bias", torch.zeros(n))
        # 注册一个名为 running_mean 的缓冲区。即 BatchNorm 中在整个训练集上估计的均值 μ。
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册一个名为 running_var 的缓冲区。即 BatchNorm 在整个训练集上估计的方差 σ²。
        self.register_buffer("running_var", torch.ones(n))

    # 这个函数的功能是自定义当模型加载预训练权重（即 state_dict）时的行为。
    # 它是一个“钩子”（Hook），让我们可以介入标准的加载流程，做一些特殊处理。
    # 在加载预训练权重时，通常会包含一些不需要的键（比如 num_batches_tracked），
    # 这些键在 FrozenBatchNorm2d 中不需要，所以需要删除。
    # 然后调用父类（即标准 BatchNorm2d）的 _load_from_state_dict 函数，完成标准的加载流程。
    # 这样，我们就可以在加载预训练权重时，删除不需要的键，并调用父类的加载方法，完成标准的加载流程。
    # 总结：_load_from_state_dict 在这里扮演了一个“兼容性适配器”的角色。
        # 它通过在加载前拦截并清理 state_dict，
        # 解决了预训练模型和我们自定义模型之间因 num_batches_tracked 参数存在与否而导致的接口不匹配问题。
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 这几个参数都是 PyTorch 加载流程的标准输入，我们主要关注 state_dict 和 prefix
        # state_dict：包含预训练权重的字典
        # prefix：键的前缀，用于构造完整的键名
            # 当模型嵌套时，state_dict 中的键会有前缀。
            # 比如，如果您的 ResNet 中有一个层叫 layer1.0.bn1，那么它的权重键就是 layer1.0.bn1.weight。
            # 这里的 prefix 就是 layer1.0.bn1.。
        # local_metadata：本地元数据，通常是空字典
        # strict：是否严格检查加载的键是否与模型参数匹配
        # missing_keys：加载时找不到的键
        # unexpected_keys：加载时找到的但模型不需要的键
        # error_msgs：加载时可能出现的错误消息

        # 1. 正确拼接出那个我们不想要的键的完整名称
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        # 2. 检查这个键是否存在于要加载的 state_dict 中
        if num_batches_tracked_key in state_dict:
            # 3. 如果存在，就删除它， 此时，state_dict 就变“干净”了，不再包含 FrozenBatchNorm2d 无法处理的键。
            del state_dict[num_batches_tracked_key]
        # 在清理完字典后，这行代码调用 nn.Module 中原始的、默认的加载函数，让它去处理那个已经被我们“消毒”过的 state_dict。
        # 4. 调用父类（即标准 BatchNorm2d）的 _load_from_state_dict 函数，完成标准的加载流程。
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    # 标准的BatchNorm2d的公式： y = γ * (x - μ) / sqrt(σ² + ϵ) + β
    # 假设 scale = γ / sqrt(σ² + ϵ)
    # 也可以写成： y = (x - μ) * scale + β = scale * x - scale * μ + β = scale * x + (β - scale * μ)
    # 其中，γ 和 β 是可学习的仿射参数(即weight和bias)，μ 和 σ² 是整个训练集上估计的均值和方差。
    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        # 将 self.weight 等缓冲区都是一维向量，形状为 (C,)，其中 C 是通道数。
        # 输入 x 是一个四维张量，形状为 (N, C, H, W)。
        # 为了让一维的 (C,) 向量能与四维的 (N, C, H, W) 张量进行广播（broadcast）运算，需要将向量的形状变为 (1, C, 1, 1)。
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        # 极小值，防止分母为0
        eps = 1e-5
        # 计算缩放因子 scale = γ / sqrt(σ² + ϵ)
        scale = w * (rv + eps).rsqrt()
        # 计算偏移量 bias = β - scale * μ
        bias = b - rm * scale
        # 最终的输出：scale * x + (β - scale * μ)
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

    # 这个前向传播过程非常清晰：
        # 1. 特征提取: 将 tensor_list 中的图像数据 (.tensors) 送入 self.body（即 IntermediateLayerGetter 包装的 ResNet）进行卷积计算，
            # 得到低分辨率的特征图。例如，[2, 3, 800, 800] 的输入图像，经过 ResNet 后，得到的特征图 x 可能是 [2, 2048, 25, 25]。
        # 2. 掩码下采样: 同时，原始的、高分辨率的 mask ([2, 800, 800]) 也被相应地进行下采样（通过 F.interpolate），变成与特征图 x 匹配的尺寸（[2, 25, 25]）。
        # 3. 重新打包: 最后，将下采样后的特征图和下采样后的掩码重新组合成一个新的 NestedTensor 对象，并放入一个字典 out 中返回。这个字典 out 就是我们最终得到的 xs。
    # 结论: 调用过程是一个“解包 -> 计算 -> 再打包”的过程，确保了输出的特征图 xs 仍然携带着与之对应的、尺寸正确的掩码信息。
    def forward(self, tensor_list: NestedTensor):
        # 1. 从 NestedTensor 中解包出图像张量
            # 注意，这里只传入了 .tensors，没有传入 .mask
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            # 2. 从原始的tensor_list中取出mask
            m = tensor_list.mask
            assert m is not None
            # 3. 对mask进行插值，使其与特征图x的形状匹配
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # 4. 将特征图 x 和新的 mask 重新打包成一个 NestedTensor
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

# 目的：将backbone和position embedding拼接起来，打包成一个模块
# 要解决的问题：Transformer需要两个输入：图像内容特征 (features) 和这些特征的空间位置信息 (positional encodings)。
class Joiner(nn.Sequential): 
    # 它继承自 nn.Sequential，这是 PyTorch 中一个非常有用的容器。
    # nn.Sequential 会按照传入的顺序，依次执行其包含的模块。
    # 接收 backbone 和 position_embedding 这两个模块作为参数。
    def __init__(self, backbone, position_embedding):
        # 这是 Joiner 的全部魔法所在。它调用父类 nn.Sequential 的构造函数，并将 backbone 和 position_embedding 传进去。
        # nn.Sequential 会自动将这两个模块存储起来。此时，我们可以通过 self[0] 访问 backbone，通过 self[1] 访问 position_embedding。
        super().__init__(backbone, position_embedding)


    # 由于我们希望forward 方法能返回两个值 out pos，所以它需要重写 nn.Sequential 的默认 forward 行为。
    # 输入是一个 NestedTensor 对象，它包含两个属性：
        # tensors：一个形状为 (B, C, H, W) 的张量，表示图像特征。
        # mask：一个形状为 (B, H, W) 的布尔掩码，表示哪些位置是有效的（即哪些像素是可见的）。
    def forward(self, tensor_list: NestedTensor):
        # 调用 self[0]，也就是 backbone 模块，对输入 tensor_list 进行处理，提取特征。
        # 在 PyTorch 中，对一个模块实例使用 () 进行调用（如 model(input)），等同于调用它的 forward 方法（model.forward(input)）。
        # 所以，xs = self[0](tensor_list) 实际上执行的是： xs = backbone.forward(tensor_list)
        # backbone 的输出 xs 是一个字典，键是层的名字（如 "0"），值是特征图 NestedTensor。
        xs = self[0](tensor_list)
        # 初始化两个空列表，分别用于存放最终的特征图和位置编码。
        out: List[NestedTensor] = []
        pos = []
        # 遍历 backbone 输出的字典。虽然 DETR 默认只用 layer4 的输出，但这个写法具有通用性，可以处理多层特征输出的情况。
        for name, x in xs.items():
            # 将特征图 x 添加到 out 列表中。
            out.append(x)
            # position encoding
            # 调用 self[1]，也就是 position_embedding 模块，并将特征图 x 作为输入。
            # position_embedding 会根据 x 的形状计算出对应的位置编码。
            # .to(x.tensors.dtype) 确保位置编码的数据类型与特征图的数据类型一致（例如都是 torch.float32）。
            # 将生成的位置编码添加到 pos 列表中。
            pos.append(self[1](x).to(x.tensors.dtype))

        # 返回包含特征图的列表 out 和包含位置编码的列表 pos。这两个输出将直接作为下一阶段 Transformer 的输入。
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
    # 5. 将backbone和position embedding拼接起来，打包成一个模块，好处是在主模型中，我们只需要调用一次model
        # 如果在主模型里分别调用它们，代码会是这样：
            #   features = self.backbone(images)
            #   positions = self.pos_encoder(features)
            #   transformer_output = self.transformer(features, positions, ...)
        # 而使用 Joiner 后，我们只需要调用一次 model(images)，代码会自动处理好位置编码，并传给 Transformer。
            # features, positions = self.joiner(images)
            # transformer_output = self.transformer(features, positions, ...)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
