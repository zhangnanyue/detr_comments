# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer

# 模型主体，整合了BackBone、Transformer和最终的预测头
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # 这是一个线形层（即全连接层），用作分类头。
        # 它将 Transformer 解码器输出的 hidden_dim 维特征向量映射到一个 num_classes + 1 维的向量，用于预测类别。
        # +1 是因为需要一个额外的类别来表示"无目标"（no object）。
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 这是一个多层感知机（MLP），用作回归头。
        # 它将 hidden_dim 维特征向量映射到一个 4 维向量，用于预测边界框的坐标 (center_x, center_y, height, width)。
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 它是一个可学习的嵌入层，尺寸为 (num_queries, hidden_dim)。
        # 这 num_queries 个向量作为输入送入 Transformer 的解码器，
        # 可以被理解为 num_queries 个学习到的"锚点"或"查询"，每个查询向量负责在图像中寻找并描述一个特定的物体。
        # 它们在训练过程中会学习到位置信息，因此可以看作是可学习的位置编码。
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 这是一个 1x1 的卷积层。
        # 它的作用是将骨干网络（如 ResNet）输出的特征图的通道数（例如 2048）降低到 Transformer 所需的 hidden_dim（例如 256），统一特征维度。
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        # 一个布尔值，决定是否计算辅助损失（Auxiliary Loss）。
        # 如果为 True，模型会对 Transformer 解码器每一层的输出都计算损失，这有助于模型训练和收敛
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # 处理输入的图像。
        # 由于批次（batch）中的图像可能大小不一，这里会将它们填充（padding）到相同尺寸，并生成一个 mask 来标记哪些区域是填充的无效区域。
        # NestedTensor 是一个自定义的数据结构，同时包含图像张量 tensor 和 mask。
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 将图像送入骨干网络（如 ResNet）提取特征。
        # features 是一个列表，包含了骨干网络输出的特征图；pos 是一个列表，包含了对应特征图的位置编码。
        features, pos = self.backbone(samples)
        # 从骨干网络最后一层的输出中提取特征图 src 和其对应的 mask。src 的维度是 [batch_size, C, H, W]。
        # 从骨干网络最后一层输出的特征中提取特征图张量和掩码
        # features[-1] 是一个 NestedTensor 对象，包含特征图张量和对应的掩码
        # decompose() 方法将 NestedTensor 分解为两个部分：
        # - src: 特征图张量，形状为 [batch_size, channels, height, width]
        # - mask: 布尔掩码张量，形状为 [batch_size, height, width]，True 表示填充区域，False 表示有效区域
        src, mask = features[-1].decompose()
        assert mask is not None
        # 这是 Transformer 的核心调用。
            # self.input_proj(src): 首先用 1x1 卷积将 src 的通道数降维到 hidden_dim。
            # mask: 传递填充区域的掩码给 Transformer 的注意力机制，使其忽略这些区域。
            # self.query_embed.weight: 将可学习的 Object Queries 作为解码器的输入。
            # pos[-1]: 将骨干网络输出的位置编码送入 Transformer。
            # hs: Transformer 解码器的输出，维度为 [dec_layers, batch_size, num_queries, hidden_dim]。它包含了6个解码器层（如果dec_layers=6）的输出结果。
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # 将解码器输出 hs 通过分类头，得到每个 query 的类别预测 logits。
        # 输出维度为 [dec_layers, batch_size, num_queries, num_classes + 1]。
        outputs_class = self.class_embed(hs)
        # 将解码器输出 hs 通过回归头，得到每个 query 的边界框预测。
        # .sigmoid() 函数将输出归一化到 [0, 1] 范围，表示相对于图像尺寸的相对坐标 (cx, cy, h, w)。
        # 输出维度为 [dec_layers, batch_size, num_queries, 4]。
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # 将最终的预测结果存储在字典 out 中。
        # 其中 pred_logits 是类别预测，pred_boxes 是边界框预测。
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # 如果启用了辅助损失，则将前面所有解码器层的预测也一并打包返回。
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

# 负责计算模型预测和真实标签之间的损失。
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        # 传入一个匹配器模块。
        # 这个匹配器的作用是为 num_queries 个预测和图像中 N 个真实物体框（Ground Truth）找到一个最佳的二分匹配。
        # 数学原理：匈牙利算法：匹配过程基于匈牙利算法。
        # 它计算一个 num_queries x N 的成本矩阵（cost matrix），其中每个元素代表"第 i 个预测"和"第 j 个真实框"的匹配成本。
        # 这个成本通常是分类损失和 L1/GIoU 损失的加权和。算法的目标是找到一个总成本最低的匹配方案。
        self.matcher = matcher
        #  一个字典，包含了不同损失项的权重，例如 {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}。
        # 这允许我们调整不同损失的重要性。
        self.weight_dict = weight_dict
        # eos (end of sentence) 在这里指 "no object" 类别。
        # 这个系数用于降低"无目标"这个类别的权重。
        # 因为 num_queries (100) 通常远大于一张图里的物体数，大部分预测都应该是"无目标"，
        # 降低其权重可以防止模型过于倾向于预测背景，有助于类别平衡。
        self.eos_coef = eos_coef
        # 一个列表，指定要计算哪些损失，例如 ['labels', 'boxes', 'cardinality']
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, we don't use it in our loss
        num_classes = 250
    device = torch.device(args.device)

    # 调用 backbone.py 中的函数构建骨干网络。
    backbone = build_backbone(args)

    # 调用 transformer.py 中的函数构建 Transformer 模型。
    transformer = build_transformer(args)

    # 创建 DETR 模型实例。
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    # 如果需要分割任务，则调用 segmentation.py 中的函数构建分割模型。
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    # 调用 matcher.py 中的函数构建匈牙利匹配器。
    matcher = build_matcher(args)
    # 根据配置参数，定义各种损失的权重。
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 定义需要计算的损失类型。
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # 实例化损失函数 SetCriterion
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    # 实例化后处理模块。
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    # 返回构建好的模型、损失函数和后处理器，可以直接用于训练和评估。
    return model, criterion, postprocessors
