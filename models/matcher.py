# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

# 实现了DETR的核心思想：集合预测（set prediction）
# 核心问题：在计算损失之前，如何将模型的100个预测框与图中不确定数量的真实物体框（Ground Truth）进行匹配？

# 2.linear_sum_assignment：scipy库中的函数，实现了匈牙利算法。
# 3.box_cxcywh_to_xyxy：将中心点坐标和宽高转换为左上角和右下角坐标。
# 4.generalized_box_iou：计算两个框的GIoU损失。
# 5.box_ops.py：包含一些用于处理边界框的辅助函数。
# 6.util.misc.py：包含一些用于处理张量的辅助函数。
# 7.util.box_ops.py：包含一些用于处理边界框的辅助函数。

# HungarianMatcher类：匈牙利匹配器，实现了论文中提到的二分图匹配（Bipartite Matching）算法。
# 背景：
    # DETR 的 Transformer Decoder 并行输出一个固定大小（比如 N=100）的预测集合（无序的），
    # 每个元素包含一个类别预测和一个边界框预测。而一张训练图像中，真实物体的数量是不定的（比如 3 个、5 个、或者 10 个）。
# 问题：
    # 在计算损失函数时，我们必须知道这 100 个预测中的哪一个应该对第 1 个真实物体负责？哪一个又该对第 2 个真实物体负责？
    # 我们不能随便拉一个预测去和一个真实物体配对，这样会让模型学习混乱
# 解决方案：
    # 使用匈牙利算法（Hungarian Algorithm）来寻找一个”最优匹配“。如何理解这个“最优”？
    # 它总能找到一个总代价最小的一对一匹配。代价越小，意味着预测框和真实框越“般配”
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    #  @torch.no_grad() 装饰器包裹，意味着整个匹配过程不计算梯度，它只是为了确定配对关系，其本身不参与反向传播。
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
