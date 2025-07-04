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

    # 只接收三个超参数 cost_class, cost_bbox, cost_giou。
    # 这三个参数是权重系数 (λ)，用于控制在计算总匹配代价时，分类的“般配”程度和框的“般配”程度各占多大的比重。
    # 最终的匹配代价 C 是这三者的加权和：C = λ_class * cost_class + λ_bbox * cost_bbox + λ_giou * cost_giou。
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class # 系数，用于控制分类代价的权重
        self.cost_bbox = cost_bbox # 系数，用于控制边界框代价的权重
        self.cost_giou = cost_giou # 系数，用于控制GIoU代价的权重
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
        # 获取批次大小和查询数量
        # outputs["pred_logits"] 的形状是 [batch_size, num_queries, num_classes]
        # 其中 batch_size 是批次大小，num_queries 是查询数量，num_classes 是整个数据集的类别数量
        # shape[:2] 取前两个维度，即 [batch_size, num_queries]
        # bs 是批次大小，表示一次处理多少张图像
        # num_queries 是查询数量，表示每张图像生成多少个预测框
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # 1. 展平预测，以便进行批量矩阵运算
        # .flatten(0, 1): 将 [batch_size, num_queries, ...] 的形状变为 [batch_size * num_queries, ...]。
            # 这是一个提效技巧，使得我们可以用一次大的矩阵运算，同时计算整个批次中所有预测和所有真实物体的配对代价。
        # outputs["pred_logits"] 的形状是 [batch_size, num_queries, num_classes]->[batch_size*num_queries, num_classes]
        # outputs["pred_boxes"] 的形状是 [batch_size, num_queries, 4]->[batch_size*num_queries, 4]
        # softmax(-1): 将模型的原始输出（logits）转换为概率分布。
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # 2. 拼接所有真实目标
        # 将一个批次中所有图像的真实标签和真实框拼接成一个长长的一维列表。
        # 例如，如果批次里有2张图，第一张有3个物体，第二张有5个，那么 total_num_targets 就是 8。
        # targets = [
        #     {
        #         "labels": tensor([0, 1]), # 真实类别: cat, dog
        #         "boxes": tensor([[0.25, 0.25, 0.1, 0.1],  # 真实框0 (cat)
        #                         [0.55, 0.55, 0.2, 0.2]]) # 真实框1 (dog)
        #     }
        # ]（长度为 bs=1）

        tgt_ids = torch.cat([v["labels"] for v in targets]) # shape: [total_num_targets], 即[2]
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # shape: [total_num_targets, 4]，即[2, 4]

        # 3. 计算匹配代价矩阵的各个部分
        # 这是一个预测 vs 真实的过程。我们的目标是构建一个成本矩阵 C
        # 行（ROWS）：表示4个模型
        # 列（COLUMNS）：表示2个真实物体
        # C.shape:[4, 2]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # 3a. 分类代码（Classification Cost）
        # 取出列向量，也就是根据真实框类别的索引，取出对应的预测概率
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        # 3b. 边界框损失（Bounding Box Cost）
        # 计算预测框和真实框之间的L1距离。这个距离越小，说明预测框和真实框越接近。
        # torch.cdist(A, B, p=1): 这个 PyTorch 函数专门用于计算两组向量之间的距离。
        # p=1 代表计算 L1 距离（曼哈顿距离），即 ∑|a_i - b_i|。
        # cost_bbox: 形状也是 [bs*nq, total_num_targets].
        # cost_bbox[i, j] 代表第 i 个预测框与第 j 个真实框之间的 L1 距离。距离越小，代价越低。
        # out_bbox.shape:[4, 4], tgt_bbox.shape:[2, 4], cost_bbox.shape:[4, 2]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # 3c. GIoU损失（Generalized Intersection over Union）
        # 计算预测框和真实框之间的GIoU损失。GIoU损失是IoU损失的改进版，考虑了边界框的尺寸和位置。
        # GIoU 是一种比标准 IoU 更好的度量，即使两个框不重叠，也能提供有意义的距离信息。GIoU 的值域是 [-1, 1]。
        # -generalized_box_iou: GIoU 越大，代表两个框越匹配。因此我们取其相反数，使得 GIoU 越大，代价越低。
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        # # 4. 计算最终代价矩阵并执行匹配
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # .view(bs, num_queries, -1): 将扁平的代价矩阵 C 重新整形回批次的形式，[bs, nq, total_num_targets_in_batch]。
            # 但这里有一个问题，每个 batch 元素的真实目标数是不同的，所以这个 -1 实际上代表了整个批次的总目标数。
        # .cpu(): 匈牙利算法的 scipy 实现要求输入在 CPU 上。
        C = C.view(bs, num_queries, -1).cpu()

        # sizes = [...]: 计算出批次中每张图像的真实物体数量，例如 [3, 5]。
        sizes = [len(v["boxes"]) for v in targets]
        # C.split(sizes, -1): 这是一个关键操作。它将总代价矩阵 C 沿着最后一个维度（目标维度）按照 sizes 进行切分。
            # C 的形状是 [bs, nq, total_targets]。
            # split 后会得到一个元组，第一个元素形状是 [bs, nq, 3]，第二个是 [bs, nq, 5]。
        # enumerate(C.split(...)): 遍历切分后的代价矩阵块。i 是批次索引（0, 1, ...），c 是对应批次的代价矩阵块。
        # linear_sum_assignment(c[i]):
            # 这就是匈牙利算法的核心调用。scipy.optimize.linear_sum_assignment 接收一个代价矩阵（这里是 c[i]，即第 i 个 batch 元素的 [nq, num_targets_in_this_image] 代价矩阵），然后解决线性指派问题 (Linear Sum Assignment Problem, LSAP)。
            # 它返回两组索引 (row_ind, col_ind)，row_ind 是被选中的预测框的索引，col_ind 是与之配对的真实框的索引，并且保证这个配对方案的总代价（sum(cost_matrix[row_ind, col_ind])）是所有可能配对中最小的。
        # indices: 存储了每个 batch 元素的匹配结果。
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # 返回结果
        # 将 scipy 输出的 numpy 数组转换回 PyTorch 张量，并整理成一个列表返回。
        # 列表的每个元素是一个元组 (prediction_indices, target_indices)，对应批次中一张图像的匹配结果。
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
