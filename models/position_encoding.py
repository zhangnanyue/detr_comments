# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor

# 基于正弦/余弦函数的固定编码，是 "Attention Is All You Need" 原始方法的二维泛化版。
# 动机：原始的 Transformer 用于处理一维的文本序列。它使用正弦和余弦函数为每个位置 pos（一个整数）生成一个唯一的向量。
# 挑战: 图像是二维的，一个像素点的位置由 (x, y) 两个坐标决定。
# 解决方案: 将一维思想进行扩展。分别计算 x 方向和 y 方向的位置编码，然后将它们拼接 (concatenate) 起来，共同构成最终的位置编码向量。
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # num_pos_feats: 这是分配给单个方向（比如 x 方向）的位置编码的特征维度。最终总的位置编码维度将是 2 * num_pos_feats。在 DETR 中，它通常是 hidden_dim / 2。
    # temperature: 就是原始 Transformer 公式中的那个大常数 10000，用于控制三角函数的频率范围。
    # normalize: 一个布尔值，决定是否要将原始的像素坐标（例如 0 到 W-1）归一化到 [0, 2π] 的范围内。归一化有助于模型处理不同尺寸的图像，是 DETR 默认开启的选项。
    # scale: 归一化后乘以的缩放常数。默认为 2π，因为三角函数的周期是 2π。
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # PE(pos, 2i) = sin(pos / 10000^(2i/d))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors     # (N, C, H, W) 图像张量
        mask = tensor_list.mask     # (N, H, W) 掩码张量，True 代表 padding
        assert mask is not None
        # # 反转 mask，现在 True 代表有效像素，~是按位取反操作。
        not_mask = ~mask            

        # 1. 计算每个像素的绝对坐标
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成一个序列[0, 1, 2, ..., num_pos_feats-1]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t // 2: // 是整除。这会产生序列 [0, 0, 1, 1, 2, 2, ...]。这是为了让正弦和余弦函数使用相同的频率 i。
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# 一种可学习的绝对位置编码。
# 它不使用固定的数学函数，而是创建两个嵌入表（Embedding Table），让模型自己去学习每个位置应该对应什么样的编码向量。
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # nn.Embedding 是一个简单的查找表。
        # 50 是最大可接受的位置索引。这意味着它假设特征图的最大高度和宽度不会超过50。这是一个硬编码的限制。
        # num_pos_feats 是为每个位置索引学习的向量维度。
        # y 坐标的嵌入表
        self.row_embed = nn.Embedding(50, num_pos_feats)
        # # x 坐标的嵌入表
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        # 获取输入特征图的高和宽
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        # 对 [0, ..., w-1] 中的每个整数，去 col_embed 表里查找对应的 D 维向量。
        x_emb = self.col_embed(i)
        # 同理，查找 y 坐标的嵌入。
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

# 一个根据配置选择使用哪种方式的工厂函数 build_position_encoding。
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
