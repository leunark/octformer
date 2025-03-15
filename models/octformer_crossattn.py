# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn
from typing import Optional, List

# Import from original OctFormer
from .octformer import OctreeT, MLP, RPE, OctreeDWConvBn

class OctreeCrossAttention(torch.nn.Module):
    """
    Cross attention mechanism for OctFormer.
    Unlike self-attention where Q, K, V all come from the same source,
    cross attention allows Q to come from one source while K, V come from another.
    This enables learning relationships between different feature representations.
    """
    def __init__(self, dim: int, patch_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 dilation: int = 1, use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        # Separate projections for query and key-value
        self.q_proj = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = torch.nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

        # Relative positional encoding
        self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None

    def forward(self, q_data: torch.Tensor, kv_data: torch.Tensor, 
                octree: OctreeT, depth: int):
        """
        Forward pass for cross attention.
        
        Args:
            q_data: Query data tensor
            kv_data: Key/value data tensor
            octree: OctreeT object for spatial structure
            depth: Depth of the data in the octree
        """
        H = self.num_heads
        K = self.patch_size
        C = self.dim
        D = self.dilation

        # Patch partition for queries
        q_data = octree.patch_partition(q_data, depth)
        if D > 1:  # dilation
            rel_pos = octree.dilate_pos[depth]
            mask = octree.dilate_mask[depth]
            q_data = q_data.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)
        else:
            rel_pos = octree.rel_pos[depth]
            mask = octree.patch_mask[depth]
        q_data = q_data.view(-1, K, C)

        # Patch partition for keys and values
        kv_data = octree.patch_partition(kv_data, depth)
        if D > 1:  # dilation
            kv_data = kv_data.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)
        kv_data = kv_data.view(-1, K, C)

        # Query, key, value projections
        q = self.q_proj(q_data).reshape(-1, K, H, C // H).permute(0, 2, 1, 3)  # (N, H, K, C')
        kv = self.kv_proj(kv_data).reshape(-1, K, 2, H, C // H).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (N, H, K, C')
        q = q * self.scale

        # Attention calculation
        attn = q @ k.transpose(-2, -1)  # (N, H, K, K)
        attn = self.apply_rpe(attn, rel_pos)  # (N, H, K, K)
        attn = attn + mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        data = (attn @ v).transpose(1, 2).reshape(-1, C)

        # Patch reverse
        if D > 1:  # dilation
            data = data.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)
        data = octree.patch_reverse(data, depth)

        # Final projection
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            attn = attn + self.rpe(rel_pos)
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
                self.dim, self.patch_size, self.num_heads, self.dilation)  # noqa


class OctFormerCrossBlock(torch.nn.Module):
    """
    Transformer block that uses cross attention instead of self-attention.
    This allows the block to learn relationships between different feature representations.
    """
    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                activation: torch.nn.Module = torch.nn.GELU, **kwargs):
        super().__init__()
        # Separate normalizations for query and key-value
        self.norm1_q = torch.nn.LayerNorm(dim)
        self.norm1_kv = torch.nn.LayerNorm(dim)
        
        # Cross attention layer
        self.cross_attention = OctreeCrossAttention(
            dim, patch_size, num_heads, qkv_bias,
            qk_scale, attn_drop, proj_drop, dilation)
        
        # Standard components (normalization, MLP, dropout)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        
        # Conditional positional encoding for query and key-value
        self.cpe_q = OctreeDWConvBn(dim, nempty=nempty)
        self.cpe_kv = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, q_data: torch.Tensor, kv_data: torch.Tensor, 
                octree: OctreeT, depth: int):
        """
        Forward pass for cross attention block.
        
        Args:
            q_data: Query data tensor
            kv_data: Key/value data tensor
            octree: OctreeT object for spatial structure
            depth: Depth of the data in the octree
        """
        # Apply conditional positional encoding
        q_data = self.cpe_q(q_data, octree, depth) + q_data
        kv_data = self.cpe_kv(kv_data, octree, depth) + kv_data
        
        # Apply cross attention
        attn = self.cross_attention(
            self.norm1_q(q_data), self.norm1_kv(kv_data), octree, depth)
        
        # Apply residual connection and dropout
        q_data = q_data + self.drop_path(attn, octree, depth)
        
        # Apply MLP, normalization, residual connection and dropout
        ffn = self.mlp(self.norm2(q_data))
        q_data = q_data + self.drop_path(ffn, octree, depth)
        
        return q_data
