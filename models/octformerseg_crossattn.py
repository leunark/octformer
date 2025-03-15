# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn
from typing import Dict, List
from ocnn.octree import Octree

# Import from original OctFormer
from .octformer import OctFormer, OctreeT
from .octformer_crossattn import OctFormerCrossBlock

class SegHeaderCrossAttn(torch.nn.Module):
    """
    Segmentation header that uses cross attention to enhance feature fusion
    between different levels of the feature hierarchy.
    """
    def __init__(
            self, out_channels: int, channels: List[int], fpn_channel: int,
            nempty: bool, num_up: int = 1, dropout: List[float] = [0.0, 0.0],
            patch_size: int = 32, num_heads: int = 6, dilation: int = 1):
        super().__init__()
        self.num_up = num_up
        self.num_stages = len(channels)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation

        # 1x1 convolutions to project features to the same dimension
        self.conv1x1 = torch.nn.ModuleList([torch.nn.Linear(
            channels[i], fpn_channel) for i in range(self.num_stages-1, -1, -1)])
        
        # Upsampling module
        self.upsample = ocnn.nn.OctreeUpsample('nearest', nempty)
        
        # 3x3 convolutions for feature refinement
        self.conv3x3 = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            fpn_channel, fpn_channel, kernel_size=[3],
            stride=1, nempty=nempty) for i in range(self.num_stages)])
        
        # Cross attention blocks for feature fusion
        self.cross_attn_blocks = torch.nn.ModuleList([
            OctFormerCrossBlock(
                dim=fpn_channel, num_heads=num_heads, patch_size=patch_size,
                dilation=dilation, nempty=nempty)
            for _ in range(self.num_stages - 1)
        ])
        
        # Upsampling convolutions for the final output
        self.up_conv = torch.nn.ModuleList([ocnn.modules.OctreeDeconvBnRelu(
            fpn_channel, fpn_channel, kernel_size=[3],
            stride=2, nempty=nempty) for i in range(self.num_up)])
        
        # Interpolation for query points
        self.interp = ocnn.nn.OctreeInterp('nearest', nempty)
        
        # Final classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout[0]),
            torch.nn.Linear(fpn_channel, fpn_channel),
            torch.nn.BatchNorm1d(fpn_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout[1]),
            torch.nn.Linear(fpn_channel, out_channels),)

    def forward(self, features: Dict[int, torch.Tensor], octree: Octree,
                query_pts: torch.Tensor):
        """
        Forward pass for the segmentation header.
        
        Args:
            features: Dictionary mapping octree depths to feature tensors
            octree: Octree object for spatial structure
            query_pts: Query points for interpolation
        """
        depth = min(features.keys())
        depth_max = max(features.keys())
        assert self.num_stages == len(features)

        # Convert octree to OctreeT for cross attention
        octree_t = OctreeT(octree, self.patch_size, self.dilation, True,
                          max_depth=depth_max, start_depth=depth)

        # Process the deepest features first
        feature = self.conv1x1[0](features[depth])
        conv_out = self.conv3x3[0](feature, octree, depth)
        out = self.upsample(conv_out, octree, depth, depth_max)
        
        # Process each level and apply cross attention
        for i in range(1, self.num_stages):
            depth_i = depth + i
            
            # Process current level features
            curr_feature = self.conv1x1[i](features[depth_i])
            
            # Upsample previous level features
            up_feature = self.upsample(feature, octree, depth_i - 1, depth_i)
            
            # Apply cross attention between current and upsampled features
            if i - 1 < len(self.cross_attn_blocks):
                # Use cross attention for feature fusion
                enhanced_feature = self.cross_attn_blocks[i-1](
                    curr_feature, up_feature, octree_t, depth_i)
                # Combined feature with residual connection
                feature = enhanced_feature + curr_feature
            else:
                feature = curr_feature + up_feature
                
            # Apply convolution and contribute to output
            conv_out = self.conv3x3[i](feature, octree, depth_i)
            out = out + self.upsample(conv_out, octree, depth_i, depth_max)

        # Final upsampling and classification
        for i in range(self.num_up):
            out = self.up_conv[i](out, octree, depth_max + i)
        out = self.interp(out, octree, depth_max + self.num_up, query_pts)
        out = self.classifier(out)
        return out


class OctFormerSegCrossAttn(torch.nn.Module):
    """
    OctFormer for segmentation with cross attention.
    This model uses the original OctFormer backbone but replaces the
    segmentation header with one that uses cross attention for feature fusion.
    """
    def __init__(
            self, in_channels: int, out_channels: int,
            channels: List[int] = [96, 192, 384, 384],
            num_blocks: List[int] = [2, 2, 18, 2],
            num_heads: List[int] = [6, 12, 24, 24],
            patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
            nempty: bool = True, stem_down: int = 2, head_up: int = 2,
            fpn_channel: int = 168, head_drop: List[float] = [0.0, 0.0], **kwargs):
        super().__init__()
        
        # Use the original OctFormer backbone
        self.backbone = OctFormer(
            in_channels, channels, num_blocks, num_heads, patch_size, dilation,
            drop_path, nempty, stem_down)
        
        # Use the new segmentation header with cross attention
        self.head = SegHeaderCrossAttn(
            out_channels, channels, fpn_channel, nempty, head_up, head_drop,
            patch_size=patch_size, num_heads=num_heads[-1] // 2, dilation=dilation)
        
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int,
                query_pts: torch.Tensor):
        """
        Forward pass for the segmentation model.
        
        Args:
            data: Input feature tensor
            octree: Octree object for spatial structure
            depth: Depth of the octree
            query_pts: Query points for interpolation
        """
        # Extract hierarchical features using the backbone
        features = self.backbone(data, octree, depth)
        
        # Apply the segmentation header with cross attention
        output = self.head(features, octree, query_pts)
        
        return output
