# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import datasets
import models
from builder import get_segmentation_dataset

# Import the original builder functions to extend them
from builder import get_segmentation_model as original_get_segmentation_model


def octsegformer_crossattn(in_channels, out_channels, **kwargs):
  return models.OctFormerSegCrossAttn(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 18, 2],
      num_heads=[6, 12, 24, 24],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octsegformer_crossattn_large(in_channels, out_channels, **kwargs):
  return models.OctFormerSegCrossAttn(
      in_channels, out_channels,
      channels=[192, 384, 768, 768],
      num_blocks=[2, 2, 18, 2],
      num_heads=[12, 24, 48, 48],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octsegformer_crossattn_small(in_channels, out_channels, **kwargs):
  return models.OctFormerSegCrossAttn(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 6, 2],
      num_heads=[6, 12, 24, 24],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def get_segmentation_model(flags):
  """
  Extended version of get_segmentation_model that includes cross attention models.
  """
  params = {
      'in_channels': flags.channel, 'out_channels': flags.nout,
      'interp': flags.interp, 'nempty': flags.nempty,
  }
  
  # Dictionary mapping model names to their builder functions
  networks = {
      # Original models
      'octsegformer': models.octsegformer if hasattr(models, 'octsegformer') 
                       else None,
      'octsegformer_large': models.octsegformer_large if hasattr(models, 'octsegformer_large') 
                            else None,
      'octsegformer_small': models.octsegformer_small if hasattr(models, 'octsegformer_small') 
                            else None,
      
      # New cross attention models
      'octsegformer_crossattn': octsegformer_crossattn,
      'octsegformer_crossattn_large': octsegformer_crossattn_large,
      'octsegformer_crossattn_small': octsegformer_crossattn_small,
  }
  
  # Check if the requested model is one of cross attention models
  if flags.name.lower() in ['octsegformer_crossattn', 'octsegformer_crossattn_large', 
                          'octsegformer_crossattn_small']:
    # Return the appropriate cross attention model
    return networks[flags.name.lower()](**params)
  else:
    # Fall back to the original implementation for other models
    return original_get_segmentation_model(flags)
