# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import shlex
import sys
from pathlib import Path
import numpy as np

np.set_printoptions(suppress=True)


run = "test"
alias = "custom"
gpu = "0"
port = "10001"


def execute_command(cmds):
  cmds = " ".join(cmds)
  argv = shlex.split(cmds)
  argv = argv[1:]
  sys.argv = argv
  print('Execute Args: \n' + str(argv) + '\n')
  script = Path(argv[0]).stem
  if script == "segmentation":
    from segmentation import SegSolver
    SegSolver.main()
  elif script == "seg_scannet":
    from tools.seg_scannet import generate_output_seg
    generate_output_seg()


def test():
  # get the predicted probabilities for each point
  ckpt = 'logs/scannet/octformer_scannet/best_model.pth'
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'LOSS.mask -255',       # to keep all points
      'SOLVER.gpu  {},'.format(gpu),
      'SOLVER.run evaluate',
      'SOLVER.eval_epoch 1',  # zero-shot prediction
      'SOLVER.alias test_{}'.format(alias),
      'SOLVER.ckpt {}'.format(ckpt),
      'DATA.test.batch_size 1',
      'DATA.test.location', 'data/custom.npz/test',
      'DATA.test.filelist', 'data/custom.npz/custom_test_npz.txt',
      'DATA.test.distort False', ]
  execute_command(cmds)

  # map the probabilities to labels
  cmds = [
      'python tools/seg_scannet.py',
      '--run generate_output_seg',
      '--path_pred logs/scannet/octformer_test_{}'.format(alias),
      '--path_out logs/scannet/octformer_test_seg_{}'.format(alias),
      '--filelist  data/custom.npz/custom_test_npz.txt', ]
  execute_command(cmds)

if __name__ == '__main__':
  eval('%s()' % run)
