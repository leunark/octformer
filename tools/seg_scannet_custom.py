from __future__ import annotations
from tools.seg_scannet import save_npz
from plyfile import PlyData
import numpy as np
from typing import List
import pandas as pd
import tqdm
from pathlib import Path

PATH_IN = "data/Custom"
PATH_OUT = "data/custom.npz"

def read_ply(filename: str) -> List[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  plydata = PlyData.read(filename)
  vertex = plydata['vertex'].data
  #print(f"Detected names: {vertex.dtype.names}")
  points = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(np.float32)
  colors = np.stack((vertex["red"], vertex["green"], vertex["blue"]), axis=1).astype(np.float32)
  normals = np.stack((vertex["nx"], vertex["ny"], vertex["nz"]), axis=1).astype(np.float32)
  labels = np.full(len(points), 255, dtype=np.int32)
  return points, normals, colors, labels

def process_scannet():
  path_in = Path(PATH_IN)
  path_out = Path(PATH_OUT)
  test_path_out = path_out / "test"
  test_path_out.mkdir(exist_ok=True)
  test_path_in = path_in / "scans_test"
  file_names = []
  for path in test_path_in.glob("*.ply"):
    points, normals, colors, labels = read_ply(path)
    output_path = test_path_out / (path.stem + ".npz")
    np.savez(output_path, points=points, normals=normals, colors=colors, labels=labels)
    file_names.append(output_path.name)
  
  with open(path_out / "custom_test_npz.txt", "w") as f:
    for file_name in file_names:
      f.write(f"{file_name} 0\n")
      print(f"Preprocessed file {file_name}")



def main():
  process_scannet()

if __name__ == "__main__":
  main()
