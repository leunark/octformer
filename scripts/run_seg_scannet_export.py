from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import numpy as np


#ALIAS = "scannet"
#INPUT_DIR = f"data/{ALIAS}.npz/train"
#LABEL_DIR = f"logs/scannet/octformer_val_seg_{ALIAS}"
#OUTPUT_DIR = f"logs/scannet/octformer_val_seg_{ALIAS}_ply"
#COUNT = 50

ALIAS = "custom"
INPUT_DIR = f"data/{ALIAS}.npz/test"
LABEL_DIR = f"logs/scannet/octformer_test_seg_{ALIAS}"
OUTPUT_DIR = f"logs/scannet/octformer_test_seg_{ALIAS}_ply"
COUNT = 10


def get_fixed_label_colors():
    """
    Generates 21 distinct colors for labels 0-20.
    Uses matplotlib's 'tab10' colormap for distinct colors.
    """
    colormap = plt.get_cmap("tab10", 21)  # Generate 21 fixed colors
    colors = (colormap(np.arange(21))[:, :3] * 255).astype(np.uint8)  # Convert to RGB 0-255
    return colors

def save_ply_with_labels(points, normals, labels, filename="output.ply"):
    """
    Saves a point cloud with normals and fixed colors per label to a binary .ply file using `plyfile`.

    :param points: (N, 3) numpy array of 3D points
    :param normals: (N, 3) numpy array of normals
    :param labels: (N,) numpy array of integer labels (0 to 20)
    :param filename: Output file name
    """
    assert points.shape[1] == 3, "Points should have shape (N,3)"
    assert normals.shape[1] == 3, "Normals should have shape (N,3)"
    assert points.shape[0] == labels.shape[0], "Labels should match the number of points"

    # Ensure labels are in range 0-20
    labels = np.clip(labels, 0, 20)

    # Get fixed colors
    label_colors = get_fixed_label_colors()
    colors = label_colors[labels]  # Assign colors based on labels

    # Create structured array for PLY format (BINARY)
    vertex_data = np.empty(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    # Assign values (NumPy vectorized)
    vertex_data['x'], vertex_data['y'], vertex_data['z'] = points.T
    vertex_data['nx'], vertex_data['ny'], vertex_data['nz'] = normals.T
    vertex_data['red'], vertex_data['green'], vertex_data['blue'] = colors.T

    # Create PLY element
    el = PlyElement.describe(vertex_data, 'vertex')

    # 🚀 Write in **binary format** (much faster than ASCII)
    PlyData([el], text=False, byte_order='<').write(filename)
    print(f"✅ Saved {filename} (Binary PLY, FAST)")


def main():
    input_dir = Path(INPUT_DIR)
    label_dir = Path(LABEL_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    label_list = sorted(list(label_dir.glob("*.txt")))
    for label_path in label_list[:COUNT]:
        input_path = input_dir / (label_path.stem + ".npz")
        data = np.load(input_path)
        points = data['points']
        points = points[:, [0,2,1]]
        points[:,2] *= -1
        normals = data['normals']
        colors = data['colors']
        labels = np.loadtxt(label_path, dtype=np.int32)
        output_path = output_dir / (label_path.stem + ".ply")
        save_ply_with_labels(points, normals, labels, output_path)

if __name__ == '__main__':
  main()