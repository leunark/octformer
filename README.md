# OctFormer: Octree-based Transformers for 3D Point Clouds
Extension of the original README.md (now README_ORIGINAL.md) for documentation purposes.

## 1. Setup Environment - Docker

### Using Devcontainer (Recommended)
This project includes a `.devcontainer` configuration for quick setup with [VS Code](https://code.visualstudio.com/).
1. Install [Docker](https://www.docker.com/) and the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VS Code extension.
2. Reopen the folder in a container:
    - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), and run `Remote-Containers: Reopen in Container`.
3. Start training in the pre-configured environment, by following with ScanNet data preparation and then 
W
### Manually Build & Run
**Run the container**:
- Running the container requires you to attach the gpus i.e. `--gpus all`
- Enough shared memory has to be provided e.g. with `--shm-size=64g` or preferrred `--ipc=host` for all
```bash
docker build -f .devcontainer/Dockerfile -t octformer .
docker run --gpus all --rm -it -v $(pwd):/workspace/octformer --ipc=host octformer bash
```

## 2. Data Preparation - ScanNet
To minimize data usage while ensuring you have the necessary files for 3D semantic segmentation using ScanNet, follow these guidelines:

### Required Files
For 3D semantic segmentation, you only need the following files:
1.	Point Cloud Files (_vh_clean_2.ply):
    - Contains the 3D point cloud data for each scene.
    - This is the primary input for 3D segmentation tasks.
2.	Label Files (_vh_clean_2.labels.ply):
    - Contains per-point semantic labels for the 20 or 200 class categories.
3.	Segmentation Metadata (_vh_clean_2.0.010000.segs.json):
    - Provides information for segment-level annotations.
4.	Aggregation Metadata (_vh_clean.aggregation.json):
    - Provides mapping between segments and object categories.
5.	Label Mapping (scannetv2-labels.combined.tsv):
    - Maps raw category names to the 20 or 200 target class labels.
6.  Meta data (.txt):
    - Attributes and meta data like color information, size ...
    - Only needed for `ScanNet200`

### Exclusion of Non-Essential Files
You can skip the following files to save space:
- 2D Data:
    - _2d-instance.zip
    - _2d-instance-filt.zip
    - _2d-label.zip
    - _2d-label-filt.zip
- Raw Sensor Data (.sens):
    - Unless you need raw RGB-D sequences, .sens files can be excluded.
- Visualization Files (_vh_clean.ply):
    - These are simpler versions of the 3D point clouds, unnecessary for semantic segmentation.
- Preprocessed Frames:
    - Avoid scannet_frames_25k.zip or scannet_frames_test.zip if not using pre-extracted image sequences.

### Command to Download Only Necessary Files

You can specify the exact file types to download using the --type argument in the download script. For example:
```bash
python tools/download-scannet.py -o ./data/ScanNet --type _vh_clean_2.ply
python tools/download-scannet.py -o ./data/ScanNet --type _vh_clean_2.labels.ply
python tools/download-scannet.py -o ./data/ScanNet --type _vh_clean_2.0.010000.segs.json
python tools/download-scannet.py -o ./data/ScanNet --type _vh_clean.aggregation.json
python tools/download-scannet.py -o ./data/ScanNet --type .txt
python tools/download-scannet.py -o ./data/ScanNet --label_map
```

This setup ensures you only download the files needed for 3D segmentation.

### Directory Structure
After downloading, your dataset directory should look like this:
```
scannet_data/
  scans/
    <scene_id>/
      *_vh_clean_2.ply
      *_vh_clean_2.labels.ply
      *_vh_clean_2.0.010000.segs.json
      *_vh_clean.aggregation.json
      *.txt
  tasks/
    scannetv2-labels.combined.tsv
```

***Total Estimate: ~21 GB (significantly smaller than the full 1.2 TB)***
***After pre-processing: ~9.5 GB***

### Certificate Error
If an issue persists and you’re working in a trusted environment (e.g., testing or internal networks), you can disable SSL certificate verification to bypass it.
    
Add to `download-scannet.py`:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## 3. Training & Evaluation

### ScanNet Segmentation

1. **Data**: Download the data from the
   [ScanNet benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/) like mentioned before.
   Unzip the data and place it to the folder <scannet_folder> e.g. *data/ScanNet*. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet.py --run process_scannet --path_in data/ScanNet
    ```
    This will generate a processed dataset into the folder *data/scannet.npz*

2. **Train**: Run the following command to train the network with a single GPU and
   port 10001. The mIoU on the validation set without voting is 74.8. The
   training takes about 3 days on 4 Nvidia Tesla T4 GPUs which can be done with `--gpu 0,1,2,3`.

    ```bash
    python scripts/run_seg_scannet.py --gpu 0 --alias scannet --port 10001
    ```

3. **Evaluate**: Run the following command to get the per-point predictions for
   the validation dataset with a voting strategy. And after voting, the mIoU is
   76.3 on the validation dataset.

    ```bash
    python scripts/run_seg_scannet.py --gpu 0 --alias scannet --run validate
    ```


### Custom Data

1. **Data**: Place the raw custom ply data with exported normals and unit meters as `.ply`
    into `data/Custom/scans_test`. Now process all those files with the following command.
    All .ply files will be exported in the same way scannet is exported to `.npz` files.
    Run `tools/seg_scannet_custom.py` with **F5** to make sure all paths are set accordingly
    with `.vscode/launch.json`.

2. **Inference**: For inference, now run the script `scripts/run_seg_scannet_custom.py` with F5.

3. **Visualize**: Export the inference results as `.ply` file with `scripts/run_seg_scannet_export.py`.
    Make sure the right alias and other parameters are set for custom dataset.

