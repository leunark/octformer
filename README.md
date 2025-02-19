# NOTES
Extension of the README.md for documentation purposes.

## Docker

### Using Devcontainer (Recommended)
This project includes a `.devcontainer` configuration for quick setup with [VS Code](https://code.visualstudio.com/).
1. Install [Docker](https://www.docker.com/) and the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VS Code extension.
2. Reopen the folder in a container:
    - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), and run `Remote-Containers: Reopen in Container`.
3. Start coding in the pre-configured environment!

### Manually Build & Run
**Run the container**:
- Running the container requires you to attach the gpus i.e. `--gpus all`
- Enough shared memory has to be provided e.g. with `--shm-size=64g` or preferrred `--ipc=host` for all
```bash
docker build -f .devcontainer/Dockerfile -t octformer .
docker run --gpus all --rm -it -v $(pwd):/workspace/octformer --ipc=host octformer bash
```

## ScanNet
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
If an issue persists and you’re working in a trusted environment (e.g., testing or internal networks), you can disable SSL certificate verification to bypass it. (
    
Add to `download-scannet.py`:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Fast compression
To move files from machine to machine, a fast approach is needed to archive files.
Make sure zstd is intalled with `apt-get update && apt-get install -y zstd`
```bash
tar --use-compress-program=zstd -cvf scannet.tar.gz data/ScanNet
```
And this is how to extract it again.
```bash
tar --use-compress-program=zstd -xvf scannet.tar.gz
```

### ScanNet in Docker
Run the training in the docker container as daemon**
```bash
docker run --gpus all --rm -dit -v $(pwd):/workspace/octformer --ipc=host --name octformer_container octformer bash -c "
  conda info &&
  nvidia-smi &&
  python scripts/run_seg_scannet.py --gpu 0 --alias scannet --port 10001
"
```
While the container is running in the background, you can see the logs with `docker logs -f octformer_container`.

***Note: For cleaner logs, set `SOLVER.progress_bar False` in `scripts/run_seg_scannet.py`.***

### ScanNet200 in Docker
**Prepare ScanNet200**
```bash
docker run --gpus all --rm -dit -v $(pwd):/workspace/octformer --ipc=host --name octformer_container octformer bash -c "
  conda info &&
  nvidia-smi &&
  python tools/seg_scannet.py --run process_scannet --path_in data/ScanNet --path_out data/scanet200.npz  --align_axis  --scannet200
"
```
**Train ScanNet200**
```bash
docker run --gpus all --rm -d -v $(pwd):/workspace/octformer --ipc=host --name octformer_container octformer bash -c "
  conda info && \
  nvidia-smi && \
  python scripts/run_seg_scannet200.py --gpu 0 --alias scannet200
"
```

## Custom Data

1. **Data**: Place the raw custom ply data with exported normals and unit meters as `.ply`
    into `data/Custom/scans_test`. Now process all those files with the following command.
    All .ply files will be exported in the same way scannet is exported to `.npz` files.
    Run `tools/seg_scannet_custom.py` with **F5** to make sure all paths are set accordingly
    with `.vscode/launch.json`.

2. **Inference**: For inference, now run the script `scripts/run_seg_scannet_custom.py` with F5.

3. **Visualize**: Export the inference results as `.ply` file with `scripts/run_seg_scannet_export.py`.
    Make sure the right alias and other parameters are set for custom dataset.
