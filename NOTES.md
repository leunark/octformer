# NOTES
Extension of the README.md for documentation purposes.

## Installation on Ubuntu 22.04 LTS
A cuda compiler nvcc is required. You might need to additionally install it before.
```bash
conda install -c conda-forge cudatoolkit-dev=11.3
```

Cuda 11.3 supports up to g++<10.0.0. Therefore, e.g if you're on Ubuntu 22, install the correct c++ compiler.
```bash
conda install -c conda-forge gxx_linux-64==9.5.0
```

Also, crpyto lib might be needed.
```bash
conda install -c conda-forge libxcrypt
```

## Docker
Set `SOLVER.dist_url tcp://127.0.0.1:10009` to avoid a socker warning because of default value for `localhost`.

### Using Devcontainer
This project includes a `.devcontainer` configuration for quick setup with [VS Code](https://code.visualstudio.com/).
1. Install [Docker](https://www.docker.com/) and the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VS Code extension.
2. Reopen the folder in a container:
    - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), and run `Remote-Containers: Reopen in Container`.
3. Start coding in the pre-configured environment!

### Build & Run
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
```bash
tar --use-compress-program=zstd -cvf scannet.tar.gz data/ScanNet
```

### ScanNet in Docker
Run the training in the docker container as daemon**
```bash
docker run --gpus all --rm -dit -v $(pwd):/workspace/octformer --ipc=host --name octformer_container octformer bash -c "
  conda info &&
  nvidia-smi &&
  pip install -r requirements.txt &&
  export NCCL_P2P_DISABLE=1 &&
  python scripts/run_seg_scannet.py --gpu 0,1,2,3 --alias scannet --port 10001
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
  pip install -r requirements.txt &&
  export NCCL_P2P_DISABLE=1 &&
  python tools/seg_scannet.py --run process_scannet --path_in data/ScanNet --path_out data/scanet200.npz  --align_axis  --scannet200
"
```
**Train ScanNet200**
```bash
docker run --gpus all --rm -d -v $(pwd):/workspace/octformer --ipc=host --name octformer_container octformer bash -c "
  conda info && \
  nvidia-smi && \
  pip install -r requirements.txt && \
  export NCCL_P2P_DISABLE=1 &&
  python scripts/run_seg_scannet200.py --gpu 0,1,2,3 --alias scannet200
"
```


## Debugging Summary: Multi-GPU Training with NCCL

**Initial Issue:**

- Multi-GPU training hung after initializing DistributedDataParallel (DDP) with NCCL.
- Setup: Single-node with 4 NVIDIA T4 GPUs, CUDA 11.3, and PyTorch.
- Goal: Resolve hanging and optimize training across GPUs.

**Debugging Process (Ordered Chronologically)**

1. **Start with NCCL Debugging**
- Observation: The training script hung during DDP(model) initialization.
- Action: Enabled NCCL debugging to inspect communication initialization:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```
- Result: NCCL initialized successfully but hung afterward, indicating a possible communication or synchronization issue.
The hanging gpus often keep stuck even after closing python. To overcome this kill all python processes or `sudo reboot`.
```bash
nvidia-smi | grep python | awk '{print $5}' | xargs -I {} sudo kill -9 {}
```

2. **Test Alternative Backends**
- Observation: To isolate whether the issue was specific to NCCL, switched to gloo:
```python
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
```
- Result: Gloo worked without issues and achieved a 4x speedup on 4 GPUs. This indicated the problem was NCCL-specific.

3. **Verify Environment Configuration**
- Action: Verified software versions and compatibility:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.nccl.version())"
nvidia-smi
```
- Result: All versions were compatible, confirming no immediate software mismatch.

4. **Check Network Interfaces**
- Observation: Suspected network misconfiguration due to multiple interfaces.
- Action: Inspected network interfaces with `ifconfig`
***Found***:
- ens4f0 (primary interface)
- team0 (network team)
- docker0 and virbr0 (irrelevant virtual interfaces).
- Solution: Restricted NCCL to the primary interface:
```bash
export NCCL_SOCKET_IFNAME=ens4f0
```
- Result: Did not resolve the hang but avoided network-related conflicts.

5. **Investigate GPU Topology**
- Action: Checked GPU interconnect topology with:
```bash
nvidia-smi topo -m
```
- Result:
    - GPUs connected via SYS (system interconnect), meaning no direct P2P paths like NVLink or PIX.
    - GPUs were distributed across different NUMA nodes, adding potential latency.

6. **Disable P2P Communication**
- Observation: Since GPUs lacked direct P2P connectivity (SYS interconnect), P2P communication likely caused the hang.
- Action: Disabled P2P communication:
```bash
export NCCL_P2P_DISABLE=1
```
- Result: NCCL worked correctly across GPUs, resolving the hang.

**Key Learnings from the Debugging Process**
1. Isolate the Backend First:
    - Switching to gloo helped confirm the issue was NCCL-specific, allowing focused debugging.
2. Inspect Topology Earlier:
    - Running nvidia-smi topo -m sooner could have revealed the SYS interconnect and NUMA distribution issues earlier.
3. Understand P2P Dependencies:
    - Disabling P2P (NCCL_P2P_DISABLE=1) worked because of the lack of direct inter-GPU paths in your system.
4. Restrict Network Interfaces:
    - Setting NCCL_SOCKET_IFNAME ensured no conflicts from virtual interfaces like docker0 or virbr0.

