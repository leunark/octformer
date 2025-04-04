
# Debugging Summary: Multi-GPU Training with NCCL

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

