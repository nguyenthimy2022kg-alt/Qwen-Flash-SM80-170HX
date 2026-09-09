# CMP 170HX: P2P and direct SSD reads

[简体中文](CMP_P2P.md) | **English** · [Home](../README.en.md)

This project uses two paths: **GPU ↔ GPU for computation data, and SSD → GPU for PLE lookup rows.**

## 1. Choose a driver

If GPU peer transfers and direct SSD reads already work, proceed to [model deployment](DEPLOYMENT.en.md).

For CMP 170HX driver adaptation, start with **[bayley/cmpunlocker at pinned revision `5a7bb4b`](https://github.com/bayley/cmpunlocker/tree/5a7bb4b7e5056306fe49e8b824787659abb19914)**. It already includes BAR1 resizing, P2P mappings, and the `0011`, `0013`, and `0015` patches previously packaged separately here. **Do not apply this repository's old patches on top.**

Follow that revision's installation instructions with matching NVIDIA driver libraries, firmware, and kernel headers, then install and cold boot. Its installer changes host settings but does not complete P2P/GDS setup for every platform; review BAR1, IOMMU, and related settings for your machine. An existing working driver does not need reinstalling.

## What each component does

| Component | Role |
|---|---|
| cmpunlocker driver | Adapts CMP BAR1 and GPU P2P so other devices can access GPU memory directly |
| NVIDIA GDS/cuFile + Linux NVMe | Provides the SSD → GPU direct-read path |
| This project | Reads PLE rows according to model input and distributes them between the GPUs |

BAR1 is the address window through which external devices access VRAM. Resizing determines its capacity; mapping identifies the VRAM locations it exposes.

## 2. Check GPU peer transfers

Run from this project's repository root:

```bash
python3 scripts/check-gds-host.py --data-path /actual/PLE/directory
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -m
```

The inventory script only collects environment information, so `NOT_TESTED` means no transfer test has run. After both directions report `OK`, verify data through actual CUDA peer copies; the [CUDA Samples P2P test](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest) can check bandwidth and latency.

## 3. Configure direct SSD reads

Follow the [NVMe GDS guide](GDS_NVME_P2PDMA_REPRODUCTION.en.md) for cuFile configuration and strict data validation. SSD direct reads require their own validation after GPU P2P passes.

The project's data flow is:

```text
PLE data on SSD → GDS direct read → Reading GPU → P2P broadcast → Other GPU
```

The first `gpu_ids` entry in `config/local.json` selects the reading GPU. The reference host uses H12D + EPYC 7532, with no PCIe switch on the SSD and GPU paths; see the [hardware and connectivity diagram](REFERENCE_HARDWARE.en.md) when choosing GPUs and connections.
