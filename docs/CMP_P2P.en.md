# CMP 170HX P2P deployment reference

[简体中文](CMP_P2P.md) | **English** · [Home](../README.en.md)

The reference environment uses [qg19932GH/cmpunlocker](https://github.com/qg19932GH/cmpunlocker) for GPU P2P support. This operates at the driver layer, independently of vLLM. This repository does not include its driver patches or install them during image builds or service startup.

The upstream commit reviewed on 2026-09-08 was [`aaddfd4ce84a2804a7e0cd332acc4c26c79063d9`](https://github.com/qg19932GH/cmpunlocker/tree/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9). This identifies the documentation review snapshot; **it has not been confirmed as the commit originally used to install the reference machine's driver**.

## Scope

The host must meet the [host acceptance criteria](DEPLOYMENT.en.md#host-acceptance-criteria). cmpunlocker and the local overlay described below are adaptations used on the reference machine, not mandatory vLLM dependencies. A machine with working P2P/GDS does not need the same patches. Choose an implementation appropriate for the hardware and validate actual data transfers and model execution.

There is no PCIe switch on this machine’s GPU/SSD paths. See [reference hardware and PCIe topology](REFERENCE_HARDWARE.en.md) for the CPU, motherboard, SSD, root-port layout, and reading-GPU selection.

## Observed reference configuration

Read from the machine hosting the service on 2026-09-08:

| Item | Active configuration |
|---|---|
| Linux / NVIDIA | `6.17.0-23-generic` / `610.43.03` Open Kernel Module |
| Driver build | Local CMP v0.3 overlay with additional BAR1/P2P patches |
| Two-GPU capability matrix | P2P reads and writes reported `OK` in both directions |
| IOMMU | `amd_iommu=off iommu=off`; no IOMMU instances |
| NVMe multipath | `N` |
| PLE filesystem | ext4, with `data=ordered` in the actual mount options |
| NVIDIA parameter file | `/etc/modprobe.d/cmpunlocker-nvidia-options.conf` |
| BAR1/P2P | `EnableResizableBar=1`, `RMForceStaticBar1=1`, `RMPcieP2PType=1`, `RmForceDisableIomapWC=1` |
| This machine's Gen2 settings | `RmForceEnableGen2=1`, `RMPcieLinkSpeed=0x1` |

The reference driver build script also uses `driver/local-src` and `driver/local-patches`, including `0011-p2p-bar1.patch`, `0013-skip-mailbox-peer-preinit.patch`, and `0015-bar1p2p-readcap-override.patch`. These local overlay directories are not present in the external commit reviewed above. The two driver versions must not be treated as identical.

This repository does not distribute the machine-specific overlay or require users to obtain it. These differences document the reference environment's provenance. The goal is working P2P/GDS capabilities, not duplication of this driver build. CMP machines that do not yet meet the requirements need adaptations suited to their own hardware.

## Differences in the external installer

Review of the external version's [README](https://github.com/qg19932GH/cmpunlocker/blob/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9/README.md) and [installer](https://github.com/qg19932GH/cmpunlocker/blob/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9/install.sh) found:

- `--p2p` explicitly enables P2P patches; they are not enabled by default.
- The default installer sets CPU IOMMU enablement parameters and `iommu=pt`. `--no-iommu` only skips those changes; it does not disable an already enabled IOMMU.
- The upstream README's feature table still marks GPU P2P as “In progress.”

These details identify differences; the external installer's default command is not this project's validated configuration. Do not apply its driver installation procedure directly to a host running the model. Devices such as A100 with native P2P support do not require CMP-specific patches.

## Validate the two paths separately

| Path | Purpose | Validation |
|---|---|---|
| GPU ↔ GPU P2P | Two-GPU communication for TEP2/TP2 | Read/write capability matrix plus actual CUDA peer-copy data, bandwidth, and latency tests |
| NVMe → GPU P2PDMA | Reading PLE data into VRAM | Strict cuFile data validation, NVMe P2PDMA logs, and no host-staging fallback |

Start with capability and topology checks:

```bash
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -m
```

The capability matrix is a preliminary check, not a substitute for an actual data-transfer test. Working P2P on the reference machine does not guarantee success on another machine after installing the external commit.

See the [GDS NVMe P2PDMA reproduction guide](GDS_NVME_P2PDMA_REPRODUCTION.en.md) for the SSD direct-read path. GDS requires independent validation even after GPU P2P passes.
