# Reference hardware and PCIe topology

[简体中文](REFERENCE_HARDWARE.md) | **English** · [Home](../README.en.md)

Read-only inspection on 2026-09-08 using `lspci -Dtv`, sysfs, `findmnt`, `lscpu`, and SMBIOS. This describes the machine on which the project works, not a compatibility list for other systems.

## Is a PCIe switch board required?

**There is no PCIe switch on the enumerated paths connecting these GPUs and the SSD. They are not behind a common switch.** The devices attach to CPU PCIe root ports. The 990 PRO and GPU `c5:00.0` use different root ports under root bus `0000:c0`; GPU `01:00.0` belongs to another root bus.

Software enumeration cannot identify passive risers or extension cables without a switch chip. Such accessories are not PCIe switches.

## Validated reference hardware

| Component | Reference configuration |
|---|---|
| CPU | One AMD EPYC 7532, 32 cores; the OS reports one NUMA node |
| Motherboard | H12D (confirmed); SMBIOS identifies it as HUANANZHI H12D-8D V2.0 |
| GPUs | Two NVIDIA CMP 170HX, SM80, nominally 64 GB each; see deployment guide for available VRAM |
| PLE storage | Samsung SSD 990 PRO 4TB, `0000:c4:00.0`, currently `/dev/nvme1n1` |
| Data filesystem | `/dev/nvme1n1p2`, ext4, mounted with `data=ordered` |
| Host memory | Approximately 32 GiB RAM and 8 GiB swap; observed configuration, not a recommendation for ample headroom |
| Observed links | SSD: PCIe 4.0 ×4; GPU `c5`: PCIe 2.0 ×16; GPU `01`: PCIe 2.0 ×4 |
| Software foundation | Ubuntu 24.04, Linux 6.17.0-23, NVIDIA 610.43.03 Open Kernel Module, CUDA 13.0 GDS, with local CMP driver adaptations |

Link values are the negotiated state on this machine, not general device specifications or minimum GDS requirements. The other drive, a Kingston NV2, does not hold this PLE dataset.

## Actual connectivity

Unrelated devices are omitted. PCI device addresses (BDFs) may change with slots, BIOS settings, or reboot.

```text
Single AMD EPYC 7532
├─ Root bus 0000:c0
│  ├─ Root port 0000:c0:01.5 ─ Samsung 990 PRO   0000:c4:00.0
│  └─ Root port 0000:c0:03.1 ─ CMP 170HX        0000:c5:00.0
└─ Root bus 0000:00
   └─ Root port 0000:00:01.1 ─ CMP 170HX        0000:01:00.0
```

The GPU pair appears as `NODE` in `nvidia-smi topo -m`: traffic crosses the interconnect between PCIe host bridges within one NUMA node. This does not mean a common switch, and the GPU matrix alone does not identify the SSD topology.

## Current project data path

```text
PLE data on the 990 PRO
  ── strict cuFile / NVMe P2PDMA ──> GPU c5 (TP rank 0)
  ── NCCL broadcast from GPU c5 ───> GPU 01 (TP rank 1)
```

Current TEP2 reads through one GPU and broadcasts the PLE rows needed for that step to the other. It does not issue duplicate SSD reads directly into both GPUs. The launcher uses the first `gpu_ids` entry in `config/local.json` as the reading GPU; the reference configuration puts the UUID for `c5` first. On another machine, choose a GPU that passes strict read validation and verify the ordering. See the [launcher](../scripts/serve.py) and [PLE transport](../src/tp_ple_transport.py).

**Direct SSD-to-VRAM transfer means the payload is not staged in CPU RAM; it does not mean bypassing the CPU package entirely.** Here, traffic traverses the CPU's PCIe interconnect. The CPU still handles read planning, filesystem operations, and submission. The [NVIDIA GDS design guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) distinguishes control and data paths.

## Evaluating another machine

A common switch can shorten the device path, but it was not necessary on this machine and does not guarantee success elsewhere. Routing across root ports depends on the CPU platform, kernel, and drivers, with compatibility checks in Linux P2PDMA. Sharing a CPU or NUMA node does not establish working direct I/O. See the [Linux P2PDMA routing documentation](https://docs.kernel.org/driver-api/pci/p2pdma.html).

First locate the disk that actually contains the PLE files and inspect its parent path alongside the reading GPU:

```bash
PLE_DIR=/absolute/path/to/ple-artifact
findmnt -T "$PLE_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -d -o NAME,MODEL,SIZE,TRAN
lspci -Dtv
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv
nvidia-smi topo -m
# Replace nvme1 and the BDF with the actual devices on the target machine.
readlink -f /sys/class/nvme/nvme1/device
readlink -f /sys/bus/pci/devices/0000:c5:00.0
```

Then validate [GPU-to-GPU P2P](CMP_P2P.en.md) and [strict SSD-to-GPU reads](GDS_NVME_P2PDMA_REPRODUCTION.en.md) separately. The latter requires actual data validation and logs for that read, not just a capability matrix or process exit code. This documentation review did not rerun benchmarks or change drivers, BIOS, or hardware settings.
