# CMP 170HX: P2P, BAR1 and GDS

[简体中文](CMP_P2P.md) | **English** · [Home](../README.en.md)

This project uses two direct-transfer paths: **GPU ↔ GPU for computation data, and SSD → GPU for PLE table reads.** They use different software interfaces and require separate validation.

## What each component does

| Component | Role in this project |
|---|---|
| **P2P (peer-to-peer transfers)** | In the two-GPU communication discussion, direct exchanges between GPUs that reduce staging in CPU RAM |
| **BAR1 (a PCIe address window into VRAM)** | Lets external devices access selected GPU memory through PCIe addresses; the window and mappings require driver support |
| **[cmpunlocker](https://github.com/qg19932GH/cmpunlocker)** | Base CMP driver adaptation, including memory-capacity unlock, BAR1 resizing and P2P support |
| **Three additional patches supplied here** | Extend the cmpunlocker base with BAR1 peer mappings and fixes for initialization conflicts and platform restrictions; see [provenance](../drivers/cmp-bar1/NOTICE.md) |
| **GDS / cuFile (NVIDIA storage software and API)** | Works with the driver and Linux NVMe support to read SSD data directly into VRAM |
| **This project's vLLM reader** | Schedules PLE table reads from model inputs and supplies the resulting data to model computation |

The preparation script downloads pinned NVIDIA driver and cmpunlocker sources, then applies the additional patches. It prepares sources only; installation is a separate step in the [driver build guide](../drivers/cmp-bar1/README.en.md).

## BAR1 resizing versus mapping

**Resizing determines the window's size; mapping determines which VRAM locations its addresses refer to.** A GPU-internal address cannot simply be reused by an external device. The driver establishes the correspondence:

```text
PCIe address used by an external device → BAR1 window → target VRAM location
```

With suitable hardware and driver support, another GPU can access target VRAM through these mappings. NVMe direct reads additionally need GDS to arrange the mappings required for storage transfers. **A larger BAR1 does not add physical VRAM, and a mapping does not prove successful data transfer.**

The current project's data flow is:

```text
PLE data on SSD
    │ GDS / NVMe P2PDMA: direct read into VRAM
    ▼
Reading GPU (rank 0)
    │ NCCL: broadcast using the validated GPU P2P path
    ▼
Second GPU (rank 1)
```

One GPU reads and then broadcasts. The CPU still submits and schedules work; the payload does not need staging in CPU RAM. See [hardware and topology](REFERENCE_HARDWARE.en.md) for reading-GPU selection.

## What the three patches fix

| Patch | Original obstacle | Change and applicability |
|---|---|---|
| **0011: BAR1 mappings** | Selecting BAR1 still requires appropriate peer mappings and address translation | Connects mapping and page-table handling; depends on driver internals and needs review when changing driver versions |
| **0013: initialization conflict** | Early mailbox registration (another peer protocol) causes BAR1 to be rejected due to a protocol conflict | Skips mailbox pre-registration when BAR1 is selected; applicable to the same initialization conflict on other hosts |
| **0015: platform read restriction, optional** | The driver rejects an unrecognized platform/topology with a chipset-not-supported read status | Overrides the status to allow reads for device IDs `0x20C2` / `0x2082`; does not validate routing, so requires explicit opt-in and real transfer tests |

The patches do not hard-code this host's PCIe slot addresses, but they are not universal CMP support. In particular, 0015 changes permission to attempt a transfer; it cannot add hardware capability. 0011 also includes some global driver behavior changes; see the [implementation limits](../drivers/cmp-bar1/README.en.md#contents).

**Validation status:** the reference host's original full driver passed GPU peer transfers and SSD direct-read validation. The extracted public version passed a [full build](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/actions/runs/34310525583), but has not been installed and runtime-tested. These are different builds.

## Where to start

| Current state | Next step |
|---|---|
| Real GPU peer transfers and strict SSD direct reads both pass | Prepare the model using the [deployment guide](DEPLOYMENT.en.md); no need to install identical patches |
| GPU peer transfers work, but SSD direct reads do not | Diagnose the storage path with the [GDS guide](GDS_NVME_P2PDMA_REPRODUCTION.en.md); SSD failure alone is not a reason to enable 0015 |
| CMP BAR1/P2P is unavailable | Run the read-only inventory below, then diagnose and adapt using the [driver build guide](../drivers/cmp-bar1/README.en.md) |

Run from the repository root:

```bash
python3 scripts/check-gds-host.py --data-path /actual/PLE/directory
```

The script inventories the environment. `NOT_TESTED` means no real transfers were tested. Use the [host acceptance criteria](DEPLOYMENT.en.md#host-acceptance-criteria) for final validation.
The reference board is H12D, with no enumerated PCIe switch on GPU/SSD paths. Sharing a CPU or NUMA node, or reporting an `OK` capability matrix, does not replace real tests.

## Observed reference configuration

The upstream commit reviewed on 2026-09-08 was [`aaddfd4ce84a2804a7e0cd332acc4c26c79063d9`](https://github.com/qg19932GH/cmpunlocker/tree/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9). This identifies the documentation review snapshot; **it has not been confirmed as the commit originally used to install the reference machine's driver**.

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

The BAR1/P2P additions are now packaged in [drivers/cmp-bar1](../drivers/cmp-bar1/README.en.md), with pinned public sources, verification, build, manual installation and recovery instructions. This extracted version excludes the host overclock configuration and is not identical to the running full v0.3 driver. The platform read-capability override requires explicit opt-in. Obsolete common-switch claims were corrected; the reference topology remains the measured H12D setup.

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
