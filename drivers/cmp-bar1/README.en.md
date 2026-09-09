# CMP BAR1/P2P reference patches

[简体中文](README.md) | **English**

An inspectable BAR1/P2P adaptation for CMP 170HX systems that need driver changes.
Skip this directory if P2P/GDS already works. The model launcher never calls it,
and nothing here automatically installs a driver.

**Status: this is a reference build on pinned public sources, without the host's
overclock configuration. It is not a byte-identical copy of the running driver.**
The historical host driver passed peer transfers and strict NVMe reads. Compilation
of this extracted version does not replace runtime validation after installation.

## Contents

| Component | Purpose and limits |
|---|---|
| Pinned cmpunlocker base | CMP initialization, memory-capacity unlock and BAR1 resizing; also contains PCIe Gen2 adaptation. The three extra patches are not a standalone driver |
| `0011` | BAR1 P2P mappings and page tables; also changes the global ReBAR default, relaxes some mapping teardown diagnostics and includes a Blackwell branch. It is not fully scoped to CMP devices |
| `0013` | Skips mailbox peer pre-registration when BAR1 is selected, avoiding a protocol conflict |
| Optional `0015` | Overrides the platform read-capability rejection for CMP IDs; explicit opt-in, with no automatic validation of PCIe routing |
| `prepare.py` | Verifies/downloads sources, enables P2P without memory-clock/timing overrides, and applies patches. Writes only the chosen source tree and cache |

See [NOTICE.md](NOTICE.md) for provenance, changes and licenses. The root Apache
license does not replace this directory's separate upstream license terms.

Old `0015` comments described a Xeon E5/PLX switch setup. The current reference
machine is **H12D/EPYC 7532 with separate root ports and no enumerated PCIe switch
on the SSD/GPU paths**. Comments are corrected; this is not a universal motherboard
detection algorithm. Cross-root-port traffic must be tested per host. A common
CPU, NUMA node, or an OK capability matrix is insufficient.

## 1. Inspect the host first

Run from the repository root, without sudo. No GPU buffers or model data reads:

```bash
python3 scripts/check-gds-host.py --data-path /actual/PLE/directory > gds-host-report.json
```

The JSON includes GPU/BAR1, PCIe tree/sysfs paths, IOMMU, NVMe multipath,
filesystem, cuFile configuration and tool locations. `acceptance: NOT_TESTED`
is intentional: this is an inventory, not a direct-I/O pass. Missing tools or
permissions are recorded; exit code 0 means only that inventory completed.
Run on the host. Missing tools inside a container do not prove they are absent
on the host. Review absolute paths and system parameters before sharing a report.

## 2. Prepare and build

Pinned to NVIDIA **610.43.03** and the cmpunlocker commit in `sources.json`.
NVIDIA userspace libraries, GSP firmware and target kernel modules must match.
Requires Python 3.12+, curl, patch, make, GCC/G++, and matching kernel headers.
With Secure Boot enabled, sign modules and enroll a key using your distribution's
procedure before loading them. See [NVIDIA's build instructions](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/610.43.03).

```bash
# Repository root; output directory must not exist.
python3 drivers/cmp-bar1/prepare.py --output "$HOME/cmp-bar1-build"
make -C "$HOME/cmp-bar1-build" -j1 modules SYSSRC="/lib/modules/$(uname -r)/build"
```

Neither command installs, unloads or reloads drivers. Compilation uses CPU/RAM;
prefer a maintenance window without model workloads.
`0015` is disabled by default. If driver diagnosis identifies the platform read
capability gate as the remaining blocker, and you are preparing to validate actual
peer transfers on that platform, use a fresh directory with explicit opt-in:

```bash
python3 drivers/cmp-bar1/prepare.py --output "$HOME/cmp-bar1-build-override" --allow-topology-override
make -C "$HOME/cmp-bar1-build-override" -j1 modules SYSSRC="/lib/modules/$(uname -r)/build"
```

Do not stack runs in one output tree. `--cache` selects the download cache.
Source archives and supplied patches are SHA256-verified; versions cannot silently
change. `cmp-bar1-build.json` records choices; `cmp-bar1-patches.log` records patch
application. Do not install unless `.ko` files were actually built.
The process does not invoke upstream `install.sh`, change IOMMU/GRUB/power/core
offsets, or copy the host's memory-overclock configuration.

## 3. Install and recover (manual Ubuntu reference)

These are maintenance-time system changes, **separate from source preparation**.
Stop all GPU workloads first, keep a bootable older kernel and recovery console,
and do not replace a driver while serving a model. Record the current module
path/version and existing cmpunlocker/DKMS/kernel-update hooks: they may overwrite
this installation after an upgrade and must be managed together.

```bash
uname -r
modinfo -n nvidia
modinfo -F version nvidia
modprobe -n -v nvidia
```

Back up the existing module directory, `/etc/modprobe.d`, `/etc/depmod.d`, related
kernel-update hooks and old initramfs to a separate location. The following example
preserves the original module directory and installs into a dedicated one.
If the destination directory or configuration file already exists, stop and
identify its owner first. Set `CMP_SOURCE` to your actual build directory;
Secure Boot users must sign the modules first.

```bash
CMP_SOURCE="$HOME/cmp-bar1-build"
CMP_KERNEL="$(uname -r)"
CMP_DEST="/lib/modules/$CMP_KERNEL/updates/qwen-cmp-bar1"
(
    set -eu
    test ! -e "$CMP_DEST"
    test ! -e /etc/depmod.d/qwen-cmp-bar1.conf
    for module in nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem; do
        test -s "$CMP_SOURCE/kernel-open/$module.ko"
        test "$(modinfo -F version "$CMP_SOURCE/kernel-open/$module.ko")" = 610.43.03
        test "$(modinfo -F vermagic "$CMP_SOURCE/kernel-open/$module.ko" | cut -d ' ' -f1)" = "$CMP_KERNEL"
    done
    sudo install -d "$CMP_DEST"
    for module in nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem; do
        sudo install -m 0644 "$CMP_SOURCE/kernel-open/$module.ko" "$CMP_DEST/"
    done
    for module in nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem; do
        printf 'override %s * updates/qwen-cmp-bar1\n' "$module"
    done | sudo tee /etc/depmod.d/qwen-cmp-bar1.conf
    sudo depmod -a "$CMP_KERNEL"
    modprobe -n -v nvidia
)
```

Confirm resolution to `updates/qwen-cmp-bar1/nvidia.ko`, then run
`sudo update-initramfs -u -k "$(uname -r)"` and schedule a full power-off and boot.
Choose BAR1/IOMMU settings separately using the [P2P reference](../../docs/CMP_P2P.en.md)
and [GDS guide](../../docs/GDS_NVME_P2PDMA_REPRODUCTION.en.md); installing modules does
not configure them. Other distributions need their own initramfs tool.
This recipe does not install automatic kernel-update support. Rebuild and validate
after kernel/driver upgrades.

**Recovery:** boot an older working kernel or recovery console, stop GPU workloads,
and move the added `updates/qwen-cmp-bar1` directory and
`/etc/depmod.d/qwen-cmp-bar1.conf` outside module/configuration search paths.
Restore any system parameters changed separately. Run `depmod -a <kernel-version>`
for the kernel being restored; verify `modprobe -S <kernel-version> -n -v nvidia`
resolves to the original module. Rebuild that kernel's initramfs and reboot.
If the original driver package was replaced, restore matching modules, userspace
libraries and firmware first; do not mix versions.

## 4. Validate after installation

1. Regenerate the inventory; check the loaded version, BAR1 and bidirectional read/write capabilities.
2. Validate real CUDA peer-copy data in both directions. The [CUDA Samples bandwidth/latency test](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest) helps diagnose performance but does not replace full data validation.
3. On the actual PLE filesystem and reading GPU, complete [strict GDS I/O validation](../../docs/GDS_NVME_P2PDMA_REPRODUCTION.en.md). Check complete operation counts, the P2PDMA path, data integrity and absence of CPU compatibility fallback.
4. Validate model loading, consecutive conversation turns and long output; record new results after driver changes.

GPU-to-GPU and SSD-to-GPU are separate acceptance checks. Compilation, enlarged
BAR1, and an OK P2P matrix do not replace them.
