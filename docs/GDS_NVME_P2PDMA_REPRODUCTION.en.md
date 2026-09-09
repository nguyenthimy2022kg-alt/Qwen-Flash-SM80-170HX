# CMP 170HX: Samsung 990 PRO → GPU NVMe P2PDMA reproduction guide

[简体中文](GDS_NVME_P2PDMA_REPRODUCTION.md) | **English** · [Home](../README.en.md)

Updated: 2026-09-08  
Validated environment: CMP 170HX / GA100-class SM80, Ubuntu 24.04, Linux 6.17, NVIDIA 610.43.03, CUDA 13.0 GDS.

Start with [the roles of P2P, BAR1, cmpunlocker and GDS](CMP_P2P.en.md#what-each-component-does). This guide covers **SSD → GPU** configuration and validation; working GPU-to-GPU P2P does not prove SSD direct reads work.

## Before you start: what does this guide cover?

This guide explains how the reference machine configures and validates its Samsung 990 PRO → GPU NVMe P2PDMA path. It starts with an environment that already has the driver and GDS tools; **installing the driver, CUDA/cuFile, and GDS from scratch is outside its scope**.

| Current environment | How to use this guide |
|---|---|
| Compatible driver, CUDA/cuFile, and GDS tools are available | Review settings against the local topology, then perform strict reads and data validation |
| CUDA/cuFile or GDS tools are not installed | Install them first; commands below assume `gdscheck.py` and `gdsio` are available |
| The CMP GPU lacks the required BAR1/P2P capabilities | Complete driver adaptation first; copying parameters or installing public cmpunlocker alone does not reproduce the reference machine |

The reference machine uses an additional CMP driver overlay. This repository now provides a [BAR1/P2P reference build without the host overclock configuration](../drivers/cmp-bar1/README.en.md), not a complete copy of the running driver. See the [CMP P2P deployment reference](CMP_P2P.en.md) for its differences from public cmpunlocker. Machines that meet the [host acceptance criteria](DEPLOYMENT.en.md#host-acceptance-criteria) through native support or another implementation do not need the same patches.

For installation, consult the [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and [GDS installation and troubleshooting guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html). Select versions compatible with the OS, driver, and intended path; these guides do not supply the local CMP adaptations. Historical validation used CUDA 13.0, GDS `1.15.1.6`, and libcufile `2.12`; current official documentation may describe other versions.

The host must provide the relevant driver and kernel support. The test environment (host or container) needs the tools above and access to the target GPU and data files. Identify the data disk and reading GPU using the [reference hardware and PCIe topology](REFERENCE_HARDWARE.en.md), then run the preliminary checks. The IOMMU, multipath, ext4, and BAR1 settings below belong to the reference path; review and change them as needed. An environment that already has the required capabilities can proceed to the strict-test setup in section 9. The project launcher does not apply these system changes.

## Data path and scope

The reference environment ultimately uses the following path, rather than the traditional `nvidia-fs` path:

```text
Control path: application → cuFile → filesystem / NVMe driver
Data path: Samsung 990 PRO → PCIe → GPU BAR1 / VRAM
           (NVMe P2PDMA; no payload staging in CPU RAM)
```

Check the following evidence across the relevant tools and actual I/O logs; no single tool is expected to print every line:

```text
PCIP2PDMACapable:1
checkIfAllGPUsSupportP2PDMA(): 1
NVMe P2PDMA: Supported
cuFile using NVME P2PDMA mode
properties.use_compat_mode : false
```

A zero exit code from `gdscheck.py`, or successful I/O in compatibility mode, does not establish that direct I/O is working.

## 1. Preliminary checks

Record the current state before changing boot parameters:

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version --format=csv
nvidia-smi topo -m

uname -a
cat /proc/cmdline
findmnt -T /absolute/path/to/ple-artifact -o TARGET,SOURCE,FSTYPE,OPTIONS
cat /sys/module/nvme_core/parameters/multipath 2>/dev/null || true
lspci -nn | grep -Ei 'NVIDIA|Non-Volatile memory|NVMe'
```

Example topology from the reference environment; replace the UUID placeholder and BDFs for the target machine:

```text
PLE GPU: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx / 0000:c5:00.0
NVMe:    Samsung 990 PRO                 / 0000:c4:00.0
```

Prefer a UUID over a GPU index in subsequent tests:

```bash
export CUDA_VISIBLE_DEVICES="GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

If the system uses real NVMe multipathing, shared namespaces, or multi-controller ANA, do not disable `multipath` without confirming that doing so will not disrupt the root or data device.

## 2. Create recoverable backups

Save copies before modifying `/etc/default/grub`, `/etc/modprobe.d`, or initramfs:

```bash
BACKUP_DIR="$HOME/gds-p2pdma-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
sudo cp -a /etc/default/grub "$BACKUP_DIR/grub.before"
sudo cp -a /boot/grub/grub.cfg "$BACKUP_DIR/grub.cfg.before"
sudo cp -a /etc/modprobe.d "$BACKUP_DIR/modprobe.d.before"
if [ -e /etc/modprobe.d/nvme.conf ]; then
    sudo cp -a /etc/modprobe.d/nvme.conf "$BACKUP_DIR/nvme.conf.before"
else
    touch "$BACKUP_DIR/nvme.conf.was-absent"
fi
sudo cp -a "/boot/initrd.img-$(uname -r)" "$BACKUP_DIR/initrd.before"
cp -a /proc/cmdline "$BACKUP_DIR/cmdline.before" 2>/dev/null || true
```

Also record GPU/NVMe BDFs, `nvidia-smi topo -m`, `findmnt`, `dmesg`, and existing Xid messages, so older errors are not mistaken for failures caused by these changes.

## 3. NVIDIA BAR1/P2P parameters

The reference cmpunlocker path uses these parameters:

```text
NVreg_EnableResizableBar=1
RMForceStaticBar1=1
RMPcieP2PType=1
RmForceDisableIomapWC=1
RmForceEnableGen2=1
RMPcieLinkSpeed=0x1
```

Example configuration:

```text
options nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords="RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1;RmForceEnableGen2=1;RMPcieLinkSpeed=0x1"
```

At the 2026-09-08 review, the reference machine kept these parameters in a single file:

```text
/etc/modprobe.d/cmpunlocker-nvidia-options.conf
```

The older `cmpunlocker-p2p.conf` no longer assigns parameters, preventing multiple `RegistryDwords` definitions from overriding each other. The final two settings are specific to this machine's PCIe Gen2 configuration; they are not universal GDS requirements. The reference machine also uses a local driver overlay, so this one configuration line does not reproduce its driver patches.

Do not add `ForceP2P=0` from generic examples to this customized path or mix in unvalidated settings such as `PeerMappingOverride` or `GrdmaPciTopoCheckOverride`. `RMPcieP2PType=1` is part of the reference cmpunlocker BAR1/P2P path.

The external cmpunlocker installer writes `iommu=pt` by default, which differs from this reference state. Its `--no-iommu` option only skips changes; it does not disable an existing IOMMU configuration. The external default installation is not a complete reproduction of this machine.

After rebooting, verify the parameters actually adopted by the NVIDIA module, not just the configuration file contents.

## 4. Disable IOMMU on the reference path

The validated reference kernel parameters are:

```text
amd_iommu=off iommu=off
```

Edit only the IOMMU-related parameters in `/etc/default/grub`, preserving other existing settings. The reference environment's final configuration was:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.dc=0 amd_iommu=off iommu=off pci=realloc pci=hpmmioprefsize=2T rootflags=data=ordered"
```

`amdgpu.dc=0`, `pci=realloc`, and `pci=hpmmioprefsize=2T` were existing hardware settings on this machine and should not be copied uncritically. This combination was validated only on the reference AMD platform. Intel systems, virtualized environments, and systems relying on device passthrough require separate assessment. Disabling IOMMU changes DMA isolation and device-passthrough capabilities.

## 5. Disable NVMe multipath on the reference path

Proceed only after confirming that there are no actual multipath namespaces. Configuration file:

```text
/etc/modprobe.d/nvme.conf
```

Contents:

```text
options nvme_core multipath=N
```

Regenerate initramfs and verify that the configuration is included:

```bash
sudo update-initramfs -u -k "$(uname -r)"
sudo lsinitramfs "/boot/initrd.img-$(uname -r)" | grep -F 'etc/modprobe.d/nvme.conf'
```

## 6. Set the ext4 root mount to ordered data mode

On the tested ext4 root filesystem, cuFile file registration required `data=ordered` in the actual mount options, consistent with [NVIDIA's ext4 mount requirements](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mounting-a-local-file-system-for-gds). When the target NVMe data is on the root filesystem, add this GRUB parameter:

```text
rootflags=data=ordered
```

This method applies to the root filesystem. If model data is on a separate mount, check that mount's filesystem and persistent mount configuration instead of copying `rootflags`. This record does not validate XFS or other filesystem configurations.

Update GRUB:

```bash
sudo update-grub
```

Check the generated `/boot/grub/grub.cfg` and confirm that `rootflags=data=ordered` appears only once in the normal boot entry.

## 7. Reboot sequence

These changes affect GRUB, initramfs, the NVIDIA module, and NVMe initialization, so at least a reboot is required. A full power-off and cold boot is recommended to clear residual PCIe/BAR1 state. The following command performs an ordinary reboot, not a power-off cold boot:

```bash
sudo reboot
```

For remotely managed machines, first confirm access to a physical console or out-of-band recovery. Do not change GRUB, NVIDIA, and NVMe settings together without a recovery path.

## 8. Post-reboot checks

Verify each item:

```bash
cat /proc/cmdline
findmnt -T /absolute/path/to/ple-artifact -o TARGET,SOURCE,FSTYPE,OPTIONS
cat /sys/module/nvme_core/parameters/multipath
find /sys/class/iommu -mindepth 1 -maxdepth 1 -print
nvidia-smi
```

Expected reference state:

```text
/proc/cmdline:       amd_iommu=off iommu=off ... rootflags=data=ordered
root mount options:  rw,relatime,data=ordered
nvme multipath:      N
/sys/class/iommu:    empty directory or no AMD IOMMU instances
```

Read the active NVIDIA module parameters and verify that RegistryDwords include the values below:

```bash
grep -E 'EnableResizableBar|RegistryDwords' /proc/driver/nvidia/params
```

```text
RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1
```

## 9. Create a strict cuFile configuration

Use a dedicated test JSON file rather than overwriting the global cuFile configuration:

```json
{
  "logging": {
    "level": "INFO"
  },
  "properties": {
    "use_pci_p2pdma": true,
    "allow_compat_mode": false,
    "force_compat_mode": false
  },
  "block": {
    "nvme": {
      "use_pci_p2pdma": true
    }
  }
}
```

For the test, set:

```bash
export CUFILE_ENV_PATH_JSON=/path/to/strict-p2pdma.json
export CUFILE_USE_PCIP2PDMA=1
export CUFILE_ALLOW_COMPAT_MODE=0
unset CUFILE_FORCE_COMPAT_MODE
export CUFILE_LOGGING_LEVEL=TRACE
GDS_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gds-p2pdma-logs.XXXXXX")
```

Use the effective configuration and actual I/O logs as evidence. Environment variables alone, or a successful process exit, do not rule out CPU compatibility fallback. The commands below use `CUFILE_LOGFILE_PATH` to save a separate log for each test. Without a configured log directory or path, logs normally go to the application's working directory, not a fixed `/var/log/cufile.log` location. See [NVIDIA's logging and environment-variable documentation](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#environment-variables-used-by-gpudirect-storage).

## 10. Strict validation

Below, `-d 0` refers to CUDA device 0 visible to the test process, which may differ from host GPU 0 in `nvidia-smi`. Verify `CUDA_VISIBLE_DEVICES` and the GPU BDF in the cuFile log against the intended reading GPU. Logs may use decimal bus numbers: for example, `197` corresponds to `c5`. `-m 0` uses CUDA device-memory allocation; `-x 0` selects synchronous GDS I/O. Adjust tool paths to the actual CUDA/GDS installation.

Run the platform check first:

```bash
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/gdscheck.cufile.log" \
  /usr/local/cuda-13.0/gds/tools/gdscheck.py -p
```

First inspect NVMe P2PDMA support and the effective configuration in the platform summary. The reference version reports the following; spacing may differ:

```text
NVMe P2PDMA: Supported
properties.use_compat_mode : false
properties.force_compat_mode : false
IOMMU: disabled
```

`PCIP2PDMACapable:1`, `checkIfAllGPUsSupportP2PDMA(): 1`, and `cuFile using NVME P2PDMA mode` appear in this tool version's cuFile log; the platform summary need not repeat them. A platform check still does not replace actual I/O on the target file.

Next, validate 64 MiB of data in a directory confirmed to reside on the target NVMe ext4 mount. The path below is a placeholder; do not put the test file on tmpfs:

```bash
TEST_DIR=/path/on/the-target-nvme-ext4
TEST_FILE="$TEST_DIR/gds-test-64m-$(date +%Y%m%d-%H%M%S).bin"

# Write data with validation enabled; -I 1 -V writes and then reads to verify.
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/write-64m.cufile.log" \
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 1 -V

# Separately validate SSD-to-GPU reads using the same size and offset.
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/read-64m.cufile.log" \
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 0 -V
```

Each command should exit successfully. Resolve any write or verification failure before running the next command. Read verification with `-V` requires data previously written with matching thread count, size, offset, and verification mode; see the [NVIDIA gdsio option reference](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html#gdsio-utility). Retain stdout/stderr and inspect this read's log:

```bash
grep -nEi 'P2PDMA|p2p chunk read|compat|bounce|errno|5001|801|POSIX' \
  "$GDS_LOG_DIR/read-64m.cufile.log"
```

Check these outcomes together:

| Evidence | Expected for this example |
|---|---|
| gdsio read summary | `DataSetSize: 65536/65536(KiB)`, `ops: 16`; no short reads, I/O errors, or data mismatches |
| This read's cuFile log | Successful file registration; 16 `cuFile p2p chunk read` operations, each 4 MiB with `errno: 0` |
| Actual transfer path | `p2p mode: 1`, `compat: 0`, `bounce-buffer ptr 0`; no CPU/POSIX payload staging |
| Errors | No CUDA 801, cuFile 5001, or other transfer errors |

Successful `gdsio -V` runs do not necessarily print a separate validation-success banner; check complete transfers and the absence of verification failures. POSIX memory-pool initialization or routing messages containing `bounce buffer` do not by themselves indicate fallback. Those lines exist in the successful reference logs, so inspect the actual path of each I/O. These formats belong to the reference version; look for equivalent evidence in other versions.

If the output says `NVMe: Unsupported` but also reports `NVMe P2PDMA: Supported`, distinguish the paths: the former usually describes the traditional `nvidia-fs` path, while the latter is the path used here. An unloaded `nvidia-fs` module does not by itself mean this procedure failed.

## 11. Optional sequential throughput test

Measure throughput only after functional validation passes. A throughput benchmark should not be the first functional test:

```bash
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/write-4g.cufile.log" \
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_DIR/gds-throughput-4g.bin" \
  -d 0 -m 0 -w 1 -s 4G -o 0 -i 4M -x 0 -I 1 -V

CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/read-4g.cufile.log" \
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_DIR/gds-throughput-4g.bin" \
  -d 0 -m 0 -w 1 -s 4G -o 0 -i 4M -x 0 -I 0
```

Historical results, not remeasured while preparing this documentation:

```text
Complete 4 GiB sequential read: 2.562087 GiB/s
-x 0 -w 1 -i 4M -I 0, TRACE logging enabled, 1024 complete reads
```

This value comes from `strict_4g_read.stdout` in the evidence directory in section 14. It is throughput during the tool's measured I/O time, excluding process initialization and teardown. It depends on the Samsung 990 PRO, PCIe topology, thread count, and gdsio parameters, and is not a performance guarantee for other machines.

## 12. Common failures

| Symptom | Meaning |
|---|---|
| `checkIfAllGPUsSupportP2PDMA(): 0` | GPU BAR1/P2P, IOMMU, topology, or driver requirements are still unmet |
| `CUDA P2P address errornum: 801` | `CUDA_ERROR_NOT_SUPPORTED`; here, obtaining a P2P address is unsupported. Inspect driver capabilities and surrounding logs |
| `cuFile error 5001` | `CU_FILE_DRIVER_NOT_INITIALIZED`; inspect initialization and P2PDMA path-selection logs. It is not a generic code for all file-registration errors |
| `NVMe P2PDMA: Unsupported` | The target NVMe P2PDMA path did not pass eligibility checks |
| `mount option not found` | The reference root mount does not show `data=ordered` in the actual mount table |
| `use_compat_mode=true` | Compatibility fallback is allowed, so this does not meet the strict configuration. An actual `compat: 1` I/O uses that path |
| gdsio exits with code 0 but prints errors | Inspect the output; do not rely on the exit code alone |

Error definitions are in the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html) and [cuFile API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#enumerations). The reference troubleshooting order is: actual `/proc/cmdline` → actual root mount options → multipath → RegistryDwords → GPU UUID/BDF and NVMe BDF → gdscheck output → gdsio logs. Do not stack additional unvalidated driver parameters on top of an unresolved configuration.

## 13. Rollback

First identify the correct backup directory and restore files according to the recorded state:

```bash
sudo cp -a "$BACKUP_DIR/grub.before" /etc/default/grub
if [ -f "$BACKUP_DIR/nvme.conf.before" ]; then
    sudo cp -a "$BACKUP_DIR/nvme.conf.before" /etc/modprobe.d/nvme.conf
elif [ -f "$BACKUP_DIR/nvme.conf.was-absent" ]; then
    # Remove only if this deployment created the file and no later edits were made.
    sudo rm -f /etc/modprobe.d/nvme.conf
else
    printf 'Missing nvme.conf backup record; check manually\n'
fi
```

Restore each modified NVIDIA configuration from `modprobe.d.before`. Delete only files confirmed to have been newly created by this procedure. Do not remove pre-existing cmpunlocker configuration. Regenerate the GRUB menu from the restored configuration rather than replacing it with an older generated file. Then run:

```bash
sudo update-initramfs -u -k "$(uname -r)"
sudo update-grub
sudo reboot
```

If the system cannot boot normally, temporarily remove `rootflags=data=ordered` from a GRUB recovery entry, then restore `/etc/default/grub` and NVIDIA/NVMe settings from the console. Do not attempt these experiments without console access.

## 14. Final reference validation record

Historical evidence directory in the original project, not included in this repository:

```text
evidence/gds-ple-rootflags-data-ordered-20260831-175248/
```

Key reports:

```text
FINAL_REPORT.md
GDS_TEST_COMMANDS.txt
POSTBOOT_GATE.txt
STRICT_RESULT_VALIDATION.txt
```

Final status:

```text
GDS_NVME_P2PDMA_DIRECT_FUNCTIONAL_PASS
```

Acceptance depends on actual NVMe P2PDMA operation, correct data, `compat=false`, and no bounce/POSIX payload fallback—not merely an installed GDS package or a zero gdscheck exit code.
