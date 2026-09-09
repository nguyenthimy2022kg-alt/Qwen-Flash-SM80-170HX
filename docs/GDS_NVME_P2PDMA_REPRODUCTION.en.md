# NVMe GDS: direct SSD-to-GPU reads

[简体中文](GDS_NVME_P2PDMA_REPRODUCTION.md) | **English**

Follow “prepare → configure → validate 64 MiB” to verify direct SSD-to-GPU reads. The reference host is **H12D, two CMP 170HX GPUs, Samsung 990 PRO, Ubuntu 24.04.4 / Linux 6.17.0-23-generic / NVIDIA 610.43.03**, with CUDA 13.0, GDS/cuFile packages 1.15.1.6-1.

```text
Application schedules reads → cuFile / filesystem / NVMe driver
Actual payload: SSD → PCIe → GPU VRAM (no staging in CPU RAM)
```

This uses **NVMe P2PDMA**, without the traditional path's `nvidia-fs` module or custom NVMe patches. GPU-to-GPU P2P and SSD direct reads need separate validation; success with one does not prove the other works.

Direct reads have been validated on the reference host. A fresh GDS deployment using the complete public driver revision has not yet been validated on a second machine. Use the real read checks in section 3 to establish success.

## 1. Prepare the environment

- **Driver:** for CMP BAR1/P2P adaptation, use the original community project [bayley/cmpunlocker](https://github.com/bayley/cmpunlocker/tree/5a7bb4b7e5056306fe49e8b824787659abb19914#gpu-to-gpu-p2p). It already contains the relevant patches; do not download and apply another copy of those patches. Complete platform configuration using its P2P instructions. Skip this step on a host with working direct-transfer support.
- **GDS tools:** prepare CUDA/cuFile versions compatible with the host and make `gdscheck.py` and `gdsio` available. If needed, follow [NVIDIA's GDS installation instructions](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html). Commands below use `/usr/local/cuda-13.0/gds/tools`.
- **Test target:** select the actual NVMe mount containing the model's PLE data and the reading GPU. On the reference machine, the SSD and reading GPU use different root ports on the same CPU, **without a PCIe switch**. Validate other machines directly.

**Set the CMP installer options explicitly**: run `sudo ./install.sh --p2p --no-iommu` from the selected cmpunlocker source directory. `--p2p` compiles in the capability override; `--no-iommu` prevents the installer from switching to `iommu=pt`. Disable any existing IOMMU setting as described in section 2. The installer does not configure static BAR1 automatically; check those settings after installation.

**Tools on Ubuntu 24.04**: configure the matching NVIDIA CUDA APT repository using NVIDIA's installation instructions, then inspect and install the validated packages below. This native path does not require `nvidia-fs`. Review the `-s` installation plan first and confirm that it will not replace the adapted GPU driver.

```bash
apt-cache policy libcufile-13-0 gds-tools-13-0
sudo apt-get -s install --no-install-recommends libcufile-13-0=1.15.1.6-1 gds-tools-13-0=1.15.1.6-1
sudo apt-get install --no-install-recommends libcufile-13-0=1.15.1.6-1 gds-tools-13-0=1.15.1.6-1
```

If that version is unavailable, resolve the repository setup or validate another version separately; do not substitute a driver metapackage that replaces the adapted CMP driver.

Identify the devices and mounts:

```bash
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version --format=csv
nvidia-smi topo -m
nvidia-smi -q -d MEMORY
grep -E '^CONFIG_(PCI_P2PDMA|ZONE_DEVICE)=' "/boot/config-$(uname -r)"
lspci -nn | grep -Ei 'NVIDIA|Non-Volatile memory|NVMe'
lsblk -o NAME,MODEL,TRAN,FSTYPE,MOUNTPOINTS
```

The kernel must enable `CONFIG_PCI_P2PDMA=y` and `CONFIG_ZONE_DEVICE=y`; the version number alone is insufficient. Also check the total under `BAR1 Memory Usage` (64 GiB on the reference host). If BAR1 remains small, resolve BIOS/PCIe address allocation or driver resizing first; a forced capability flag cannot replace that.

Replace these three values with local paths and the GPU UUID. `TEST_DIR` must be an existing directory on the intended NVMe, not `/tmp`, tmpfs or another disk:

```bash
export CUDA_VISIBLE_DEVICES="GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
GDS_TOOLS=/usr/local/cuda-13.0/gds/tools
TEST_DIR=/path/on/the-target-nvme-ext4
findmnt -T "$TEST_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS
```

`-d 0` will select the GPU identified by that UUID. The reference reading GPU is `0000:c5:00.0` and SSD is `0000:c4:00.0`; do not copy these addresses to another machine.

## 2. Check host configuration

A host with verified direct reads can proceed to section 3. These are the reference host's working settings. Before making changes, expand the backup procedure below. Reboot as required after driver or boot changes; use a full power-off and cold boot for initial BAR1 configuration. The project launcher does not change these settings.

<details>
<summary>Before changing settings: create recovery backups</summary>

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

Keep the actual `BACKUP_DIR` path for recovery after reboot. Remote machines should retain a console recovery path.

</details>

| Setting | Working reference state | Action |
|---|---|---|
| NVIDIA BAR1/P2P | A usable large BAR1, static BAR1 and BAR1 P2P | Follow cmpunlocker's P2P instructions; reference parameters are below |
| IOMMU | `amd_iommu=off iommu=off` | Edit only related parameters in `/etc/default/grub`; assess Intel and virtualized systems separately |
| NVMe multipath | `N` | If no real multipath namespaces are in use, put `options nvme_core multipath=N` in `/etc/modprobe.d/nvme.conf` |
| ext4 | Actual mount options include `data=ordered` | For data on the root filesystem, add `rootflags=data=ordered` to GRUB; for a separate mount, edit that mount's persistent configuration |

See [NVIDIA's filesystem requirements](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mounting-a-local-file-system-for-gds) for ext4. Do not disable active NVMe multipathing. Disabling IOMMU affects device passthrough and DMA isolation. cmpunlocker's `--no-iommu` only skips changes; it **does not disable** an existing IOMMU configuration.

<details>
<summary>Reference NVIDIA parameters and post-reboot checks</summary>

The reference host keeps this configuration in `/etc/modprobe.d/cmpunlocker-nvidia-options.conf`. If `NVreg_RegistryDwords` already exists, merge into that configuration to avoid competing assignments:

```text
options nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords="RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1;RmForceEnableGen2=1;RMPcieLinkSpeed=0x1"
```

The last two entries are this host's PCIe Gen2 settings, not universal GDS requirements. Do not mix in `ForceP2P=0`, which conflicts with this BAR1 path, or other unvalidated overrides.

After making the relevant changes, update the boot files on Ubuntu:

```bash
sudo update-initramfs -u -k "$(uname -r)"
sudo update-grub
```

After a cold boot, set the three variables from section 1 again and check the effective state:

```bash
cat /proc/cmdline
findmnt -T "$TEST_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS
cat /sys/module/nvme_core/parameters/multipath
find /sys/class/iommu -mindepth 1 -maxdepth 1 -print
grep -E 'EnableResizableBar|RegistryDwords' /proc/driver/nvidia/params
nvidia-smi
```

Reference expectations: IOMMU disabled, multipath `N`, target ext4 mounted with `data=ordered`, and NVIDIA parameters containing `RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1`. Check the active values after reboot.

</details>

## 3. Validate direct reads and data correctness

Create a dedicated test configuration and log directory without replacing the system cuFile configuration:

```bash
set -o pipefail
GDS_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gds-p2pdma-logs.XXXXXX")
export CUFILE_ENV_PATH_JSON="$GDS_LOG_DIR/strict-p2pdma.json"
cat > "$CUFILE_ENV_PATH_JSON" <<'JSON'
{
  "logging": {"level": "INFO"},
  "properties": {
    "use_pci_p2pdma": true,
    "allow_compat_mode": false,
    "force_compat_mode": false
  },
  "block": {"nvme": {"use_pci_p2pdma": true}}
}
JSON
export CUFILE_USE_PCIP2PDMA=1
export CUFILE_ALLOW_COMPAT_MODE=0
unset CUFILE_FORCE_COMPAT_MODE
export CUFILE_LOGGING_LEVEL=TRACE
```

Configuration and logs may use a temporary directory. **Test data must use the NVMe directory identified in section 1.** Run the platform check:

```bash
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/gdscheck.cufile.log" \
  "$GDS_TOOLS/gdscheck.py" -p 2>&1 | tee "$GDS_LOG_DIR/gdscheck.stdout"
```

The reference version should show:

```text
NVMe P2PDMA: Supported
properties.use_compat_mode : false
properties.force_compat_mode : false
IOMMU: disabled
```

If `NVMe: Unsupported` appears alongside `NVMe P2PDMA: Supported`, the former usually refers to the traditional `nvidia-fs` path and does not mean this procedure failed. Continue with actual file I/O after the platform check.

**Write 64 MiB of verification data first, then validate a separate read. Inspect each command's output before continuing; stop on write, short-read or verification errors.** `-m 0` uses device memory; `-x 0` uses synchronous GDS I/O.

```bash
TEST_FILE="$TEST_DIR/gds-test-64m-$(date +%Y%m%d-%H%M%S).bin"
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/write-64m.cufile.log" \
  "$GDS_TOOLS/gdsio" -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 1 -V \
  2>&1 | tee "$GDS_LOG_DIR/write-64m.stdout"
```

```bash
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/read-64m.cufile.log" \
  "$GDS_TOOLS/gdsio" -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 0 -V \
  2>&1 | tee "$GDS_LOG_DIR/read-64m.stdout"
grep -nEi 'P2PDMA|p2p chunk read|compat|bounce|errno|5001|801|POSIX' \
  "$GDS_LOG_DIR/read-64m.cufile.log"
```

Read verification with `-V` uses verification data previously written with matching thread count, size and offset; see the [gdsio option reference](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html#gdsio-utility). All of the following must pass:

| Check | Expected in this example |
|---|---|
| Complete, correct data | `DataSetSize: 65536/65536(KiB)`, `ops: 16`; no short reads, I/O errors or verification mismatches |
| Actual read path | 16 `cuFile p2p chunk read` operations, each 4 MiB with `errno: 0`; `p2p mode: 1`, `compat: 0`, `bounce-buffer ptr 0` |
| Target GPU | The cuFile log's GPU BDF matches the selected reading GPU; decimal bus number `197` equals hexadecimal `c5` |

These log formats are from the reference version; check equivalent evidence with other versions. `gdsio -V` need not print a separate success banner. A successful exit cannot replace complete-data and actual-path checks. POSIX memory-pool initialization or routing messages mentioning `bounce buffer` do not themselves indicate CPU fallback; inspect each I/O.

**Passing this step verifies direct reads from the target NVMe to the selected GPU.**

<details>
<summary>Optional: 4 GiB sequential throughput test</summary>

Measure throughput only after functional validation passes. A throughput benchmark should not be the first functional test:

```bash
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/write-4g.cufile.log" \
  "$GDS_TOOLS/gdsio" \
  -f "$TEST_DIR/gds-throughput-4g.bin" \
  -d 0 -m 0 -w 1 -s 4G -o 0 -i 4M -x 0 -I 1 -V

CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/read-4g.cufile.log" \
  "$GDS_TOOLS/gdsio" \
  -f "$TEST_DIR/gds-throughput-4g.bin" \
  -d 0 -m 0 -w 1 -s 4G -o 0 -i 4M -x 0 -I 0
```

Historical results, not remeasured while preparing this documentation:

```text
Complete 4 GiB sequential read: 2.562087 GiB/s
-x 0 -w 1 -i 4M -I 0, TRACE logging enabled, 1024 complete reads
```

This is throughput during the tool's measured I/O time, excluding process initialization and teardown. It depends on the local SSD, topology and test parameters; it is not a performance guarantee for other machines.

</details>

## 4. Troubleshooting

| Symptom | Meaning |
|---|---|
| `checkIfAllGPUsSupportP2PDMA(): 0` | GPU BAR1/P2P, IOMMU, topology, or driver requirements are still unmet |
| `CUDA P2P address errornum: 801` | `CUDA_ERROR_NOT_SUPPORTED`; here, obtaining a P2P address is unsupported. Inspect driver capabilities and surrounding logs |
| `cuFile error 5001` | `CU_FILE_DRIVER_NOT_INITIALIZED`; inspect initialization and P2PDMA path-selection logs. It is not a generic code for all file-registration errors |
| `NVMe P2PDMA: Unsupported` | The target NVMe P2PDMA path did not pass eligibility checks |
| `mount option not found` | The reference root mount does not show `data=ordered` in the actual mount table |
| `use_compat_mode=true` | Compatibility fallback is allowed, so this does not meet the strict configuration. An actual `compat: 1` I/O uses that path |
| gdsio exits with code 0 but prints errors | Inspect the output; do not rely on the exit code alone |

Troubleshoot in this order: active boot parameters → target mount options → multipath → NVIDIA parameters → GPU/SSD addresses → gdscheck output → this test's I/O logs.

## 5. Restore configuration

Set `BACKUP_DIR` to the actual backup path saved in section 2, then restore files according to the recorded state:

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

If the system cannot boot normally, restore boot parameters and the files above through a GRUB recovery entry or console. To undo a driver installation, follow the uninstall instructions for the cmpunlocker version used.


<details>
<summary>Reference validation record</summary>

Original evidence directory, not included in this repository: `evidence/gds-ple-rootflags-data-ordered-20260831-175248/`. `STRICT_RESULT_VALIDATION.txt` records 64 MiB data verification and a complete 4 GiB read; `strict_4g_read.stdout` records the throughput above. The historical acceptance status is `GDS_NVME_P2PDMA_DIRECT_FUNCTIONAL_PASS`.

</details>
