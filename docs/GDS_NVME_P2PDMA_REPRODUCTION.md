# NVMe GDS：SSD 直读显存

**简体中文** | [English](GDS_NVME_P2PDMA_REPRODUCTION.en.md)

本文按“准备环境 → 配置 → 64 MiB 数据校验”完成 SSD 直读显存验证。参考机器为 **H12D、双 CMP 170HX、Samsung 990 PRO、Ubuntu 24.04.4 / Linux 6.17.0-23-generic / NVIDIA 610.43.03**，使用 CUDA 13.0、GDS/cuFile 软件包 1.15.1.6-1。

```text
应用安排读取 → cuFile / 文件系统 / NVMe 驱动
数据实际传输：SSD → PCIe → GPU 显存（不经过 CPU 内存中转）
```

这里使用 **NVMe P2PDMA** 路径，不需要传统路径的 `nvidia-fs` 模块或定制 NVMe 补丁。双卡 P2P 与 SSD 直读分别验证；前者成功不代表后者已经可用。

已验证的是本机的实际直读；完整公开驱动版本在另一台全新机器上的从零 GDS 部署尚未验证。下面是可复现的配置与验收流程，最终以第 3 节的真实读取结果为准。

## 1. 准备环境

- **驱动**：CMP 的 BAR1/P2P 适配优先使用原始社区项目 [bayley/cmpunlocker](https://github.com/bayley/cmpunlocker/tree/5a7bb4b7e5056306fe49e8b824787659abb19914#gpu-to-gpu-p2p)。它已包含相关补丁，无需另外下载并叠加同名补丁；按其 P2P 说明完成平台配置。已有直通能力的机器跳过这一步。
- **GDS 工具**：准备匹配系统的 CUDA/cuFile，确保 `gdscheck.py` 和 `gdsio` 可用。未安装时按 [NVIDIA GDS 安装说明](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)完成安装。下面统一以 `/usr/local/cuda-13.0/gds/tools` 为例。
- **测试目标**：选择模型 PLE 数据所在的真实 NVMe 挂载点和读盘 GPU。参考机器 SSD 与读盘卡位于同一 CPU 的不同根端口下，**不经过 PCIe switch**；其他机器需实际校验。

**CMP 安装参数必须明确**：在所选 cmpunlocker 源码目录安装时使用 `sudo ./install.sh --p2p --no-iommu`。`--p2p` 才会编入能力解锁；`--no-iommu` 防止安装器自动改成 `iommu=pt`，已有 IOMMU 仍需按第 2 节关闭。安装器不会替你写好静态 BAR1 参数，安装后仍须检查该节配置。

**Ubuntu 24.04 的工具包**：先按 NVIDIA 安装说明配置对应 CUDA APT 软件源，再查看并安装本机已验证版本。无需为这条原生路径安装 `nvidia-fs`；先用 `-s` 查看安装计划，确认不会替换已适配的显卡驱动。

```bash
apt-cache policy libcufile-13-0 gds-tools-13-0
sudo apt-get -s install --no-install-recommends libcufile-13-0=1.15.1.6-1 gds-tools-13-0=1.15.1.6-1
sudo apt-get install --no-install-recommends libcufile-13-0=1.15.1.6-1 gds-tools-13-0=1.15.1.6-1
```

若软件源没有该版本，先处理软件源或另行验证可用版本，不要直接安装会替换 CMP 驱动的整套驱动元包。

先查看设备和挂载信息：

```bash
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version --format=csv
nvidia-smi topo -m
nvidia-smi -q -d MEMORY
grep -E '^CONFIG_(PCI_P2PDMA|ZONE_DEVICE)=' "/boot/config-$(uname -r)"
lspci -nn | grep -Ei 'NVIDIA|Non-Volatile memory|NVMe'
lsblk -o NAME,MODEL,TRAN,FSTYPE,MOUNTPOINTS
```

内核应启用 `CONFIG_PCI_P2PDMA=y`、`CONFIG_ZONE_DEVICE=y`，仅看版本号不够；同时检查 `BAR1 Memory Usage` 的总容量，本机为 64 GiB。若 BAR1 仍很小，先解决 BIOS/PCIe 地址空间分配或驱动扩容，不能靠强制能力标志代替。

将以下三个值替换为本机路径和 GPU UUID；`TEST_DIR` 必须是已存在的 NVMe 数据目录，不能是 `/tmp`、tmpfs 或其他磁盘：

```bash
export CUDA_VISIBLE_DEVICES="GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
GDS_TOOLS=/usr/local/cuda-13.0/gds/tools
TEST_DIR=/path/on/the-target-nvme-ext4
findmnt -T "$TEST_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS
```

`-d 0` 将指向上述 UUID 对应的 GPU。参考读盘卡为 `0000:c5:00.0`，SSD 为 `0000:c4:00.0`；这些地址不能照抄到其他机器。

## 2. 核对主机配置

已经通过直读验证的机器可直接进入第 3 节。以下是本机成功配置；需要修改时先展开备份步骤。驱动或启动配置变更后按相应流程重启，首次 BAR1 配置采用完整断电冷启动；本项目启动器不会修改这些设置。

<details>
<summary>修改前：保存恢复备份</summary>

在修改 `/etc/default/grub`、`/etc/modprobe.d` 或 initramfs 前保存副本：

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

保存好 `BACKUP_DIR` 的实际路径，重启后恢复时仍需使用。远程机器应保留控制台恢复途径。

</details>

| 配置项 | 本机成功状态 | 如何处理 |
|---|---|---|
| NVIDIA BAR1/P2P | 可用的大 BAR1、静态 BAR1 与 BAR1 P2P 路径 | 按 cmpunlocker 的 P2P 说明配置，实际参数见下方 |
| IOMMU | `amd_iommu=off iommu=off` | 在 `/etc/default/grub` 中只修改相关参数；Intel、虚拟化环境另行评估 |
| NVMe multipath | `N` | 无真实多路径 namespace 时，在 `/etc/modprobe.d/nvme.conf` 写入 `options nvme_core multipath=N` |
| ext4 | 实际挂载选项有 `data=ordered` | 数据位于根盘时，在 GRUB 加 `rootflags=data=ordered`；独立挂载点修改对应持久挂载配置 |

ext4 要求参见 [NVIDIA 文件系统说明](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mounting-a-local-file-system-for-gds)。不要关闭正在使用的 NVMe 多路径；关闭 IOMMU 会影响设备直通和 DMA 隔离。cmpunlocker 的 `--no-iommu` 仅跳过修改，**不会自动关闭**现有 IOMMU。

<details>
<summary>参考 NVIDIA 参数与重启后检查</summary>

本机将以下配置集中保存在 `/etc/modprobe.d/cmpunlocker-nvidia-options.conf`。如已有 `NVreg_RegistryDwords`，合并到原有配置，避免多处赋值相互覆盖：

```text
options nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords="RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1;RmForceEnableGen2=1;RMPcieLinkSpeed=0x1"
```

最后两项是本机 PCIe Gen2 配置，不是通用 GDS 要求。不要混入与该 BAR1 路径冲突的 `ForceP2P=0` 或其他未经验证的覆盖参数。

完成相关修改后，在 Ubuntu 更新启动文件：

```bash
sudo update-initramfs -u -k "$(uname -r)"
sudo update-grub
```

冷启动后重新设置第 1 节的三个变量，检查实际生效状态：

```bash
cat /proc/cmdline
findmnt -T "$TEST_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS
cat /sys/module/nvme_core/parameters/multipath
find /sys/class/iommu -mindepth 1 -maxdepth 1 -print
grep -E 'EnableResizableBar|RegistryDwords' /proc/driver/nvidia/params
nvidia-smi
```

参考预期：IOMMU 关闭、multipath 为 `N`、目标 ext4 有 `data=ordered`，NVIDIA 参数包含 `RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1`。以重启后的实际值为准。

</details>

## 3. 严格直读与数据校验

创建独立测试配置和日志目录，不覆盖系统 cuFile 配置：

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

配置和日志可放在临时目录，**测试数据必须放在第 1 节确认的 NVMe 目录**。先运行平台检查：

```bash
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/gdscheck.cufile.log" \
  "$GDS_TOOLS/gdscheck.py" -p 2>&1 | tee "$GDS_LOG_DIR/gdscheck.stdout"
```

参考版本应显示：

```text
NVMe P2PDMA: Supported
properties.use_compat_mode : false
properties.force_compat_mode : false
IOMMU: disabled
```

`NVMe: Unsupported` 若与 `NVMe P2PDMA: Supported` 同时出现，前者通常指传统 `nvidia-fs` 路径，不代表本流程失败。平台检查后，继续真实读写测试。

**先写入 64 MiB 校验数据，再单独验证读取。每条命令完成后检查输出；有写入、短读或校验错误就停止。** `-m 0` 使用显存，`-x 0` 使用同步 GDS I/O。

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

`-V` 读校验使用此前按相同线程数、尺寸、偏移写入的校验数据，见 [gdsio 参数说明](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html#gdsio-utility)。验收需要同时满足：

| 检查 | 本例预期 |
|---|---|
| 数据完整与正确 | `DataSetSize: 65536/65536(KiB)`、`ops: 16`，无短读、I/O 错误或校验不匹配 |
| 实际读路径 | 16 次 `cuFile p2p chunk read`，每次 4 MiB、`errno: 0`；`p2p mode: 1`、`compat: 0`、`bounce-buffer ptr 0` |
| 目标 GPU | cuFile 日志的 GPU BDF 对应选中的读盘卡；十进制总线号 `197` 即十六进制 `c5` |

以上为参考版本的日志格式，其他版本检查等价证据。`gdsio -V` 不一定另打印“校验成功”；正常退出也不能代替完整数据与实际路径检查。POSIX 内存池初始化、带 `bounce buffer` 的路由日志本身不等于 CPU 回退，要看每次 I/O。

**通过这一步，才算目标 NVMe → 指定 GPU 的直读走通。**

<details>
<summary>可选：4 GiB 顺序吞吐测试</summary>

功能校验通过后再测吞吐，不要把吞吐测试当作首次功能验证：

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

历史记录中的结果（本次文档整理未重新测量）：

```text
完整 4 GiB 顺序读取：2.562087 GiB/s
-x 0 -w 1 -i 4M -I 0，TRACE 日志开启，1024 次完整读取
```

这是工具测得的 I/O 区间吞吐，不含进程初始化与退出；它依赖本机 SSD、拓扑和测试参数，不是其他机器的承诺值。

</details>

## 4. 常见问题

| 现象 | 含义 |
|---|---|
| `checkIfAllGPUsSupportP2PDMA(): 0` | GPU BAR1/P2P、IOMMU、拓扑或驱动参数仍未满足 |
| `CUDA P2P address errornum: 801` | `CUDA_ERROR_NOT_SUPPORTED`；此上下文中获取 P2P 地址的操作不受支持，需查驱动能力和相邻日志 |
| `cuFile error 5001` | `CU_FILE_DRIVER_NOT_INITIALIZED`；检查初始化及 P2PDMA 选路日志，不能将其泛指所有文件注册错误 |
| `NVMe P2PDMA: Unsupported` | 目标 NVMe P2PDMA 资格未通过 |
| `mount option not found` | 根盘没有以 `data=ordered` 出现在实际挂载表中 |
| `use_compat_mode=true` | 允许 compatibility fallback，未满足本文严格配置；实际 `compat: 1` 表示本次 I/O 使用该路径 |
| gdsio 退出 0 但输出有错误 | 以输出内容为准，不能只信退出码 |

排查顺序：实际启动参数 → 目标挂载选项 → multipath → NVIDIA 参数 → GPU/SSD 地址 → gdscheck 输出 → 本次 I/O 日志。

## 5. 恢复配置

先将 `BACKUP_DIR` 设为第 2 节保存的实际备份目录，再按记录恢复：

```bash
sudo cp -a "$BACKUP_DIR/grub.before" /etc/default/grub
if [ -f "$BACKUP_DIR/nvme.conf.before" ]; then
    sudo cp -a "$BACKUP_DIR/nvme.conf.before" /etc/modprobe.d/nvme.conf
elif [ -f "$BACKUP_DIR/nvme.conf.was-absent" ]; then
    # 仅在该文件由本次部署新建，且没有后续修改时删除。
    sudo rm -f /etc/modprobe.d/nvme.conf
else
    printf '缺少 nvme.conf 备份记录，请人工核对\n'
fi
```

逐个将本次修改的 NVIDIA 配置从 `modprobe.d.before` 恢复；仅删除确认由本次部署新增的文件。不要删除原有 cmpunlocker 配置。GRUB 菜单由恢复后的配置重新生成，而不是覆盖历史生成文件。然后：

```bash
sudo update-initramfs -u -k "$(uname -r)"
sudo update-grub
sudo reboot
```

如果系统不能正常启动，从 GRUB 恢复项或控制台还原启动参数及上述配置。驱动安装的恢复按所用 cmpunlocker 版本的卸载说明进行。


<details>
<summary>参考验证记录</summary>

原始证据目录（未包含在本仓库）：`evidence/gds-ple-rootflags-data-ordered-20260831-175248/`。`STRICT_RESULT_VALIDATION.txt` 记录了 64 MiB 数据校验和完整 4 GiB 读取，`strict_4g_read.stdout` 记录了上述吞吐。该路径历史验收为 `GDS_NVME_P2PDMA_DIRECT_FUNCTIONAL_PASS`。

</details>
