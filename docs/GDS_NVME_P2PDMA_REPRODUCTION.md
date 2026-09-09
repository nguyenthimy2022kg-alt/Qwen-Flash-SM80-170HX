# CMP 170HX：Samsung 990 PRO → GPU NVMe P2PDMA 复现指南

**简体中文** | [English](GDS_NVME_P2PDMA_REPRODUCTION.en.md)

更新时间：2026-09-08  
验证环境：CMP 170HX/GA100 类 SM80、Ubuntu 24.04、Linux 6.17、NVIDIA 610.43.03、CUDA 13.0 GDS

## 阅读前：这篇文档能完成什么？

本文说明参考机器如何配置并验收 Samsung 990 PRO → GPU 的 NVMe P2PDMA 路径。它从已具备驱动和 GDS 工具的环境开始；**驱动、CUDA/cuFile 和 GDS 的从零安装不在本文范围内**。

| 当前环境 | 应如何使用本文 |
|---|---|
| 已有兼容驱动、CUDA/cuFile 和 GDS 工具 | 根据本机拓扑核对配置，再按本文做 strict 读取与数据校验 |
| 尚未安装 CUDA/cuFile 或 GDS 工具 | 先完成相应安装；本文后面的命令默认 `gdscheck.py`、`gdsio` 已可用 |
| CMP 尚未具备所需的 BAR1/P2P 能力 | 先完成驱动适配；只复制本文参数或安装公开 cmpunlocker，不能保证复现参考机器 |

参考机器使用额外的 CMP 驱动 overlay（本地补丁）。本仓库现在提供 [不含本机超频配置的 BAR1/P2P 参考构建](../drivers/cmp-bar1/README.md)，并非运行驱动的完整复制；来源差异见 [CMP P2P 部署参考](CMP_P2P.md)。已有原生支持或通过其他方式满足 [主机验收条件](从零部署.md#主机验收条件) 的机器无需使用相同补丁。

安装入口见 [CUDA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) 和 [GDS 安装与排障指南](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)。按操作系统、驱动和目标路径选择匹配版本；这些资料不包含本机 CMP 适配。历史验证使用 CUDA 13.0、GDS `1.15.1.6`、libcufile `2.12`，当前官方文档可能对应其他版本。

主机须提供相应的驱动和内核支持；测试环境（主机或容器）须能使用上述工具并访问目标 GPU 与数据文件。先按 [参考硬件与 PCIe 拓扑](REFERENCE_HARDWARE.md) 确认数据盘和读盘卡，再做前置检查。下面的 IOMMU、multipath、ext4 和 BAR1 设置属于参考路径，按需核对和修改；已具备相应能力的环境可直接进入第 9 节严格验证准备。本项目启动器不会修改这些系统配置。

## 数据路径与适用范围

参考环境最终使用的不是传统 `nvidia-fs` 路径，而是：

```text
控制路径：应用 → cuFile → 文件系统 / NVMe 驱动
数据路径：Samsung 990 PRO → PCIe → GPU BAR1 / 显存
          （NVMe P2PDMA，数据本体不经过 CPU 内存中转）
```

在对应工具的输出和实际读取日志中核对以下证据（不是要求每个工具都打印所有行）：

```text
PCIP2PDMACapable:1
checkIfAllGPUsSupportP2PDMA(): 1
NVMe P2PDMA: Supported
cuFile using NVME P2PDMA mode
properties.use_compat_mode : false
```

仅仅 `gdscheck.py` 返回退出码 0、或 compatibility 模式读写成功，都不能证明直通成功。

## 1. 前置检查

先保存现场，不要一开始就改启动参数：

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

参考环境的拓扑示例（UUID 为占位符，BDF 需按目标机器替换）：

```text
PLE GPU: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx / 0000:c5:00.0
NVMe:    Samsung 990 PRO                 / 0000:c4:00.0
```

后续测试优先使用 UUID，而不是 GPU index：

```bash
export CUDA_VISIBLE_DEVICES="GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

如果系统有真实 NVMe 多路径、共享 namespace 或多控制器 ANA，不要盲目关闭
`multipath`；先确认关闭不会破坏根盘或数据盘。

## 2. 保存可恢复备份

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

同时记录 GPU/NVMe BDF、`nvidia-smi topo -m`、`findmnt`、`dmesg` 和当前 Xid，避免把旧错误
误判成这轮改动造成的错误。

## 3. NVIDIA BAR1/P2P 参数

参考环境 cmpunlocker 路径使用以下参数：

```text
NVreg_EnableResizableBar=1
RMForceStaticBar1=1
RMPcieP2PType=1
RmForceDisableIomapWC=1
RmForceEnableGen2=1
RMPcieLinkSpeed=0x1
```

对应配置示例：

```text
options nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords="RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1;RmForceEnableGen2=1;RMPcieLinkSpeed=0x1"
```

2026-09-08 核查时，参考机器将这些参数集中在单一配置文件中：

```text
/etc/modprobe.d/cmpunlocker-nvidia-options.conf
```

旧 `cmpunlocker-p2p.conf` 已不包含参数赋值，避免多处 `RegistryDwords` 相互覆盖。最后两项是该机器 PCIe Gen2 配置，不是所有平台的 GDS 必需参数。参考机器还使用本地驱动 overlay，不能只靠这行参数复现驱动补丁。

不要在这个定制路径中擅自加入官方通用示例里的 `ForceP2P=0`，也不要同时混用
`PeerMappingOverride`、`GrdmaPciTopoCheckOverride` 等未经验证的参数。`RMPcieP2PType=1`
是参考环境 cmpunlocker BAR1/P2P 路径的一部分。

外部 cmpunlocker 安装器默认会写入 `iommu=pt`，与此参考状态不同。其 `--no-iommu` 仅跳过修改，不会自动关闭已有 IOMMU；外部项目默认安装步骤不是本文参考机器的完整复现流程。

修改后不要只看配置文件，必须在重启后确认 NVIDIA 模块实际采用了这些参数。

## 4. 关闭 IOMMU

参考环境验证成功的内核参数是：

```text
amd_iommu=off iommu=off
```

编辑 `/etc/default/grub`，只替换 IOMMU 相关参数，保留系统已有的其他参数。例如参考环境最终为：

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.dc=0 amd_iommu=off iommu=off pci=realloc pci=hpmmioprefsize=2T rootflags=data=ordered"
```

其中 `amdgpu.dc=0`、`pci=realloc` 和 `pci=hpmmioprefsize=2T` 是参考环境已有的硬件参数，
不应不加判断地复制到另一台机器。上述组合仅在该 AMD 平台验证。Intel、虚拟化或依赖设备直通的环境需单独评估；关闭 IOMMU 会改变 DMA 隔离和设备直通能力。

## 5. 关闭 NVMe multipath

仅当已经确认没有真实多路径 namespace 时执行：

```text
/etc/modprobe.d/nvme.conf
```

内容：

```text
options nvme_core multipath=N
```

重新生成 initramfs，并检查配置文件已收录：

```bash
sudo update-initramfs -u -k "$(uname -r)"
sudo lsinitramfs "/boot/initrd.img-$(uname -r)" | grep -F 'etc/modprobe.d/nvme.conf'
```

## 6. 让 ext4 根挂载进入 ordered data mode

在该 ext4 根文件系统的实测环境中，cuFile 文件注册要求实际挂载选项显示 `data=ordered`；这也符合 [NVIDIA 的 ext4 挂载要求](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mounting-a-local-file-system-for-gds)。当目标 NVMe 数据位于根文件系统时，在 GRUB 内加入：

```text
rootflags=data=ordered
```

该方法针对根文件系统；若模型位于独立挂载点，应检查该挂载点的文件系统与持久挂载配置，而不是照搬 `rootflags`。此记录未验证 XFS 或其他文件系统的配置。

更新 GRUB：

```bash
sudo update-grub
```

更新后检查生成的 `/boot/grub/grub.cfg`，确认正常启动项中 `rootflags=data=ordered` 只出现一次。

## 7. 重启顺序

这几项涉及 GRUB、initramfs、NVIDIA 模块和 NVMe 初始化，至少需要重启；为了清除 PCIe/BAR1
残留状态，建议做一次完整断电冷启动。下面命令仅执行普通重启，不等同于断电冷启动：

```bash
sudo reboot
```

如果机器有远程维护需求，必须先确认有物理控制台或带外恢复手段。不要在没有恢复途径时
同时修改 GRUB、NVIDIA 模块和 NVMe 参数。

## 8. 重启后的门禁检查

逐项确认：

```bash
cat /proc/cmdline
findmnt -T /absolute/path/to/ple-artifact -o TARGET,SOURCE,FSTYPE,OPTIONS
cat /sys/module/nvme_core/parameters/multipath
find /sys/class/iommu -mindepth 1 -maxdepth 1 -print
nvidia-smi
```

预期：

```text
/proc/cmdline:       amd_iommu=off iommu=off ... rootflags=data=ordered
root mount options:  rw,relatime,data=ordered
nvme multipath:      N
/sys/class/iommu:    空目录或无 AMD IOMMU 实例
```

读取 NVIDIA 模块实际参数，并确认 RegistryDwords 包含下列值：

```bash
grep -E 'EnableResizableBar|RegistryDwords' /proc/driver/nvidia/params
```

```text
RMForceStaticBar1=1;RMPcieP2PType=1;RmForceDisableIomapWC=1
```

## 9. 创建严格 cuFile 配置

不要先覆盖系统全局 cuFile 配置。创建一个测试专用 JSON：

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

测试时设置：

```bash
export CUFILE_ENV_PATH_JSON=/path/to/strict-p2pdma.json
export CUFILE_USE_PCIP2PDMA=1
export CUFILE_ALLOW_COMPAT_MODE=0
unset CUFILE_FORCE_COMPAT_MODE
export CUFILE_LOGGING_LEVEL=TRACE
GDS_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gds-p2pdma-logs.XXXXXX")
```

以生效配置和实际 I/O 日志为准；仅设置环境变量或看到进程正常退出，不能排除 CPU compatibility fallback。下面通过 `CUFILE_LOGFILE_PATH` 为每条命令保存独立日志；未指定日志目录或路径时，日志通常在应用当前工作目录，并非固定为 `/var/log/cufile.log`。参见 [NVIDIA 日志与环境变量说明](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#environment-variables-used-by-gpudirect-storage)。

## 10. 严格验证

下面 `-d 0` 指测试进程所见的 CUDA 设备 0，不一定是主机 `nvidia-smi` 的 GPU 0。核对 `CUDA_VISIBLE_DEVICES` 和 cuFile 日志中的 GPU BDF，确保测的是准备承担读盘的卡；日志可能用十进制表示总线号，例如 `197` 对应 `c5`。`-m 0` 使用 CUDA 显存分配，`-x 0` 选择同步 GDS I/O。工具路径按实际 CUDA/GDS 安装位置替换。

先运行平台检查：

```bash
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/gdscheck.cufile.log" \
  /usr/local/cuda-13.0/gds/tools/gdscheck.py -p
```

先检查平台摘要中的 NVMe P2PDMA 支持与生效配置；参考版本输出如下（空格可能不同）：

```text
NVMe P2PDMA: Supported
properties.use_compat_mode : false
properties.force_compat_mode : false
IOMMU: disabled
```

`PCIP2PDMACapable:1`、`checkIfAllGPUsSupportP2PDMA(): 1` 和 `cuFile using NVME P2PDMA mode` 在本版工具的 cuFile 日志中，不要求平台摘要重复打印。平台检查仍不能代替目标文件的实际 I/O。

然后在确认位于目标 NVMe ext4 挂载点的目录中做 64 MiB 数据校验。下面的路径只是示例，
不要把测试文件放到 tmpfs：

```bash
TEST_DIR=/path/on/the-target-nvme-ext4
TEST_FILE="$TEST_DIR/gds-test-64m-$(date +%Y%m%d-%H%M%S).bin"

# 先写入带校验模式的数据；-I 1 -V 会写入后读取校验。
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/write-64m.cufile.log" \
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 1 -V

# 再单独验证 SSD 到 GPU 的读取，复用相同尺寸与偏移。
CUFILE_LOGFILE_PATH="$GDS_LOG_DIR/read-64m.cufile.log" \
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 0 -V
```

每条命令都应正常退出；若写入或校验失败，先解决错误，再运行下一条。`-V` 的读校验要求文件先以同样的线程数、尺寸、偏移和校验模式写入，见 [NVIDIA gdsio 参数说明](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html#gdsio-utility)。保留 stdout/stderr，并检查本次读取的日志：

```bash
grep -nEi 'P2PDMA|p2p chunk read|compat|bounce|errno|5001|801|POSIX' \
  "$GDS_LOG_DIR/read-64m.cufile.log"
```

验收时同时核对：

| 证据 | 本例预期 |
|---|---|
| gdsio 读摘要 | `DataSetSize: 65536/65536(KiB)`，`ops: 16`；无短读、I/O 错误或数据不匹配 |
| 本次 cuFile 读日志 | 成功注册文件；16 次 `cuFile p2p chunk read`，每次 4 MiB、`errno: 0` |
| 实际传输路径 | `p2p mode: 1`、`compat: 0`、`bounce-buffer ptr 0`；无 CPU/POSIX 数据中转 |
| 错误检查 | 无 CUDA 801、cuFile 5001 或其他传输错误 |

`gdsio -V` 成功时不一定打印单独的“数据校验成功”提示，需结合完整传输和无校验失败判断。日志中的 POSIX 内存池初始化或包含 `bounce buffer` 的路由消息本身也不表示发生了回退；本机成功日志中就有这些行，应检查每次 I/O 的实际路径。以上是参考版本的格式，其他版本应检查等价证据。

如果输出有 `NVMe: Unsupported`，但同时有 `NVMe P2PDMA: Supported`，要看清楚：前者通常
描述传统 nvidia-fs 路径，后者才是本流程使用的 NVMe P2PDMA 路径。`nvidia-fs` 不加载并不
自动表示本流程失败。

## 11. 可选顺序吞吐测试

功能校验通过后再测吞吐，不要把吞吐测试当作首次功能验证：

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

历史记录中的结果（本次文档整理未重新测量）：

```text
完整 4 GiB 顺序读取：2.562087 GiB/s
-x 0 -w 1 -i 4M -I 0，TRACE 日志开启，1024 次完整读取
```

该数值来自第 14 节证据目录的 `strict_4g_read.stdout`，是工具报告的 I/O 时间内吞吐，不含进程初始化和退出。测试依赖 Samsung 990 PRO、PCIe 拓扑、线程数和 gdsio 参数，不是其他机器的承诺值。

## 12. 常见失败解释

| 现象 | 含义 |
|---|---|
| `checkIfAllGPUsSupportP2PDMA(): 0` | GPU BAR1/P2P、IOMMU、拓扑或驱动参数仍未满足 |
| `CUDA P2P address errornum: 801` | `CUDA_ERROR_NOT_SUPPORTED`；此上下文中获取 P2P 地址的操作不受支持，需查驱动能力和相邻日志 |
| `cuFile error 5001` | `CU_FILE_DRIVER_NOT_INITIALIZED`；检查初始化及 P2PDMA 选路日志，不能将其泛指所有文件注册错误 |
| `NVMe P2PDMA: Unsupported` | 目标 NVMe P2PDMA 资格未通过 |
| `mount option not found` | 根盘没有以 `data=ordered` 出现在实际挂载表中 |
| `use_compat_mode=true` | 允许 compatibility fallback，未满足本文严格配置；实际 `compat: 1` 表示本次 I/O 使用该路径 |
| gdsio 退出 0 但输出有错误 | 以输出内容为准，不能只信退出码 |

错误码定义见 [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html) 与 [cuFile API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#enumerations)。参考环境排查顺序是：确认实际 `/proc/cmdline` → 实际根挂载选项 → multipath → RegistryDwords
→ GPU UUID/BDF 与 NVMe BDF → gdscheck 内容 → gdsio 日志。不要一次叠加更多未经验证的驱动参数。

## 13. 回滚

回滚前先确认备份目录，按实际文件恢复：

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

如果系统无法正常启动，可从 GRUB recovery entry 临时去掉 `rootflags=data=ordered`，再从
控制台恢复 `/etc/default/grub` 和 NVIDIA/NVMe 配置。没有控制台时不要进行这类实验。

## 14. 参考环境最终验收记录

原始项目的历史证据目录（未包含在本仓库）：

```text
evidence/gds-ple-rootflags-data-ordered-20260831-175248/
```

关键报告：

```text
FINAL_REPORT.md
GDS_TEST_COMMANDS.txt
POSTBOOT_GATE.txt
STRICT_RESULT_VALIDATION.txt
```

最终状态：

```text
GDS_NVME_P2PDMA_DIRECT_FUNCTIONAL_PASS
```

验收重点是实际 `NVMe P2PDMA`、数据校验、`compat=false` 和无 bounce/POSIX payload，
而不是“安装了某个 GDS 包”或“gdscheck 返回 0”。
