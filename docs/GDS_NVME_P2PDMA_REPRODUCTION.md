# CMP 170HX：Samsung 990 PRO → GPU NVMe P2PDMA 复现指南

更新时间：2026-09-08  
验证环境：CMP 170HX/GA100 类 SM80、Ubuntu 24.04、Linux 6.17、NVIDIA 610.43.03、CUDA 13.0 GDS

本文根据项目原有 GDS 复现记录整理，已替换设备 UUID，并调整配置备份、恢复及先写入测试数据再校验读取的示例。驱动层 P2P 项目与版本说明见 [CMP P2P 部署参考](CMP_P2P.md)。以下系统配置属于特定参考环境，需要按目标机器逐项核对；本项目启动器不会执行这些修改。

## 数据路径与适用范围

参考环境最终使用的不是传统 `nvidia-fs` 路径，而是：

```text
Samsung 990 PRO
  → Linux NVMe PCI_P2PDMA
  → cuFile
  → GPU BAR1 / 显存
```

成功判定必须同时看到：

```text
PCIP2PDMACapable:1
checkIfAllGPUsSupportP2PDMA(): 1
NVMe P2PDMA: Supported
cuFile using NVME P2PDMA mode
use_compat_mode=false
```

仅仅 `gdscheck.py` 返回退出码 0、或 compatibility 模式读写成功，都不能证明直通成功。

这套流程是 CMP170HX 定制环境上的实测方案，不是适用于所有 NVIDIA GPU 的通用解锁方案。
不要直接复制驱动、固件、VBIOS 或 cmpunlocker；先确认 GPU UUID、PCI BDF、NVMe BDF 和拓扑。

## 1. 前置检查

先保存现场，不要一开始就改启动参数：

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version --format=csv
nvidia-smi topo -m

uname -a
cat /proc/cmdline
findmnt -no SOURCE,FSTYPE,OPTIONS /
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

然后检查文件确实进入 initramfs：

```bash
sudo update-initramfs -u -k "$(uname -r)"
```

## 6. 让 ext4 根挂载进入 ordered data mode

在该 ext4 根文件系统的实测环境中，cuFile 文件注册要求实际挂载选项显示 `data=ordered`。当目标 NVMe 数据位于根文件系统时，在 GRUB 内加入：

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
findmnt -no SOURCE,FSTYPE,OPTIONS /
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

同时确认 NVIDIA RegistryDwords 的实际值包含：

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
    "allow_compat_mode": false
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
export CUFILE_LOGGING_LEVEL=TRACE
```

以 JSON 中的 `allow_compat_mode=false` 及实际日志为准；仅设置环境变量或看到进程正常退出，不能排除 CPU compatibility fallback。

## 10. 严格验证

先运行平台检查：

```bash
/usr/local/cuda-13.0/gds/tools/gdscheck.py -p
```

不要只看退出码，要检查输出内容是否包含：

```text
PCIP2PDMACapable:1
checkIfAllGPUsSupportP2PDMA(): 1
NVMe P2PDMA: Supported
cuFile using NVME P2PDMA mode
IOMMU: disabled
```

然后在确认位于目标 NVMe ext4 挂载点的目录中做 64 MiB 数据校验。下面的路径只是示例，
不要把测试文件放到 tmpfs：

```bash
TEST_DIR=/path/on/the-target-nvme-ext4
TEST_FILE="$TEST_DIR/gds-test-64m-$(date +%Y%m%d-%H%M%S).bin"

# 先写入带校验模式的数据；-I 1 -V 会写入后读取校验。
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 1 -V

# 再单独验证 SSD 到 GPU 的读取，复用相同尺寸与偏移。
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_FILE" \
  -d 0 -m 0 -w 1 -s 64M -o 0 -i 4M -x 0 -I 0 -V
```

同时检查 cuFile 日志：

```bash
rg -n "P2PDMA|compat|bounce|5001|801|POSIX" /var/log/cufile.log
```

必须满足：

```text
NVMe P2PDMA: Supported
use_compat_mode=false
数据校验成功
无 error 801
无 cuFile 5001
无 POSIX/bounce payload fallback
```

如果输出只有 `NVMe: Unsupported`，但同时有 `NVMe P2PDMA: Supported`，要看清楚：前者通常
描述传统 nvidia-fs 路径，后者才是本流程使用的 NVMe P2PDMA 路径。`nvidia-fs` 不加载并不
自动表示本流程失败。

## 11. 可选顺序吞吐测试

功能校验通过后再测吞吐，不要把吞吐测试当作首次功能验证：

```bash
/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_DIR/gds-throughput-4g.bin" \
  -d 0 -m 0 -w 1 -s 4G -o 0 -i 4M -x 0 -I 1 -V

/usr/local/cuda-13.0/gds/tools/gdsio \
  -f "$TEST_DIR/gds-throughput-4g.bin" \
  -d 0 -m 0 -w 1 -s 4G -o 0 -i 4M -x 0 -I 0
```

历史记录中的结果（本次文档整理未重新测量）：

```text
4 GiB 顺序读取最佳稳定配置：约 3.919 GiB/s
4K 随机读取：约 22,137 IOPS，45.156 us
```

这是 Samsung 990 PRO、PCIe 拓扑、线程数和 gdsio 参数下的结果，不是所有机器的承诺值。

## 12. 常见失败解释

| 现象 | 含义 |
|---|---|
| `checkIfAllGPUsSupportP2PDMA(): 0` | GPU BAR1/P2P、IOMMU、拓扑或驱动参数仍未满足 |
| `CUDA P2P address errornum: 801` | GPU P2P 地址映射/资格检查失败 |
| `cuFile error 5001` | strict cuFile 路径未建立，常见于 nvidia-fs/文件注册/平台门禁问题 |
| `NVMe P2PDMA: Unsupported` | 目标 NVMe P2PDMA 资格未通过 |
| `mount option not found` | 根盘没有以 `data=ordered` 出现在实际挂载表中 |
| `use_compat_mode=true` | 不是直通结果，可能走了 CPU/compat fallback |
| gdsio 退出 0 但输出有错误 | 以输出内容为准，不能只信退出码 |

参考环境排查顺序是：确认实际 `/proc/cmdline` → 实际根挂载选项 → multipath → RegistryDwords
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
