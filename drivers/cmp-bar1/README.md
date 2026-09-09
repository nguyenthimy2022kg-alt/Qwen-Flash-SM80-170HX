# CMP BAR1/P2P 参考补丁

**简体中文** | [English](README.en.md)

为需要驱动适配的 CMP 170HX 提供可审查的 BAR1/P2P 补丁与固定版本构建入口。
已有可用 P2P/GDS 的机器直接跳过。本目录不会被服务启动器调用，也不会自动安装驱动。

**状态：这是去掉本机超频配置、接到公开固定版本源码上的参考构建，并非当前运行驱动的逐字复制。**
本机历史驱动的双卡互传和 NVMe 直读已经验证；此整理版的编译检查不能代替装机后的运行验证。

## 包含什么

| 部分 | 作用与边界 |
|---|---|
| 固定 cmpunlocker 基础 | 提供 CMP 初始化、显存容量解锁和 BAR1 扩容等前置功能；基础代码还包含 PCIe Gen2 适配，并非仅三个增量补丁就能独立运行 |
| `0011` | 接入 BAR1 P2P 的映射与页表路径；包含全局 ReBAR 默认值、部分映射销毁诊断放宽及 Blackwell 分支变化，不能称为完全限定于 CMP 的通用安全补丁 |
| `0013` | 使用 BAR1 时跳过 mailbox peer 预登记，避免两种互传机制冲突 |
| `0015`，可选 | 对 CMP 型号覆盖平台读取能力拒绝；需要显式选择，不检测、更不创造实际可用的 PCIe 路由 |
| `prepare.py` | 下载校验源码、生成不启用显存超频/时序调整的配置并应用补丁；只写指定的源码目录与缓存 |

来源、修改记录和许可见 [NOTICE.md](NOTICE.md)。本目录单独遵循其许可，不以仓库根目录的 Apache 许可替代上游许可。

旧 `0015` 注释来自“Xeon E5＋PLX switch”环境。当前参考机器是 **H12D＋EPYC 7532，SSD 与 GPU 位于不同根端口，无枚举出的 PCIe switch**。
此处已经修正注释，但没有把它改成自动判断所有主板的算法。跨根端口能否传输必须逐机验证；同 CPU、同 NUMA、能力矩阵 OK 都不是充分条件。

## 1. 先检查，选择是否需要适配

在仓库根目录运行（不需要 sudo，不分配 GPU 工作区、不读取模型数据）：

```bash
python3 scripts/check-gds-host.py --data-path /实际的/PLE目录 > gds-host-report.json
```

JSON 报告包含 GPU/BAR1、PCIe 树与 sysfs 路径、IOMMU、NVMe multipath、文件系统、cuFile 配置和工具位置。
`acceptance: NOT_TESTED` 是有意保留：这只是环境清单，不宣称直通成功。缺少工具或权限会单独记录；退出码 0 只表示清单生成完成。
在宿主机执行；容器内未找到 GDS 工具不等于宿主机未安装。分享报告前检查绝对路径和系统参数是否适合公开。

## 2. 准备源码与构建

仅固定支持 NVIDIA **610.43.03** 源码与 `sources.json` 中的 cmpunlocker 提交。
主机 NVIDIA 用户态库、GSP 固件和目标内核模块版本须匹配。还需 Python 3.12+、curl、patch、make、GCC/G++、匹配的内核头文件。
Secure Boot 开启时须按发行版要求签名并登记密钥，否则不能加载未签名模块。
[NVIDIA 构建说明](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/610.43.03)提供基础要求。

```bash
# 在仓库根目录运行，输出目录必须不存在。
python3 drivers/cmp-bar1/prepare.py --output "$HOME/cmp-bar1-build"
make -C "$HOME/cmp-bar1-build" -j1 modules SYSSRC="/lib/modules/$(uname -r)/build"
```

这两个命令不会安装、卸载或重载驱动。编译仍会占用 CPU/内存，建议在没有模型负载时执行。
默认不应用 `0015`。若驱动诊断表明最后受阻于平台读取能力判断，且准备进行该平台的实际互传验证，可另选全新目录：

```bash
python3 drivers/cmp-bar1/prepare.py --output "$HOME/cmp-bar1-build-override" --allow-topology-override
make -C "$HOME/cmp-bar1-build-override" -j1 modules SYSSRC="/lib/modules/$(uname -r)/build"
```

不要依次在同一个目录叠加。`--cache` 可指定下载缓存；所有源归档和附带补丁均核对 SHA256，不接受静默更换版本。
`cmp-bar1-build.json` 记录选择，`cmp-bar1-patches.log` 保存应用日志。没有编译出 `.ko` 文件就不能进入安装。
此流程不调用上游 `install.sh`，不改 IOMMU、GRUB、功耗、核心偏移或本机显存超频设置。

## 3. 安装与恢复（Ubuntu 手动参考）

以下是计划维护时才执行的系统修改，**不是构建过程的一部分**。先停止所有 GPU 工作负载，保留可用旧内核与恢复控制台；不要在正在提供模型服务时替换驱动。
先记录当前模块路径、版本，以及已有 cmpunlocker/DKMS/内核升级钩子。它们可能在升级后覆盖本次安装，需由部署者统一管理。

```bash
uname -r
modinfo -n nvidia
modinfo -F version nvidia
modprobe -n -v nvidia
```

建议将现有模块目录、`/etc/modprobe.d`、`/etc/depmod.d`、相关内核升级钩子和旧 initramfs 备份至独立位置。
确认可恢复后，以下示例将新模块放入专用目录，保留原目录。若目录或配置文件已存在，停止并先核对来源。
将 `CMP_SOURCE` 指向你实际构建的目录；Secure Boot 用户先完成模块签名。

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

确认上面解析到 `updates/qwen-cmp-bar1/nvidia.ko` 后，再执行 `sudo update-initramfs -u -k "$(uname -r)"`，安排关机断电后重新开机。
BAR1、IOMMU 等设置另外按 [P2P 参考](../../docs/CMP_P2P.md)与 [GDS 指南](../../docs/GDS_NVME_P2PDMA_REPRODUCTION.md)选择；安装模块本身不会自动配置它们。
其他发行版使用自身的 initramfs 工具。此流程无自动内核更新支持，内核/驱动升级后须重新构建并验收。

**恢复：** 从可用旧内核或恢复控制台启动，停止 GPU 工作负载，将本次新增的 `updates/qwen-cmp-bar1` 目录和 `/etc/depmod.d/qwen-cmp-bar1.conf` 移出模块/配置搜索路径；恢复本次另外修改的系统参数。
对需要恢复的内核运行 `depmod -a <内核版本>`，确认 `modprobe -S <内核版本> -n -v nvidia` 指向原模块，再为该内核重建 initramfs 并重启。
若原驱动包已被替换，先恢复匹配的原驱动、用户态库和固件；不要混用版本。

## 4. 安装后验收

1. 重新生成环境报告，确认实际加载版本、BAR1 和双向 read/write 能力。
2. 运行双向 CUDA peer-copy 的数据正确性与带宽测试；[CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest)可用于带宽/延迟排查，不能单独代替完整数据校验。
3. 在真实 PLE 文件系统、实际读盘卡上完成 [严格 GDS 读写与校验](../../docs/GDS_NVME_P2PDMA_REPRODUCTION.md)，核对完整操作计数、P2PDMA 路径、数据一致性与无 CPU 中转回退。
4. 再验证模型加载、连续多轮对话与长输出；驱动变更后的模型测试结果须重新记录。

GPU↔GPU 与 SSD→GPU 是两项独立验收。补丁编译通过、BAR1 变大、P2P 显示 OK 都不能代替上述测试。
