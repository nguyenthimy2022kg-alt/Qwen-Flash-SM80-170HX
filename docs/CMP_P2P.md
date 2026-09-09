# CMP 170HX：P2P、BAR1 与 GDS

**简体中文** | [English](CMP_P2P.en.md) · [项目首页](../README.md)

本项目用两条直传通路完成双卡推理：**GPU ↔ GPU 用于交换计算数据，SSD → GPU 用于读取 PLE 查表数据。** 两条通路使用不同的软件接口，需要分别验证。

## 各组件负责什么

| 组件 | 在本项目中的职责 |
|---|---|
| **P2P（设备直接互传）** | 本文讨论双卡通信时，指 GPU 之间直接交换数据，减少 CPU 内存中转 |
| **BAR1（访问显存的 PCIe 地址窗口）** | 让外部设备能够通过 PCIe 地址访问指定显存；窗口和映射都需要驱动支持 |
| **[cmpunlocker](https://github.com/qg19932GH/cmpunlocker)** | 提供 CMP 基础驱动适配，包括显存容量解锁、BAR1 扩容及 P2P 支持 |
| **本仓库附带的三个增量补丁** | 在 cmpunlocker 基础上接通 BAR1 互传映射，处理初始化冲突和平台限制；来源见 [补丁声明](../drivers/cmp-bar1/NOTICE.md) |
| **GDS / cuFile（NVIDIA 的存储直读软件与接口）** | 配合驱动和 Linux NVMe 支持，让 SSD 数据直接进入显存 |
| **本项目的 vLLM 读取程序** | 根据模型输入安排 PLE 查表读取，再把准备好的数据交给模型计算 |

准备脚本会下载固定版本的 NVIDIA 驱动源码和 cmpunlocker，再应用增量补丁。它只准备源码，安装步骤另见 [驱动构建指南](../drivers/cmp-bar1/README.md)。

## BAR1 扩容与映射有什么区别

**扩容决定窗口有多大，映射决定窗口里的地址对应哪块显存。** 显卡内部的地址不能直接作为其他设备访问它的地址，需要驱动建立对应关系：

```text
外部设备使用的 PCIe 地址 → BAR1 窗口 → 目标显存位置
```

在硬件与驱动允许的情况下，另一张 GPU 可以通过映射访问目标显存；NVMe 直读则需要 GDS 配合建立其数据传输所需的映射。**BAR1 变大不会增加物理显存，映射建立也不等于传输已经验证成功。**

当前项目的数据流是：

```text
SSD 上的 PLE 数据
    │ GDS / NVMe P2PDMA：直接读入显存
    ▼
读盘 GPU（rank 0）
    │ NCCL：双卡广播，使用已验证的 GPU P2P 通路
    ▼
另一张 GPU（rank 1）
```

当前由一张卡读盘后再广播；CPU 仍负责提交和安排任务，数据本体无需在 CPU 内存中转。具体读盘卡选择见 [硬件与拓扑](REFERENCE_HARDWARE.md)。

## 三个补丁分别解决什么

| 补丁 | 原来的阻碍 | 改动及适用范围 |
|---|---|---|
| **0011：BAR1 映射** | 选择了 BAR1 路径，还缺少适用的互传映射与地址转换 | 接通映射和页表处理；涉及驱动内部实现，换驱动版本需重新适配 |
| **0013：初始化冲突** | 驱动提前登记 mailbox（另一种互传机制），使 BAR1 路径因协议冲突被拒绝 | 选择 BAR1 时跳过 mailbox 预登记；适用于遇到同类冲突的环境 |
| **0015：平台读取限制，可选** | 驱动不认可平台或拓扑，返回“芯片组不支持读取” | 对设备 ID `0x20C2` / `0x2082` 覆盖为“允许读取”；不验证实际路由，需显式开启并实测 |

这些补丁没有写死参考机器的插槽地址，但也不是所有 CMP 主机通用。尤其是 0015，只改变驱动是否允许尝试，不能补足硬件能力。0011 还包含部分全局驱动行为修改；完整实现边界见 [补丁构建指南](../drivers/cmp-bar1/README.md#包含什么)。

**验证状态：** 参考机器原有整套驱动已通过双卡互传与 SSD 直读校验；公开整理版已通过 [完整编译](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/actions/runs/34310525583)，尚未装机验证。它们不是完全相同的构建。

## 应该从哪里开始

| 当前情况 | 下一步 |
|---|---|
| 双卡互传与严格 SSD 直读都已通过 | 按 [部署指南](从零部署.md)准备模型，无需安装相同补丁 |
| 双卡互传正常，SSD 直读未通过 | 按 [GDS 指南](GDS_NVME_P2PDMA_REPRODUCTION.md)检查存储路径，不因 SSD 失败就直接启用 0015 |
| CMP 的 BAR1/P2P 不可用 | 先做下面的只读检查，再按 [驱动构建指南](../drivers/cmp-bar1/README.md)定位和适配 |

在仓库根目录运行：

```bash
python3 scripts/check-gds-host.py --data-path /实际的/PLE目录
```

脚本只汇总环境，输出 `NOT_TESTED` 表示没有进行实际传输测试。最终以 [主机验收条件](从零部署.md#主机验收条件) 为准。
参考机器为 H12D，GPU 与 SSD 路径上没有枚举出的 PCIe switch；同 CPU、同 NUMA 或能力矩阵 `OK` 都不能代替实测。

## 本项目实际参考状态

2026-09-08 核查的上游提交为 [`aaddfd4ce84a2804a7e0cd332acc4c26c79063d9`](https://github.com/qg19932GH/cmpunlocker/tree/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9)。这是文档审查版本，**尚未核实它是否等于参考机器最初安装驱动时使用的提交**。

2026-09-08 在运行服务所在机器读取到的状态：

| 项目 | 已生效配置 |
|---|---|
| Linux / NVIDIA | `6.17.0-23-generic` / `610.43.03` Open Kernel Module |
| 驱动构建 | CMP v0.3 本地 overlay，包含额外 BAR1/P2P 补丁 |
| 双卡能力矩阵 | P2P read、write 双向均 `OK` |
| IOMMU | `amd_iommu=off iommu=off`，无 IOMMU 实例 |
| NVMe multipath | `N` |
| PLE 文件系统 | ext4，实际挂载选项含 `data=ordered` |
| NVIDIA 参数文件 | `/etc/modprobe.d/cmpunlocker-nvidia-options.conf` |
| BAR1/P2P | `EnableResizableBar=1`、`RMForceStaticBar1=1`、`RMPcieP2PType=1`、`RmForceDisableIomapWC=1` |
| 该机器 Gen2 配置 | `RmForceEnableGen2=1`、`RMPcieLinkSpeed=0x1` |

参考机器的驱动构建脚本还使用 `driver/local-src`、`driver/local-patches`，包括 `0011-p2p-bar1.patch`、`0013-skip-mailbox-peer-preinit.patch`、`0015-bar1p2p-readcap-override.patch` 等。这些本地 overlay 目录不存在于上面审查的外部提交中，不能将两者视为完全相同的驱动版本。

本仓库已将 BAR1/P2P 增量补丁整理到 [drivers/cmp-bar1](../drivers/cmp-bar1/README.md)，提供固定公开源码、校验、构建、手动安装与恢复说明。整理版不携带本机超频配置，也不等同于正在运行的完整 v0.3 驱动；平台读取能力覆盖需显式选择。旧补丁的同 switch 假设已经修正，当前参考拓扑仍以 H12D 实测为准。

## 外部项目的配置差异

按外部版本的 [README](https://github.com/qg19932GH/cmpunlocker/blob/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9/README.md) 和 [安装脚本](https://github.com/qg19932GH/cmpunlocker/blob/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9/install.sh) 核查：

- `--p2p` 用于显式启用 P2P 补丁，默认不启用。
- 默认安装会设置 CPU 的 IOMMU 启用参数及 `iommu=pt`；`--no-iommu` 只跳过修改，不会自动关闭已有 IOMMU。
- 上游 README 的功能表仍将 GPU P2P 标为 “In progress”。

这些信息仅用于定位差异，不将外部默认安装命令作为本项目已验证配置。当前环境为准，不能在正在运行模型的主机上直接套用外部驱动安装流程。A100 等原生具备 P2P 能力的设备不需要 CMP 专用补丁。

## 两条路径分别验证

| 路径 | 作用 | 验证方式 |
|---|---|---|
| GPU ↔ GPU P2P | TEP2/TP2 的双卡通信 | P2P read/write 能力矩阵，以及实际 CUDA peer copy 数据与带宽/延迟测试 |
| NVMe → GPU P2PDMA | 读取 PLE 数据至显存 | strict cuFile 数据校验、NVMe P2PDMA 日志及无主机中转回退 |

可先检查能力与拓扑：

```bash
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -m
```

能力矩阵只用于前置检查，不能替代实际数据传输测试。参考环境 P2P 已工作，不代表其他机器安装外部提交后必然成功。

NVMe 直通的完整参考步骤见 [GDS NVMe P2PDMA 复现指南](GDS_NVME_P2PDMA_REPRODUCTION.md)。GPU P2P 通过后仍需独立验证 GDS，不能据此直接判定 SSD 直通成功。
