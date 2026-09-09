# CMP 170HX P2P 部署参考

**简体中文** | [English](CMP_P2P.en.md)

本项目参考环境的 GPU P2P 支持使用 [qg19932GH/cmpunlocker](https://github.com/qg19932GH/cmpunlocker)。该项目运行在驱动层，独立于 vLLM；本仓库不包含其驱动补丁，也不会在镜像构建或服务启动时自动安装。

2026-09-08 核查的上游提交为 [`aaddfd4ce84a2804a7e0cd332acc4c26c79063d9`](https://github.com/qg19932GH/cmpunlocker/tree/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9)。这是文档审查版本，**尚未核实它是否等于参考机器最初安装驱动时使用的提交**。

## 适用定位

主机应满足 [主机验收条件](从零部署.md#主机验收条件)。cmpunlocker 和下述本地 overlay 是参考机器的适配方式，不是 vLLM 的必需依赖；已有可用 P2P/GDS 的机器无需安装相同补丁。其他机器可按硬件选择实现方式，以实际传输和模型运行验证为准。

## 本项目实际参考状态

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

本仓库不分发该机器专用的驱动 overlay，也不要求部署者取得它。上面的差异用于说明参考环境来源；目标是实现所需的 P2P/GDS 能力，而非复刻这套驱动。尚未满足条件的 CMP 机器仍需根据自身硬件完成适配。

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
