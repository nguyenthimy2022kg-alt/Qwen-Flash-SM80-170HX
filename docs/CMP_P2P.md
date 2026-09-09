# CMP 170HX：P2P 与 SSD 直通

**简体中文** | [English](CMP_P2P.en.md) · [项目首页](../README.md)

本项目需要两条通路：**GPU ↔ GPU 交换计算数据，SSD → GPU 读取 PLE 查表数据。**

## 1. 选择驱动

已有可用双卡 P2P 和 SSD 直读的机器，可以直接[部署模型](从零部署.md)。

CMP 170HX 需要驱动适配时，优先使用 **[bayley/cmpunlocker 固定版本 `5a7bb4b`](https://github.com/bayley/cmpunlocker/tree/5a7bb4b7e5056306fe49e8b824787659abb19914)**。它已包含 BAR1 扩容、P2P 映射，以及原先单独整理的 `0011`、`0013`、`0015` 补丁，**不需要再叠加本仓库补丁**。

按该版本的安装说明准备匹配的 NVIDIA 驱动、固件和内核头文件，再安装并冷启动。安装器会修改主机配置，但不会自动完成每台机器的 P2P/GDS 适配；BAR1、IOMMU 等设置仍需结合平台核对。已有正常工作的驱动无需重装。

安装时明确使用 `--p2p --no-iommu`：前者启用 P2P 能力覆盖，后者避免安装器自动开启 IOMMU；已有 IOMMU 和静态 BAR1 设置继续按 GDS 指南核对。仅运行默认安装命令不等于开启这条路径。

## 各组件负责什么

| 组件 | 作用 |
|---|---|
| cmpunlocker 驱动 | 适配 CMP 的 BAR1 和 GPU P2P，使显存可以被其他设备直接访问 |
| NVIDIA GDS/cuFile＋Linux NVMe | 建立 SSD → GPU 的直读路径 |
| 本项目 | 按模型输入读取 PLE 数据，并在两张 GPU 之间分发 |

BAR1 是外部设备访问显存的地址窗口；扩容决定窗口大小，映射指定它对应的显存位置。

## 2. 检查双卡互传

在本项目仓库根目录执行：

```bash
python3 scripts/check-gds-host.py --data-path /实际的/PLE目录
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -m
```

检查脚本只收集环境，所以报告中的 `NOT_TESTED` 表示尚未进行真实传输。双向能力为 `OK` 后，还应执行实际 CUDA peer-copy 数据校验；[CUDA Samples 的 P2P 测试](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest)可检查带宽与延迟。

## 3. 配置 SSD 直通

按 [NVMe GDS 指南](GDS_NVME_P2PDMA_REPRODUCTION.md)完成 cuFile 配置和严格数据校验。GPU P2P 通过后，SSD 直读仍需单独验证。

本项目的数据流为：

```text
SSD 上的 PLE 数据 → GDS 直读 → 读盘 GPU → P2P 广播 → 另一张 GPU
```

`config/local.json` 的 `gpu_ids` 第一项是读盘 GPU。参考机器为 H12D＋EPYC 7532，SSD 与两张显卡的路径上没有 PCIe switch；[硬件与连接图](REFERENCE_HARDWARE.md)可供选卡和布线时参考。
