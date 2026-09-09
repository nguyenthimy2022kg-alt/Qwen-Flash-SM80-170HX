# 参考硬件与 PCIe 拓扑

**简体中文** | [English](REFERENCE_HARDWARE.en.md)

2026-09-08 通过本机 `lspci -Dtv`、sysfs、`findmnt`、`lscpu` 和 SMBIOS 只读核查。以下是已跑通本项目的参考配置，不是其他机器的兼容性清单。

## 是否需要 PCIe switch 板？

**本机 GPU 和 SSD 的连接路径中没有 PCIe switch（PCIe 交换芯片）。SSD 与显卡也不是挂在同一个 switch 下。** 它们连接到 CPU 的 PCIe 根端口；990 PRO 与 `c5:00.0` 显卡属于同一根总线下的不同根端口。另一张 `01:00.0` 显卡属于另一组根总线。

这里描述的是软件枚举出的连接关系，不能据此辨认是否使用无交换芯片的延长线或被动转接板；这些配件不等于 PCIe switch。

## 已验证的硬件

| 部件 | 参考配置 |
|---|---|
| CPU | 单路 AMD EPYC 7532，32 核；系统显示一个 NUMA 节点 |
| 主板 | SMBIOS 报告为 HUANANZHI H12D-8D V2.0；未按实物丝印核对 |
| GPU | 两张 NVIDIA CMP 170HX，SM80，每卡标称 64 GB；运行可用容量见部署指南 |
| PLE 数据盘 | Samsung SSD 990 PRO 4TB，`0000:c4:00.0`，本机为 `/dev/nvme1n1` |
| 数据文件系统 | `/dev/nvme1n1p2`，ext4，实际挂载含 `data=ordered` |
| 主机内存 | 约 32 GiB RAM、8 GiB swap；这是参考配置，不是宽裕内存建议 |
| 本次读取的链路 | SSD：PCIe 4.0 ×4；`c5` GPU：PCIe 2.0 ×16；`01` GPU：PCIe 2.0 ×4 |
| 软件基础 | Ubuntu 24.04、Linux 6.17.0-23、NVIDIA 610.43.03 Open Kernel Module、CUDA 13.0 GDS；包含本机 CMP 驱动适配 |

链路为参考机器当前协商状态，不是设备通用规格或 GDS 的最低要求。另一块 Kingston NV2 不是本次 PLE 数据所在盘。

## 实际连接图

省略与数据路径无关的设备。BDF（PCI 设备地址）可能随插槽、BIOS 或重启变化。

```text
单路 AMD EPYC 7532
├─ 根总线 0000:c0
│  ├─ 根端口 0000:c0:01.5 ─ Samsung 990 PRO   0000:c4:00.0
│  └─ 根端口 0000:c0:03.1 ─ CMP 170HX        0000:c5:00.0
└─ 根总线 0000:00
   └─ 根端口 0000:00:01.1 ─ CMP 170HX        0000:01:00.0
```

`nvidia-smi topo -m` 中双 GPU 的关系为 `NODE`：同一 NUMA 节点内经过 PCIe Host Bridge 之间的互连。它不表示两卡位于同一 switch，也不能单凭这张 GPU 表确定 SSD 拓扑。

## 当前项目的数据路径

```text
990 PRO 上的 PLE 数据
  ── strict cuFile / NVMe P2PDMA ──> c5 GPU（TP rank 0）
  ── 由 c5 GPU 通过 NCCL 广播 ─────> 01 GPU（TP rank 1）
```

当前 TEP2 由一张卡读盘，再把本步所需的 PLE 行广播给另一张卡；没有让 SSD 分别向两卡重复读取。启动器把 `config/local.json` 的 `gpu_ids` 第一项设为读盘卡；参考配置第一项是 `c5` 对应的 UUID。其他机器应选择已通过 strict 读取校验的卡，并核实排序。实现见 [启动器](../scripts/serve.py) 和 [PLE 双卡传输](../src/tp_ple_transport.py)。

**SSD 直通显存指数据本体不在 CPU 内存中暂存，并不表示完全绕过 CPU 芯片。** 本机数据经过 CPU 内的 PCIe 互连路由；CPU 仍处理读取计划、文件系统和提交操作。[NVIDIA GDS 设计说明](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) 区分了数据路径和控制路径。

## 换一台机器应怎样判断？

同一 switch 可能提供更短的设备间路径，但不是本机成功的前提，也不是买一张 switch 板就必然成功。跨根端口能否路由还取决于 CPU 平台、内核和驱动；Linux 对这类 P2PDMA 路径有兼容性检查。不能把“同一个 CPU”或“同一 NUMA 节点”当成直通已成功的证据。参见 [Linux P2PDMA 路由说明](https://docs.kernel.org/driver-api/pci/p2pdma.html)。

先定位真正存放 PLE 文件的盘，再检查它与读盘 GPU 的完整上级路径：

```bash
PLE_DIR=/absolute/path/to/ple-artifact
findmnt -T "$PLE_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -d -o NAME,MODEL,SIZE,TRAN
lspci -Dtv
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv
nvidia-smi topo -m
# 将 nvme1 和 BDF 替换为目标机器实际值。
readlink -f /sys/class/nvme/nvme1/device
readlink -f /sys/bus/pci/devices/0000:c5:00.0
```

随后分别验证 [GPU ↔ GPU P2P](CMP_P2P.md) 和 [SSD → GPU strict 读取](GDS_NVME_P2PDMA_REPRODUCTION.md)。后者需要实际数据校验和本次读取日志，不能只看能力矩阵或工具退出码。本次文档核查未重新测速，也未改驱动、BIOS 或硬件设置。
