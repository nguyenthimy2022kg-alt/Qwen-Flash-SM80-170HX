# Qwen-Flash-SM80

**面向双 SM80 显卡的 Qwen3.8-Flash-Next 社区 vLLM 运行方案。**

结合 **GPUDirect Storage（GDS）**、**TEP2**、**MTP6** 和形状专用内核，在双 CMP 170HX 上运行 NVFP4 主模型，并从 SSD 直接读取 FP8 PLE 数据。

已记录的单请求长输出测试：**平均 decode 吞吐量 170.826 tok/s**，生成 29,010 token 后自然结束。提示词为“写个网站网页”。

这是针对特定模型、版本和双卡硬件整理的社区方案，与通用上游 vLLM 的安装包有所不同。已验证硬件为双 CMP 170HX（SM80、每卡约 63.39 GiB 显存）；其他 SM80 设备及模型的兼容性与性能尚未验证。

## 已集成优化

| 已采用的优化 | 实现 | 已有结果 |
|---|---|---|
| GDS＋PLE 接入 | 通过 cuFile 直接读取 PLE 数据至显存，接入输入准备与缓冲区生命周期 | 数据及生命周期验证通过；未单独测量整模型收益 |
| C++ 读取计划器 | 将读取任务规划移至 C++，减少 Python 对象构造 | 历史 4096 行规划约 8.655→1.188 ms，仅为局部耗时 |
| 常驻读取线程、固定缓冲区 | 复用线程、读取句柄与工作区 | 当前配置为 32 个读取线程、3 个缓冲槽位 |
| 同进程异步读取与交接修复 | 重叠 I/O 与模型计算，减少跨进程等待，修复输入过早复用 | 历史组合版本 51.33→66.74 tok/s，非单项独立收益 |
| TEP2 与 P2P 通信 | 组合张量并行与专家并行，使用双卡直接通信 | 早期 MTP6 各五轮汇总：TEP2 131.044、TP2 127.820 tok/s |
| HC 单行 GEMV | 为 HC 单行输入实现专用矩阵向量乘内核 | 当时 TP2 对照 74.55→80.02 tok/s |
| TileLang 输入投影 | 融合单行 QKVZ/BA 投影，复用原有输出缓冲区 | 已集成；未单独测量整模型收益 |
| CUDA Graph 与融合算子 | 复用计算图，减少内核提交开销 | 保留整体计算图与原有 MTP GDN 融合路径 |
| MTP6＋七行 PLE 行号规划 | 对多 token 验证使用 NumPy 行号规划快路径 | 整理版历史五轮汇总 131.044；行号规划版随后单次 158.013 tok/s |
| 全词表 INT8 草稿打分＋BF16 复核 | INT8 计算完整词表分数，每卡 top-32 候选使用 BF16 权重复核，再汇总最优候选 | 单次 170.826 tok/s；历史前版单次 158.013，观察差异约 +8.1% |
| 加载内存保护 | 限制容器内存，在主机可用内存低于阈值时停止受管服务 | 保留加载修复与内存保护；未计入 decode 收益 |

表中记录来自不同阶段和条件，**不能相加或相乘**。最新 INT8 与历史版本的输出内容不同，8.1% 是观察差异，尚不能据此确定量化的独立收益。[查看完整测试条件](docs/性能记录.md)。

## 快速开始

```bash
git clone https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80.git
cd Qwen-Flash-SM80
```

需要 Linux、Docker、NVIDIA Container Toolkit、兼容 CUDA 13 的驱动，以及正常工作的 GDS/cuFile 与 GPU P2P。两卡合计显存和 SSD 数据布局必须满足当前模型要求。`allow_compat_mode=false`：GDS 不可用时不会静默退回主机中转。

1. 准备含 MTP 权重的对应 NVFP4 模型，以及 **FP8 PLE 检查点和 GDS 数据**。仓库不包含模型权重。[数据准备说明](docs/使用说明.md#模型与-ple-数据)
2. 构建固定上游版本的镜像：

   ```bash
   docker build -t qwen-flash-sm80:0.1.1 .
   ```

3. 复制示例配置，设置模型父目录、模型子目录、PLE 目录和 GPU 标识：

   ```bash
   cp config/example.json config/local.json
   ```

4. 按使用说明生成 PLE 数据校验配置后，检查启动命令，再启动：

   ```bash
   python3 scripts/serve.py start --config config/local.json --dry-run
   python3 scripts/serve.py start --config config/local.json
   ```

默认仅监听本机 `127.0.0.1:18420`，模型名 `qwen3.8-flash-next`，提供 OpenAI 兼容接口。首次启动包括编译和模型预热。启动命令返回后，使用 `/health` 确认服务就绪。

停止时使用启动器打印的容器名：

```bash
python3 scripts/serve.py stop --name <容器名>
```

[完整使用说明、回退与验证范围](docs/使用说明.md) · [源码来源与许可证](docs/来源与许可.md)

## 仓库内容

- `src/`：相对固定上游的运行源码，包括 GDS、HC、TileLang 和草稿 INT8。
- `csrc/`：GDS 读取扩展的 C++ 源码；构建时生成本机扩展。
- `src/preload/`：该固定 SM80 环境使用的 Triton 预加载内核和 SHA256 清单，约 17 MB，不含模型权重。
- `patches/`：上游源码校验值与来源分类，防止误覆盖其他版本。
- `config/`、`scripts/`：中文配置说明、启动/停止、数据准备及构建入口。
- `docs/`：已采用优化、性能数据与使用范围。公开文档仅提供中文版本。

## 致谢与许可

感谢 vLLM、Qwen 模型实现、原 Qwen3.8-Flash-DGX 社区项目，以及 NVIDIA GDS、Marlin、Triton、TileLang 等项目。仓库保留引用源码的既有声明，项目代码采用 Apache-2.0；模型、CUDA/cuFile、容器与第三方依赖按各自许可证使用。详见[来源与许可](docs/来源与许可.md)。
