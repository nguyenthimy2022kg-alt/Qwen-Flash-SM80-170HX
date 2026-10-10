# PR #2：可选读取路径验证 / Optional read-path validation

[中文使用说明](PLE_HOST_READ.md) · [English setup](PLE_HOST_READ.en.md) · [PR #2](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX/pull/2)

2026-10-09（PDT），在维护者的 H12D-8D、CMP 170HX / SM80、Linux 6.17、驱动 610.43.03、cuFile 1.15.1.6 环境检查。使用原有 NVMe 上的 NVIDIA 检查点 PLE 数据，32 个读取线程；未加载整模型，未创建 51.2 GB 内存盘。默认开关仍为 `Q38_PLE_HOST_READ=0`。

| 检查 | 结果 |
|---|---|
| Python / Shell 语法与 CPU 回归 | 47 项通过，包含数据损坏、复制失败、已有设备保护、低内存保护和只读复用 |
| 严格 GDS，`--host-read 0` | 20 轮，1,217,180 行与同步 cuFile 结果逐字节一致 |
| SSD＋主机中转，`--host-read 1` | 20 轮，1,217,180 行与同步 cuFile 结果逐字节一致 |
| 仍有异步复制和消费任务时关闭读取器 | 关闭前确认队列未完成；关闭后 2,048 行逐字节一致，重复关闭正常 |
| 严格模式保护 | 兼容模式配置配合 `--host-read 0` 被拒绝，未静默回退 |
| 运行文件 | 596 项哈希已核对；相对 v0.2.0，仅原生 PLE 读取库及其 Python 包装文件改变 |

合并前修正了检查脚本的队列顺序、复制异常及退出时的缓冲区寿命，重写了内存盘保护逻辑，补齐中文说明和构建哈希。RAM 盘脚本只复制已登记的当前数据，核验完整内容；不会格式化已有但不属于本配置的设备。

**验证边界：**本次证明有限读取案例的结果一致，未重跑整模型长期验收，也未在本机实测完整 `brd` 内存盘。贡献者在 PR 中报告约 79 小时、986 次请求未出现 `PLE readiness failed`，这是其大内存机器上的原型记录，不等于本次修订版本的完整验收。另一台机器仍需按使用说明运行检查。

机器可读结果见 [host-read-validation.json](host-read-validation.json)。原 v0.2.0 的整模型验收独立保留，本机常用镜像未被覆盖，测试结束后模型保持停止。

## English

Checked on 2026-10-09 (PDT), using the maintainer's H12D-8D / CMP 170HX (SM80), Linux 6.17, driver 610.43.03 and cuFile 1.15.1.6. Both paths used the same NVIDIA checkpoint PLE artifact on NVMe and 32 reader workers. No full model or 51.2 GB RAM disk was loaded. `Q38_PLE_HOST_READ=0` remains the default.

- Python/Shell syntax and all **47 CPU tests** passed, including corruption, failed-copy cleanup, unowned-device refusal, low-memory guards and read-only reuse.
- Strict GDS and host-mediated SSD reads each passed **20 rounds / 1,217,180 byte-identical rows** against synchronous cuFile reads.
- Closing with pending copies/consumer work preserved **2,048 rows**, waited for completion and allowed a repeated close.
- Strict mode rejected compatibility-mode configuration instead of silently falling back.
- **596 runtime hashes** were checked. Only the native persistent PLE reader and its Python wrapper differ from v0.2.0.

The review fixed test-stream ordering, buffer lifetime on copy errors/shutdown, RAM-disk ownership and full-content verification, and added Chinese documentation plus updated build hashes.

**Scope:** these are bounded reader checks, not a new full-model endurance test or a local full-size `brd` test. The contributor's approximately 79 hours / 986 requests without `PLE readiness failed` were reported for their prototype on a different host; they are not an endurance result for this revision. Run the documented checks on each host. The existing local model image was not replaced, and the model remains stopped.
