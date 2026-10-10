# 可选的 PLE 主机内存中转读取

**简体中文** | [English](PLE_HOST_READ.en.md) · [返回首页](../README.md)

**默认仍是 SSD → GPU 的 GDS 直通。** 此选项先将所需 PLE 数据读入一小块主机内存，再复制到显卡，可用于排查其他机器的直通读取故障。它不是所有卡住问题的通用修复，不改变模型计算，也不开放多请求并发。

此功能在 v0.2.0 之后加入 `main`。**需要重新构建镜像，已有 v0.2.0 镜像不含此功能。** 不要求内存盘，建议先使用现有 SSD 数据。

## 先使用现有 SSD

停止模型后，更新仓库并单独构建镜像：

```bash
git pull --ff-only
docker build -t qwen-flash-sm80:host-read .
```

1. 将 `config/local.json` 的镜像改为 `"image": "qwen-flash-sm80:host-read"`。
2. 将 `config/runtime-env.json` 中的 `Q38_PLE_HOST_READ` 改为 `"1"`。
3. 首次检查保留原来的 `ple_artifact`、`ple_identity` 和严格模式的 `cufile_config` 路径。
4. 运行下方字节一致性检查，通过后照常用 `python3 scripts/serve.py start` 启动。

`sync_mt`（多线程读取）改用 `pread(O_DIRECT)` 将数据读入固定主机缓冲区，再在同一条 CUDA stream（显卡任务队列）中完成复制和数据整理。完成事件保证上一轮复制结束后才复用或释放缓冲区。每个读取器额外固定的主机内存不超过一个暂存槽大小，默认约 **32 MiB**，不会整表载入 51.2 GB 数据。

**仍然需要 cuFile（NVIDIA 文件读取库）**：读取器初始化、文件与缓冲区注册、启动时一次全局缩放数据读取仍沿用原实现。如果 cuFile 初始化或文件注册不能通过，仅打开此开关不够。此模式不支持 `batch`（批量读取诊断模式）；正常服务使用 `sync_mt`。启动日志及 `stats()["host_read"]` 会显示实际选择。

## 可选：将整张表放入内存盘

仅适用于主机内存充足的机器。约 **47.7 GiB** 的表要另占内存，不能挤占模型和系统所需空间。脚本要求“实际数据大小＋**24 GiB 预留**＋1 GiB 文件系统余量”，此模型约需 **73 GiB 当前可用内存**。预留量是最低保护，不是模型峰值预测；按需调高 `PLE_RAM_RESERVE_GIB`。32 GiB 主机应继续用 SSD。

内存盘使用 Linux 的 `brd` 块设备和 ext4，**不是 tmpfs**；cuFile 即使在兼容模式下也需要通过 udev 识别文件所在块设备。

安装两个脚本，将以下路径替换为自己的现有 PLE 数据和身份文件：

```bash
sudo install -m 755 scripts/ple-ram-setup.sh scripts/ple-ram-artifact.py /usr/local/sbin/
sudo env PLE_SOURCE=/data/ple-gds \
  PLE_IDENTITY=/opt/Qwen-Flash-SM80-170HX/config/local-ple-identity.json \
  /usr/local/sbin/ple-ram-setup.sh
```

默认使用 `/dev/ram0`，挂载到 `/mnt/ple-ram`，容量 52 GiB。脚本会拒绝已有且不属于本次配置的 `brd` 设备及非空挂载目录；**只复制身份文件指定的当前数据版本**，校验完整内容后改为只读挂载。重复运行只校验已创建的内存盘，不重新格式化；中途失败会保留现场，重试也不会覆盖。

将脚本输出的目标目录填入 `config/local.json`，继续使用原身份文件：

```json
"image": "qwen-flash-sm80:host-read",
"ple_artifact": "/mnt/ple-ram/ple-gds",
"cufile_config": "config/cufile-ram.json"
```

末级目录名随 `PLE_SOURCE` 而定，不一定叫 `ple-gds`。`cufile-ram.json` 启用兼容模式，必须配合 `Q38_PLE_HOST_READ=1`。检查 `grep ID_FS_USAGE /run/udev/data/b1:0` 应输出 `ID_FS_USAGE=filesystem`，再运行读取检查。内存盘不能配合默认的严格 GDS 模式。

如需开机自动准备，将 `scripts/qwen-ple-ram.service` 安装到 `/etc/systemd/system/`，修改其中的 `PLE_SOURCE`、`PLE_IDENTITY` 和 `RequiresMountsFor`，然后执行 `sudo systemctl daemon-reload` 及 `sudo systemctl enable --now qwen-ple-ram.service`。模型必须等它成功完成后再启动；每次重启都会从 SSD 重建并校验内存副本。

## 启动模型前检查正确性

在模型停止时运行。数据路径填写自己的目录；显卡 UUID 和 BDF 从 `python3 scripts/serve.py start --dry-run` 输出的 `VLLM_PLE_GDS_GPU_UUID`、`_BDF` 中取得。SSD 使用 `config/cufile.json`，内存盘则替换为 `config/cufile-ram.json`。

```bash
docker run --rm --gpus all --ipc=host \
  -v /data/ple-gds:/ple:ro \
  -v "$PWD/config/cufile.json:/config/cufile.json:ro" \
  -v /run/udev:/run/udev:ro \
  -e CUFILE_ENV_PATH_JSON=/config/cufile.json -e CUFILE_USE_PCIP2PDMA=1 \
  qwen-flash-sm80:host-read python3 /opt/qwen-flash-sm80/scripts/check-ple-host-read.py \
  --gpu-uuid GPU-YOUR-UUID --gpu-bdf 0000:xx:00.0 --rounds 20 --host-read 1
```

检查将新读取结果与原来的同步 cuFile 读取逐字节比较，覆盖首尾行、随机及重复行、长输入、分块读取，以及不同显卡任务队列中同时持有两份结果。通过会输出 `OK`，不一致会立即报错。`--host-read 0` 可检查默认多线程路径，但要求合适的 SSD 和严格 GDS 配置。这只是读取器验证，不等同于完整模型质量或长期稳定性验收。[本次验证范围](HOST_READ_VALIDATION.md)。

## 恢复直通

停止模型，将 `Q38_PLE_HOST_READ` 改回 `"0"`，恢复 SSD 数据路径和 `config/cufile.json`，再启动。镜像可以继续使用新构建的，也可选回原镜像。

内存盘断电即失。如不再需要自动准备，可执行 `sudo systemctl disable --now qwen-ple-ram.service`；这一步不会自动卸载或释放内存盘。停止使用它的程序后，重启主机即可释放。手动清理前应核对 `/run/qwen-ple-ram.owner`、挂载点和所有 `brd` 使用者，不要在其他内存盘仍使用时卸载整个模块。上述脚本不会删除 SSD 原数据。
