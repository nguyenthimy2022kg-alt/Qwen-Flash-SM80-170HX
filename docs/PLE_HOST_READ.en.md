# Optional PLE reads through host memory

[简体中文](PLE_HOST_READ.md) | **English** · [Home](../README.en.md)

**The default remains SSD → GPU GDS.** This optional path reads PLE rows into a small pinned host buffer, then copies them to the GPU. It is useful for investigating direct-read failures on another host. It is not a confirmed fix for every hang, and does not change model computation or enable concurrent requests.

This feature was added on `main` after v0.2.0. **Rebuild the image to use it; an existing v0.2.0 image does not contain it.** No RAM disk is required: try the existing SSD artifact first.

## Use the existing SSD

With the model stopped, update the repository and build a separate image:

```bash
git pull --ff-only
docker build -t qwen-flash-sm80:host-read .
```

1. Set `"image": "qwen-flash-sm80:host-read"` in `config/local.json`.
2. Set `"Q38_PLE_HOST_READ": "1"` in `config/runtime-env.json`.
3. Keep the existing `ple_artifact`, `ple_identity` and strict `cufile_config` paths for the first check.
4. Run the byte-comparison check below, then start with the usual `python3 scripts/serve.py start` command.

`sync_mt` workers use `pread(O_DIRECT)` and pinned host memory. Copies and gather operations run on the caller's CUDA stream, in order. A completion event prevents refilling/freeing buffers while the GPU is still reading them. Additional pinned memory is at most the staging-slot size per reader (32 MiB by default), not the entire 51.2 GB PLE table.

**cuFile is still required:** reader initialization, file/buffer registration and the one-off global-scale read retain the existing implementation. If the host cannot initialize cuFile or register the artifact, this switch alone will not make it work. `batch` mode is unsupported with host reads. The normal service uses `sync_mt`; reader logs and `stats()["host_read"]` identify the selected mode.

## Optional: keep the table in a RAM disk

Only consider this with ample host RAM. The roughly **47.7 GiB** payload is separate from model and OS memory. The setup script requires payload size + **24 GiB reserve** + 1 GiB filesystem allowance: approximately **73 GiB of currently available RAM** for this table. The reserve is a minimum guard, not a prediction of your model's peak usage; increase `PLE_RAM_RESERVE_GIB` as needed. A 32 GiB host should use the SSD path.

The RAM disk uses Linux `brd` with ext4, **not tmpfs**. cuFile still needs to resolve the file's block device through udev in compatibility mode.

Install both scripts, then substitute your existing artifact and enrolled identity paths:

```bash
sudo install -m 755 scripts/ple-ram-setup.sh scripts/ple-ram-artifact.py /usr/local/sbin/
sudo env PLE_SOURCE=/data/ple-gds \
  PLE_IDENTITY=/opt/Qwen-Flash-SM80-170HX/config/local-ple-identity.json \
  /usr/local/sbin/ple-ram-setup.sh
```

The default device is `/dev/ram0`, mount point `/mnt/ple-ram`, capacity 52 GiB. The script refuses existing unowned `brd` devices and nonempty mount points. It copies **only the enrolled active generation**, checks the complete payload hash, and remounts read-only. Repeated execution verifies the previously owned disk instead of formatting it. Failed setup is left for inspection; retries will not overwrite it.

Use the destination printed by the script in `config/local.json`, retaining the original identity:

```json
"image": "qwen-flash-sm80:host-read",
"ple_artifact": "/mnt/ple-ram/ple-gds",
"cufile_config": "config/cufile-ram.json"
```

The final directory name follows `PLE_SOURCE`, so replace `ple-gds` if yours differs. `config/cufile-ram.json` enables compatibility mode; it must be paired with `Q38_PLE_HOST_READ=1`. Verify that `grep ID_FS_USAGE /run/udev/data/b1:0` prints `ID_FS_USAGE=filesystem` before running the reader check. Do not use a RAM artifact with the default strict-GDS mode.

For optional boot-time setup, install `scripts/qwen-ple-ram.service` into `/etc/systemd/system/`, edit `PLE_SOURCE`, `PLE_IDENTITY` and `RequiresMountsFor` to your paths, then run `sudo systemctl daemon-reload` and `sudo systemctl enable --now qwen-ple-ram.service`. The model must start after this service completes. Each reboot reconstructs and verifies the RAM copy from the SSD.

## Check correctness before serving

Run with the model stopped. Replace the artifact path and GPU values with those from `python3 scripts/serve.py start --dry-run` (`VLLM_PLE_GDS_GPU_UUID` / `_BDF`). Use `config/cufile.json` for the SSD; replace it with `config/cufile-ram.json` for the RAM disk.

```bash
docker run --rm --gpus all --ipc=host \
  -v /data/ple-gds:/ple:ro \
  -v "$PWD/config/cufile.json:/config/cufile.json:ro" \
  -v /run/udev:/run/udev:ro \
  -e CUFILE_ENV_PATH_JSON=/config/cufile.json -e CUFILE_USE_PCIP2PDMA=1 \
  qwen-flash-sm80:host-read python3 /opt/qwen-flash-sm80/scripts/check-ple-host-read.py \
  --gpu-uuid GPU-YOUR-UUID --gpu-bdf 0000:xx:00.0 --rounds 20 --host-read 1
```

The check compares `sync_mt` with the existing synchronous cuFile path byte for byte: first/last rows, random/repeated rows, long inputs, multi-chunk reads and two outstanding reads on separate CUDA streams. It prints `OK` or fails at the first mismatch. `--host-read 0` checks the default workers and requires strict GDS on a suitable SSD. This is a reader check, not a complete model-quality or long-run stability test. Validation scope is recorded in [PR #2 checks](HOST_READ_VALIDATION.md).

## Return to direct GDS

Stop the model, set `Q38_PLE_HOST_READ` to `"0"`, restore the SSD artifact and `config/cufile.json`, then restart. You can retain the rebuilt image or select your previous image.

RAM storage is volatile. To retire automatic setup, disable the optional service with `sudo systemctl disable --now qwen-ple-ram.service`; this alone does not unmount or free the disk. After stopping its users, a reboot releases it. For manual cleanup, first check `/run/qwen-ple-ram.owner`, mounts and all users of `brd`; do not unload the module if another RAM disk is in use. The original SSD artifact is never removed by these scripts.
