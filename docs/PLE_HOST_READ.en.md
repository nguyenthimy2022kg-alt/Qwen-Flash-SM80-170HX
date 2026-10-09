# Optional host-read PLE backend

By default the PLE reader is strict GPUDirect Storage: `sync_mt` workers call `cuFileRead` straight into GPU
staging memory, and the reader refuses to start when cuFile compatibility mode is enabled. That stays the
default.

`Q38_PLE_HOST_READ=1` is an opt-in alternative for hosts where the direct NVMe→GPU path is unreliable or
unavailable. It is a workaround with a different I/O path, not a fix for a known root cause.

## What changes

- Each `sync_mt` worker gets a pinned host buffer (`cuMemHostAlloc`, one lane of the staging slot) instead of
  a cuFile handle, and fills it with `pread` on the same `O_DIRECT` descriptor and the same planned ranges.
- After all workers finish, the reader copies each lane to GPU staging with `cuMemcpyHtoDAsync` on the
  caller's stream, then launches the unchanged gather kernel on that stream.
- Before workers refill the host lanes for the next read, the reader waits on an event recorded after the
  previous copies. `close()` waits on it as well before workers free their lanes.
- cuFile compatibility mode is accepted in this mode. It only affects the one-off `sync` reads (the global
  scale at startup), because the workers no longer call cuFile.
- `batch` mode is refused, since it needs the worker cuFile handles. `stats()["host_read"]` reports the mode,
  and every reader logs a warning naming it when it starts.

Pinned host memory per reader equals its staging slot size (32 MiB by default).

## Enabling it

Set `"Q38_PLE_HOST_READ": "1"` in `config/runtime-env.json` and restart. The artifact can stay where it is:
on a GDS-capable NVMe this keeps direct I/O and the same access pattern, and only the GPU transfer moves to
host memory.

To serve the artifact from RAM instead, which takes the SSD out of the request path, the host needs about
56 GB of free memory for the 51.2 GB artifact:

```bash
sudo install -m 755 scripts/ple-ram-setup.sh /usr/local/sbin/
sudo install -m 644 scripts/qwen-ple-ram.service /etc/systemd/system/
sudoedit /etc/systemd/system/qwen-ple-ram.service      # set PLE_SOURCE, PLE_IDENTITY, RequiresMountsFor
sudo systemctl daemon-reload && sudo systemctl enable --now qwen-ple-ram.service
grep ID_FS_USAGE /run/udev/data/b1:0                    # must print ID_FS_USAGE=filesystem
```

The RAM disk is a `brd` block device with ext4, not tmpfs: cuFile resolves the block device behind the file
through udev even in compatibility mode, and refuses to register the file otherwise. The script adds a udev
rule for `ram0` and verifies that the copy matches the enrolled identity, so the existing identity file
stays valid. Then point `config/local.json` at the copy:

```json
"ple_artifact": "/mnt/ple-ram/ple-gds",
"cufile_config": "config/cufile-ram.json"
```

`config/cufile-ram.json` differs from `config/cufile.json` only in `allow_compat_mode: true`.

With the service running, `runs/<container>/service/cufile.log` shows
`using cuFile posix read/write with compat mode enabled`, and the read counter of the NVMe
(first field of `/sys/block/<nvme>/stat`) stays flat under load.

## Checking correctness

`scripts/check-ple-host-read.py` reads the same row IDs through `sync` (the existing `cuFileRead` path) and
`sync_mt` (host read) of one reader and compares the outputs byte for byte. It covers single rows, short and
long prompts, repeated rows, multi-chunk reads and two reads in flight at once. Run it in the image on the
PLE GPU while the service is stopped:

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v "<ple_artifact>:/ple:ro" -v "$PWD/config/<cufile config>:/config/cufile.json:ro" -v /run/udev:/run/udev:ro \
  -e CUFILE_ENV_PATH_JSON=/config/cufile.json \
  qwen-flash-sm80:0.2.0 python3 /opt/qwen-flash-sm80/scripts/check-ple-host-read.py \
  --gpu-uuid <GPU-...> --gpu-bdf <0000:xx:00.0>
```

It prints `OK: ...` or exits 1 at the first mismatch. UUID and BDF are the values `serve.py start --dry-run`
passes as `VLLM_PLE_GDS_GPU_UUID` and `VLLM_PLE_GDS_GPU_BDF`.

## Rolling back

Set `Q38_PLE_HOST_READ` back to `"0"`, restore `ple_artifact` and `cufile_config` in `config/local.json`,
and restart. To free the memory:
`sudo systemctl disable --now qwen-ple-ram && sudo umount /mnt/ple-ram && sudo rmmod brd`.
