# Deployment

[简体中文](从零部署.md) | **English** · [Home](../README.en.md)

v0.2.0 uses the pinned NVIDIA checkpoint. Run the commands on the Linux server, from the repository root after cloning. The repository contains no model weights or prebuilt image.

## Environment and capacity

- Linux x86_64, Python 3.10+, Git, Docker, NVIDIA Container Toolkit and a CUDA 13-compatible driver.
- Validated machine: two CMP 170HX SM80 GPUs, about 64 GiB per GPU; about 32 GiB host RAM and 8 GiB swap. Other GPUs have not completed full-model validation.
- The original checkpoint occupies 132.68 GB. GDS conversion adds approximately 51.2 GB; the loading view adds 2.52 GB on the same filesystem using hard links. Keep the original files. Reserve at least 220 GB, plus Docker images and build/cache space; across filesystems the view copies the other weights too and needs about 80 GB more.
- Container limits: 21 GiB RAM, 26 GiB RAM + swap. The launcher stops its service when host available RAM falls below 2 GiB.

## Host prerequisites

GPU P2P and GDS are separate checks. CMP users can follow [bayley/cmpunlocker](https://github.com/bayley/cmpunlocker/tree/5a7bb4b7e5056306fe49e8b824787659abb19914) for BAR1/P2P, then the [NVMe GDS guide](GDS_NVME_P2PDMA_REPRODUCTION.en.md) for actual SSD-to-GPU reads with CPU fallback disabled. No extra driver patch from this repository is needed. The container does not configure host drivers.

The [reference H12D machine](REFERENCE_HARDWARE.en.md) does not use a PCIe switch board. Choose the first GPU as the PLE reader according to the SSD/GPU topology. Existing working hosts can skip host reconfiguration.

## 1. Download source and model

Use a data directory on the NVMe filesystem that passed GDS validation. Adjust all paths consistently if it is not your home filesystem.

```bash
git clone https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX.git
cd Qwen-Flash-SM80-170HX
python3 -m venv .venv-hf
.venv-hf/bin/pip install 'huggingface_hub==1.29.0'
mkdir -p "$HOME/qwen-flash-data/models"
.venv-hf/bin/hf download nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --local-dir "$HOME/qwen-flash-data/models/NVIDIA-Qwen3.8-Flash-Next-NVFP4" \
  --max-workers 2
```

The [pinned NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47) has NVFP4 main weights, mixed FP8/BF16 MTP weights and FP8 PLE. Re-run the command to resume a download.

## 2. Check files and prepare the loading view

```bash
python3 scripts/check-model.py \
  --model "$HOME/qwen-flash-data/models/NVIDIA-Qwen3.8-Flash-Next-NVFP4"
python3 scripts/prepare-nvidia-view.py \
  --source "$HOME/qwen-flash-data/models/NVIDIA-Qwen3.8-Flash-Next-NVFP4" \
  --target "$HOME/qwen-flash-data/models/NVIDIA-Qwen3.8-Flash-Next-GDS"
```

The first command checks pinned metadata hashes and all indexed tensor headers and lengths; it does not hash every weight payload. The second extracts the MTP weights and PLE scale from the combined shard, verifies copied payloads, and excludes PLE rows from the model loading index. This avoids scanning the entire PLE table during model loading. It preserves the original checkpoint and refuses an existing target directory.

## 3. Convert and register GDS data

```bash
python3 scripts/prepare-ple.py mapping \
  --index "$HOME/qwen-flash-data/models/NVIDIA-Qwen3.8-Flash-Next-NVFP4/model.safetensors.index.json" \
  --output "$HOME/qwen-flash-data/ple-mapping.json"
python3 scripts/convert-ple-to-gds-layout.py \
  --checkpoint-index "$HOME/qwen-flash-data/models/NVIDIA-Qwen3.8-Flash-Next-NVFP4/model.safetensors.index.json" \
  --mapping "$HOME/qwen-flash-data/ple-mapping.json" \
  --output "$HOME/qwen-flash-data/ple-gds" --compact --block-rows 256
python3 scripts/prepare-ple.py enroll \
  --artifact "$HOME/qwen-flash-data/ple-gds" \
  --output config/local-ple-identity.json
```

Use the **original checkpoint index**, not the loading view, for conversion. Conversion streams the table without loading all 51.2 GB into RAM; enrollment verifies the output. Existing verified NVIDIA artifacts can be reused. Do not reuse RadixArk PLE with NVIDIA weights.

## 4. Build and configure

```bash
docker build -t qwen-flash-sm80:0.2.0 .
cp -n config/example.json config/local.json
python3 - <<'PYCONFIG'
import json
from pathlib import Path
p = Path('config/local.json')
c = json.loads(p.read_text())
root = Path.home() / 'qwen-flash-data'
c.update(image='qwen-flash-sm80:0.2.0', models_root=str(root / 'models'),
         model_subdir='NVIDIA-Qwen3.8-Flash-Next-GDS', ple_artifact=str(root / 'ple-gds'))
p.write_text(json.dumps(c, indent=2) + '\n')
PYCONFIG
```

Review `gpu_ids`, paths and `port` in `config/local.json` before starting. Relative paths resolve from the repository root. The default API/model name remains unchanged. The build compiles the PCIe communication module without requiring GPU access; running it still requires working P2P. If Docker lacks buildx, install it or use `DOCKER_BUILDKIT=0 docker build ...`.

## 5. Start and connect

```bash
python3 scripts/serve.py start --config config/local.json --dry-run
python3 scripts/serve.py start --config config/local.json
curl --fail http://127.0.0.1:18420/health
curl --fail http://127.0.0.1:18420/v1/models
```

The launcher rejects another concurrently running service carrying this project’s label. Wait for loading and warmup to finish and `/health` to return HTTP 200. `--dry-run` prints the command without loading the model. Logs are in `runs/<container>/`; use `docker logs -f <container>` for startup progress.

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:18420/v1` |
| Model | `qwen3.8-flash-next` |
| API key | `local` (placeholder) |

```bash
curl --no-buffer --fail-with-body http://127.0.0.1:18420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"写个网站网页"}],"max_tokens":50000,"stream":true}'
```

Default: thinking enabled, `preserve_thinking=true`, `reasoning_effort=xhigh`; temperature 1, top-p 0.95, top-k 20. For timing tests send `"temperature":0` and `"chat_template_kwargs":{"enable_thinking":false}`. Tool calling uses `qwen3_coder`; send `tools` and `tool_choice` in requests. The client must return tool results and any reasoning history it needs preserved.

One request runs at a time. The combined context limit is 262,144 tokens; output is limited by both `max_tokens` and remaining context. Images: up to 128 configured, subject to actual context/VRAM; video is disabled. Prefix matching uses 36-token units while physical blocks remain 3,456 tokens. Do not enable async scheduling or increase active request slots with this release.

The service binds to localhost without authentication. For remote access, run SSH forwarding on the client machine: `ssh -N -L 18420:127.0.0.1:18420 user@server`.

### Stop

```bash
python3 scripts/serve.py stop --name "<container>"
```

## Troubleshooting

- PLE/GDS errors: check the host direct-read validation, artifact identity and checkpoint pairing. CPU fallback is disabled.
- No HTTP 200: inspect the startup log and `error.txt`; a submitted start is not a ready service.
- `memory-stop.txt`: host available RAM reached the guard threshold; check other workloads before restarting.
- Wrong model revision: use the pinned checkpoint and do not edit its original config/index.
- Existing output: retain verified data, or choose new output paths and update the matching identity.

## Upgrading an existing deployment

Stop the old model before loading another. v0.2.0 changes the supported checkpoint to NVIDIA: **a v0.1.7 RadixArk deployment must complete steps 1–3** and update the model/PLE/identity paths. Keep its old image and configuration for rollback. Already prepared, verified NVIDIA data can be reused.

```bash
git pull --ff-only
docker build -t qwen-flash-sm80:0.2.0 .
# Update config/local.json paths and image before starting.
python3 scripts/serve.py start --config config/local.json
```

Set `image` to `qwen-flash-sm80:0.2.0` and keep `draft_int8=true`. The old standalone `draft_int8=false` toggle does not apply to the new probability/history drafting path and is rejected early. Other startup parameters are stored in `config/vllm-args.json`; keep the documented single-request/synchronous constraints. [Validation scope](RELEASE_VALIDATION.en.md).
