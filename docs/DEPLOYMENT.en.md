# Deployment guide

[简体中文](从零部署.md) | **English** · [Home](../README.en.md)

## Requirements

- Linux x86_64, Python 3.10 or newer, Docker, NVIDIA Container Toolkit, and a CUDA 13-compatible driver.
- Validated hardware: two CMP 170HX GPUs, approximately 63.39 GiB VRAM each. Two 40 GB cards are not an equivalent configuration. Other GPUs have not undergone whole-model validation.
- The reference host has approximately 32 GiB RAM and 8 GiB swap. Default container limits are 21 GiB RAM and 26 GiB RAM plus swap. Leave memory available for the host and other applications.
- Model files total approximately 135.2 GB. Converted PLE data adds approximately 51.2 GB. Reserve at least 220 GB for data, plus separate space for Docker layers and compilation caches.
- Working GPU P2P and strict GDS/cuFile reads on the target NVMe/filesystem/driver/topology. Direct-read data validation must pass; installing GDS alone does not establish that the path works. The supplied configuration disables host-staging fallback with `allow_compat_mode=false`.

Reference driver and filesystem settings are examples, not a requirement to reproduce the same machine. The default cuFile configuration uses NVMe P2PDMA. Other GDS paths require their own configuration and direct-read validation. See the [CMP P2P reference](CMP_P2P.en.md), [GDS reproduction guide](GDS_NVME_P2PDMA_REPRODUCTION.en.md), and [NVIDIA GDS documentation](https://docs.nvidia.com/gpudirect-storage/).

## 1. Download the pinned model

Run preparation commands from the repository root. The examples place data under `$HOME/qwen-flash-data`; choose a location on a suitable filesystem with sufficient free space.

```bash
git clone https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX.git
cd Qwen-Flash-SM80-170HX
python3 -m venv .venv-hf
.venv-hf/bin/pip install 'huggingface_hub==1.29.0'
mkdir -p "$HOME/qwen-flash-data/models"
```

Checkpoint: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594), pinned revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`. It already includes NVFP4 main-model weights, 31 BF16 MTP tensors, 128 FP8 PLE shards, and a BF16 scale. No additional PLE quantization or separate MTP checkpoint is required.

Review the download plan, then download:

```bash
.venv-hf/bin/hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir "$HOME/qwen-flash-data/models/Qwen3.8-Flash-Next-NVFP4" \
  --dry-run

.venv-hf/bin/hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir "$HOME/qwen-flash-data/models/Qwen3.8-Flash-Next-NVFP4" \
  --max-workers 2
```

Rerun the download command after a network interruption. Keep the original FP8 PLE shards: conversion and startup checks still need them. This path does not require historical `.plefp8.bak` files or the legacy checkpoint-view script.

## 2. Check model files

```bash
python3 scripts/check-model.py \
  --model "$HOME/qwen-flash-data/models/Qwen3.8-Flash-Next-NVFP4"
```

This checks the pinned config/index hashes, 206 safetensors files, tensor headers and file sizes, PLE/MTP formats, and tokenizer file presence. It does not load the model or hash all weight payloads. `weight_payload_sha256_verified: false` explicitly reports that limitation.

For full downloaded-file verification:

```bash
.venv-hf/bin/hf cache verify RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir "$HOME/qwen-flash-data/models/Qwen3.8-Flash-Next-NVFP4" \
  --fail-on-missing-files
```

## 3. Convert and enroll PLE data

```bash
python3 scripts/prepare-ple.py mapping \
  --index "$HOME/qwen-flash-data/models/Qwen3.8-Flash-Next-NVFP4/model.safetensors.index.json" \
  --output "$HOME/qwen-flash-data/ple-mapping.json"

python3 scripts/convert-ple-to-gds-layout.py \
  --checkpoint-index "$HOME/qwen-flash-data/models/Qwen3.8-Flash-Next-NVFP4/model.safetensors.index.json" \
  --mapping "$HOME/qwen-flash-data/ple-mapping.json" \
  --output "$HOME/qwen-flash-data/ple-gds" --compact --block-rows 256

python3 scripts/prepare-ple.py enroll \
  --artifact "$HOME/qwen-flash-data/ple-gds" \
  --output config/local-ple-identity.json
```

Conversion streams the table into its GDS layout without requiring the whole 51.2 GB table in RAM. Enrollment verifies the full converted data by default; retain that check for a new deployment. Mapping and enrollment refuse to overwrite existing output files. Once valid data is prepared, reuse it instead of reconverting on every startup.

## 4. Build and configure

```bash
docker build -t qwen-flash-sm80:0.1.3 .
cp config/example.json config/local.json
python3 - <<'PY'
import json
from pathlib import Path
p = Path('config/local.json')
c = json.loads(p.read_text())
root = Path.home() / 'qwen-flash-data'
c['models_root'] = str(root / 'models')
c['model_subdir'] = 'Qwen3.8-Flash-Next-NVFP4'
c['ple_artifact'] = str(root / 'ple-gds')
p.write_text(json.dumps(c, ensure_ascii=False, indent=2) + '\n')
PY
```

Review `gpu_ids` in `config/local.json`. GPU indices default to `0` and `1`; full UUIDs are also accepted. The first GPU owns PLE reads, so choose the order with SSD/GPU topology in mind. Paths in the configuration resolve relative to the repository root. Model symlinks must remain accessible through the `/models` mount; mount their common parent directory when necessary.

| Setting | Purpose |
|---|---|
| `models_root`, `model_subdir` | Parent model directory and model subdirectory |
| `ple_artifact` | Converted PLE directory containing `CURRENT` |
| `ple_identity` | JSON file generated by enrollment |
| `cufile_config` | cuFile configuration, default `config/cufile.json` |
| `mode` | `tep2` by default; `tp2` is also available |
| `draft_int8` | Enables full-vocabulary INT8 draft scoring with BF16 reranking |
| `port` | Default 18420, localhost only |
| `container_memory_gib` | Host RAM limit, default 21 GiB; not a VRAM limit |
| `container_memory_and_swap_gib` | RAM plus swap limit, default 26 GiB |
| `min_host_available_gib` | Stops the managed service below this available-host-memory threshold; default 2 GiB |
| `core_offset_guard` | Optional read-only GPU core-offset check; disabled by default, never changes hardware |
| `validation_dir` | Optional historical draft-validation samples; disabled by default |

Defaults enable TEP2, MTP6, and INT8 draft scoring with BF16 reranking. Keep the original BF16 draft weights. To use the original draft scoring path, set `draft_int8=false` before startup; MTP6 and other optimizations remain enabled.

## 5. Start and connect

```bash
python3 scripts/serve.py start --config config/local.json --dry-run
python3 scripts/serve.py start --config config/local.json
```

The dry run prints the launch command and reads GPU identifiers; it does not load the model or check that data files exist. A normal start checks required paths before submitting the container startup.

The launcher prints a container name and log directory under `runs/`. The first load includes compilation and warmup. Follow progress with `docker logs -f "<container-name>"`. Wait for HTTP 200 from both endpoints:

```bash
curl --fail http://127.0.0.1:18420/health
curl --fail http://127.0.0.1:18420/v1/models
```

| Client setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:18420/v1` |
| Full chat-completions URL | `http://127.0.0.1:18420/v1/chat/completions` |
| Model | `qwen3.8-flash-next` |
| API key | Empty; a placeholder is acceptable if required by the client |

Example streaming request:

```bash
curl http://127.0.0.1:18420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"写个网站网页"}],"max_tokens":50000,"stream":true}'
```

The example prompt means “Build a web page.” Thinking is enabled with the template default `xhigh`. Output may include both reasoning and answer content. The output limit is a ceiling, not a required length.

## Stop and troubleshoot

```bash
python3 scripts/serve.py stop --name "<container-name>"
```

The stop command operates only on containers labeled as belonging to this project. To change settings or fall back, stop the old service, edit the configuration, and start again with a new container name. The tools do not modify drivers, clocks, fans, or power limits.

| Symptom | Check |
|---|---|
| `POST /v1` returns 404 | The client may require the full `/v1/chat/completions` URL |
| Missing model or PLE files | Paths, download completeness, conversion, and enrollment |
| Model revision mismatch | Use the pinned revision and unmodified config/index |
| cuFile registration or read failure | Host driver, filesystem, topology, and strict GDS validation |
| `memory-stop.txt` appears | Available host memory fell below the configured threshold |
| Container startup fails | Inspect `runs/<container-name>/supervisor.log`, `error.txt`, and `docker logs` |

The configured maximum context is 262,144 tokens with up to eight request slots; those are configuration limits, not validated full-length/concurrency claims. The published context benchmark covers one request at a time, from 8K to 128K input. It is not a full output-quality or generated-website functionality evaluation.

[Performance overview](../README.en.md) · [Detailed release validation (Chinese)](发布检查.md) · [Source provenance and licensing (Chinese)](来源与许可.md)
