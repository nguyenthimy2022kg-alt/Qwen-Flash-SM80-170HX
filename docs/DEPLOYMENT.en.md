# Deployment guide

[简体中文](从零部署.md) | **English** · [Home](../README.en.md)

This guide covers model download, PLE conversion, image building, and service access on a prepared Linux host with two SM80 GPUs. Prepare the driver, Docker, GPU P2P, and GDS first. Except for cloning the repository and forwarding SSH, run commands from the repository root on the service host.

[Requirements](#requirements) · [Download and check](#1-download-source-and-model) · [Convert PLE](#3-convert-and-enroll-ple-data) · [Build and configure](#4-build-and-configure) · [Start and connect](#5-start-and-connect) · [Troubleshoot](#stop-and-troubleshoot)

## Requirements

- Linux x86_64, Git, curl, Python 3.10 or newer with `venv` and pip, Docker, NVIDIA Container Toolkit, and a CUDA 13-compatible driver. The repository's data-preparation scripts use only the Python standard library; the download tool is installed in its own virtual environment.
- Validated hardware: two CMP 170HX GPUs, approximately 63.39 GiB VRAM each. Two 40 GB cards are not an equivalent configuration. Other GPUs have not undergone whole-model validation.
- The reference host has approximately 32 GiB RAM and 8 GiB swap. Default container limits are 21 GiB RAM and 26 GiB RAM plus swap. Leave memory available for the host and other applications.
- Model files total approximately 135.2 GB. Converted PLE data adds approximately 51.2 GB. Reserve at least 220 GB for data, plus separate space for Docker layers and compilation caches.
- Working GPU P2P and strict GDS/cuFile reads on the target NVMe/filesystem/driver/topology. Direct-read data validation must pass; installing GDS alone does not establish that the path works. The supplied configuration disables host-staging fallback with `allow_compat_mode=false`.

The [CMP P2P reference](CMP_P2P.en.md) and [GDS reproduction guide](GDS_NVME_P2PDMA_REPRODUCTION.en.md) describe the reference machine's configuration and validation. They are not standalone installers for a fresh host, and optional patches are now provided separately in the [CMP BAR1/P2P build guide](../drivers/cmp-bar1/README.en.md). Start with its read-only inventory and choose adaptations for your hardware. For host setup, also see the [NVIDIA GDS documentation](https://docs.nvidia.com/gpudirect-storage/) and [Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

See [reference hardware and PCIe topology](REFERENCE_HARDWARE.en.md): one EPYC 7532, two CMP 170HX GPUs, and a 990 PRO 4TB, with no PCIe switch on their paths.

## Host acceptance criteria

| Check | Required outcome |
|---|---|
| GPU and capacity | Two SM80 GPUs with sufficient VRAM for the model, MTP, cache, and workspaces; enough host RAM and disk space for loading |
| Software and model | The pinned software stack, model revision, and PLE layout; model-file checks pass |
| GPU P2P | Required peer-read/write capability and a successful peer-copy data check; a capability matrix alone is insufficient |
| Direct SSD reads | Correct data reaches the GPU through this project's cuFile path with compatibility fallback disabled; data checks and logs establish that the payload is not staged through host RAM |
| Model service | Loading and warmup complete, health checks pass, and an actual generation request succeeds |

Driver patches, BAR1, IOMMU, NVMe multipath, and filesystem settings depend on the target hardware and driver. A host that meets these conditions can proceed without cmpunlocker or the reference machine's local overlay. The default cuFile configuration uses NVMe P2PDMA; other supported GDS paths need their own configuration and the same strict direct-read validation.

These are functional criteria, not a throughput guarantee or proof that other GPUs have been tested. Measure performance on the target hardware; whole-model validation currently covers only the documented reference machine.

## 1. Download source and model

The examples place data under `$HOME/qwen-flash-data`. Its `ple-gds` directory must reside on an NVMe filesystem validated for GDS. If your home directory is on an unsuitable disk, choose a data location first and replace the paths below, including `root` in step 4's configuration script. If you already cloned the repository from the README, skip the first two commands.

```bash
git clone https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX.git
cd Qwen-Flash-SM80-170HX
python3 -m venv .venv-hf
.venv-hf/bin/pip install 'huggingface_hub==1.29.0'
mkdir -p "$HOME/qwen-flash-data/models"
```

Checkpoint: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594), pinned revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`, also recorded in [`config/model-source.json`](../config/model-source.json). It includes NVFP4 main-model weights, 31 BF16 MTP tensors, 128 FP8 PLE shards, and a BF16 scale. No additional PLE quantization or separate MTP checkpoint is required. The checkpoint retains its source license.

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

Rerun the download command after a network interruption. Keep the complete checkpoint, including its original FP8 PLE shards, for model-file checks and future conversion. This path does not require historical `.plefp8.bak` files or the legacy checkpoint-view script.

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

Conversion streams the table into its GDS layout without requiring the whole 51.2 GB table in RAM. Enrollment verifies the full converted data by default and records its identity in `config/local-ple-identity.json`; retain that check for a new deployment. Mapping and enrollment refuse to overwrite existing output files. Once valid data is prepared, reuse it instead of reconverting on every startup.

## 4. Build and configure

```bash
docker build -t qwen-flash-sm80:0.1.4 .
```

Build the image locally; no prebuilt project image is published. Run the following configuration initialization only for a first deployment. If `config/local.json` already exists, skip this block and edit its local paths and image tag directly.

```bash
cp -n config/example.json config/local.json
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

Review `gpu_ids` in `config/local.json`. GPU indices default to `0` and `1`; full UUIDs are also accepted. The first GPU owns PLE reads, so choose the order with SSD/GPU topology in mind. Model, PLE, and identity paths must refer to the files prepared above. Paths in the configuration resolve relative to the repository root. Model symlinks must remain accessible through the `/models` mount; mount their common parent directory when necessary.

| Setting | Purpose |
|---|---|
| `image` | Locally built Docker image tag; must match the build command |
| `models_root`, `model_subdir` | Parent model directory and model subdirectory |
| `ple_artifact` | Converted PLE directory containing `CURRENT` |
| `ple_identity` | JSON file generated by enrollment |
| `cufile_config` | cuFile configuration, default `config/cufile.json` |
| `gpu_ids` | Two distinct GPU indices or full UUIDs, in rank order; the first GPU reads PLE |
| `mode` | `tep2` by default; `tp2` is also available |
| `draft_int8` | Enables full-vocabulary INT8 draft scoring with BF16 reranking |
| `port` | Default 18420, localhost only |
| `container_memory_gib` | Host RAM limit, default 21 GiB; not a VRAM limit |
| `container_memory_and_swap_gib` | RAM plus swap limit, default 26 GiB |
| `min_host_available_gib` | Stops the managed service below this available-host-memory threshold; default 2 GiB |
| `core_offset_guard` | Optional read-only GPU core-offset check; disabled by default, never changes hardware |
| `validation_dir` | Optional historical draft-validation samples; disabled by default |

Defaults enable TEP2, MTP6, and INT8 draft scoring with BF16 reranking. Keep the original BF16 draft weights; fallback settings are described below.

## 5. Start and connect

```bash
python3 scripts/serve.py start --config config/local.json --dry-run
python3 scripts/serve.py start --config config/local.json
```

The dry run validates configuration and reads GPU identifiers before printing the launch command. It does not load the model or check that data files exist. A normal start also checks required paths, file/directory types, Docker access, the local image, and container-name conflicts before submitting startup. It does not replace step 2's model-file checks or establish that P2P/GDS works.

The launcher prints a container name and log directory under `runs/`. The first load includes compilation and warmup. Follow progress with `docker logs -f "<container-name>"`. Submission does not mean the service is ready; wait for HTTP 200 from both endpoints before sending a request:

```bash
curl --fail http://127.0.0.1:18420/health
curl --fail http://127.0.0.1:18420/v1/models
```

| Client setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:18420/v1` |
| Full chat-completions URL | `http://127.0.0.1:18420/v1/chat/completions` |
| Model | `qwen3.8-flash-next` |
| API key | Authentication is not configured; use `local` if the client requires a value |

Example streaming request:

```bash
curl --no-buffer --fail-with-body http://127.0.0.1:18420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"写个网站网页"}],"max_tokens":50000,"stream":true}'
```

The example prompt means “Build a web page.” Thinking is enabled with the template default `xhigh`; the client must handle both reasoning and answer content. `max_tokens` limits their combined output and is a ceiling, not a required length. Verify that content streams and the request finishes successfully to complete the generation check. Benchmark sampling and timing details are in the [performance record (Chinese)](性能记录.md).

These URLs use the default port 18420. If you change `port`, update client URLs and forwarding ports accordingly. The service binds to localhost. For access from another machine, run an SSH tunnel on the client machine:

```bash
ssh -N -L 18420:127.0.0.1:18420 username@service-host
```

Then use the same localhost base URL on the client machine. For clients running inside a container, `127.0.0.1` refers to that container; configure a network path to the service host.

## Stop and troubleshoot

```bash
python3 scripts/serve.py stop --name "<container-name>"
```

The stop command operates only on containers labeled as belonging to this project. To change settings or fall back, stop the old service, edit the configuration, and start again; a new container name is generated by default. `draft_int8=false` restores the original BF16 full-vocabulary draft scoring while retaining MTP6 and other optimizations. `mode=tp2` disables expert parallelism. The tools do not modify drivers, clocks, fans, or power limits.

| Symptom | Check |
|---|---|
| `POST /v1` returns 404 | The client may require the full `/v1/chat/completions` URL |
| Missing model or PLE files | Paths, download completeness, conversion, and enrollment |
| Mapping or identity output already exists | Reuse validated data; for a new generation, choose new output filenames and update the corresponding configuration paths |
| Model revision mismatch | Use the pinned revision and unmodified config/index |
| cuFile registration or read failure | Host driver, filesystem, topology, and strict GDS validation |
| `memory-stop.txt` appears | Available host memory fell below the configured threshold |
| Image not found | Build the image and check that its tag matches `config/local.json` |
| Docker is inaccessible | Check that Docker is running and the current user has permission to access it |
| Health check fails or no container appears | Inspect `runs/<container-name>/supervisor.log`, `error.txt`, and `docker logs` |

The configured maximum context is 262,144 tokens with up to eight request slots; those are configuration limits, not validated full-length/concurrency claims. The published context benchmark covers one request at a time, from 8K to 128K input. It is not a full output-quality or generated-website functionality evaluation. A fresh whole-model deployment on a second machine has not yet been recorded.

[Performance overview](../README.en.md) · [Detailed release validation (Chinese)](发布检查.md) · [Source provenance and licensing (Chinese)](来源与许可.md)
