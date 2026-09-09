# Qwen-Flash-SM80 / CMP 170HX

[简体中文](README.md) | **English**

**A community vLLM runtime for Qwen3.8-Flash-Next on two SM80 GPUs.**

Combines **GPUDirect Storage (GDS)**, **TEP2**, **MTP6**, and kernels tailored to specific tensor shapes to run the NVFP4 main model on two CMP 170HX GPUs while reading FP8 PLE data directly from SSD.

Completed single-request long-output tests with **8K–128K input: 151.05–168.29 tok/s decode throughput**, all ending naturally. A separate short-prompt run before packaging recorded **170.826 tok/s**. The task instruction was **“写个网站网页”** (“Build a web page”) in all cases. **With shorter reasoning, output speed can reach approximately 200 tok/s.**

This runtime targets a specific model, software stack, and two-GPU configuration. Validated hardware: two CMP 170HX GPUs (SM80, approximately 63.39 GiB VRAM each). Other SM80 devices and models have not been validated for compatibility or performance.

## Project highlights

- **Direct SSD-to-GPU PLE reads:** GDS/cuFile reads the required lookup rows without staging the data payload in CPU memory, overlapping reads with model computation. Upstream already supports PLE offload to host RAM; this project adds an SSD-to-GPU path to the pinned upstream version.
- **Avoids approximately 47.68 GiB of full-table VRAM storage:** the **51.2 GB** FP8 PLE table stays on SSD, without keeping the entire table resident in VRAM or host RAM. GPUs retain the required data and working buffers. This figure is for one complete table, not a per-GPU saving.
- **From approximately 38 to 170.826 tok/s:** **4.50× the initial deployment's throughput, or roughly a 350% increase**. Code-heavy outputs have approached **200 tok/s**. These are results from different stages of the project, not a controlled comparison against stock vLLM.

## Integrated optimizations

| Optimization | Implementation | Recorded result |
|---|---|---|
| GDS + PLE integration | Direct cuFile reads into VRAM, with input preparation and buffer lifecycle integration | Data and lifecycle validation passed; no isolated whole-model speedup measured |
| C++ read planner | Moves read planning out of Python to reduce object creation | Historical 4,096-row planning: 8.655 → 1.188 ms; local operation only |
| Persistent read threads and fixed buffers | Reuses threads, handles, and workspaces | 32 read threads and 3 buffer slots |
| In-process asynchronous reads and handoff fix | Overlaps I/O with compute and prevents premature input reuse | Historical combined version: 51.33 → 66.74 tok/s; not an isolated gain |
| TEP2 and P2P communication | Tensor + expert parallelism with direct GPU communication | Earlier MTP6 five-run aggregates: TEP2 131.044, TP2 127.820 tok/s |
| HC single-row GEMV | Specialized matrix-vector kernels for single-row state mixing | TP2 comparison at that stage: 74.55 → 80.02 tok/s |
| TileLang input projection | Fuses single-row QKVZ/BA projections and reuses output buffers | Integrated; no isolated whole-model gain measured |
| CUDA Graph and fused operators | Reuses execution graphs to reduce kernel submission overhead | Retains overall graph execution and the existing MTP GDN fused path |
| MTP6 + seven-row PLE planning | NumPy fast path for row planning during multi-token verification | Earlier five-run aggregate: 131.044 tok/s; later single run: 158.013 tok/s |
| Full-vocabulary INT8 draft scoring + BF16 reranking | Scores the full vocabulary in INT8, reranks each GPU's top 32 candidates using BF16 weights, then combines results | Single run: 170.826 tok/s; historical predecessor: 158.013, an observed difference of about 8.1% |
| Host-memory protection during loading | Limits container memory and stops the managed service when available host memory falls below a threshold | Includes loading fixes and memory protection; not counted as a decode gain |

Results come from different stages and conditions and **must not be added or multiplied**. The INT8 runs generated different content; the 8.1% difference does not establish an isolated quantization benefit. [Detailed historical records (Chinese)](docs/性能记录.md).

## 8K–128K benchmark

Task prompt: **“写个网站网页”** (“Build a web page”), preceded by web-design reference text to reach each input length. **Thinking enabled, using the template's default `xhigh` (highest supported level), without a separate thinking budget.** TEP2 + MTP6; output limit: 50,000 tokens. One run per length, all ending naturally. Both reasoning and answer tokens count toward output.

| Input length | Prefill throughput* (tok/s) | Decode throughput (tok/s) |
|---|---:|---:|
| 8K | 1986.38 | **168.29** |
| 16K | 2083.10 | **152.99** |
| 32K | 2068.85 | **156.44** |
| 64K | 2036.21 | **159.88** |
| 128K | 2059.93 | **151.05** |

**Outputs with less reasoning have approached 200 tok/s in testing (199 tok/s displayed by the client).**

*Prefill is a client-side approximation: input tokens divided by time to first nonempty output, including input processing and first-output generation. Decode is output tokens divided by the interval from the first to the last nonempty streamed output. It includes reasoning and answer text; empty events and the final completion notification do not extend this interval.

Actual input lengths were 8,193 / 16,385 / 32,769 / 65,537 / 131,073 tokens, including the chat template. Sampling used service defaults: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5; no fixed seed. Each request used a unique cache salt. [Machine-readable results](docs/context-benchmark-20260908.json) · [Full prompt construction and timing details (Chinese)](docs/性能记录.md).

## Quick start

For a new deployment, follow the **[English deployment guide](docs/DEPLOYMENT.en.md)** to download the pinned checkpoint, validate its files, and prepare the PLE data. The public checkpoint already includes FP8 PLE and BF16 MTP weights.

```bash
git clone https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX.git
cd Qwen-Flash-SM80-170HX
```

Requires Linux, Docker, NVIDIA Container Toolkit, a CUDA 13-compatible driver, working GDS/cuFile, and GPU P2P. The GPU memory capacity and SSD layout must meet this model's requirements. With `allow_compat_mode=false`, GDS failures do not silently fall back to host-memory staging.

**Validate capabilities on the target host.** The reference driver's patches and IOMMU/filesystem settings are examples, not mandatory settings to copy. The required outcomes are working two-GPU P2P and validated strict cuFile direct reads, alongside the specified hardware, model format, and software stack. CMP reference documents: [P2P setup (Chinese)](docs/CMP_P2P.md) · [NVMe GDS reproduction guide (Chinese)](docs/GDS_NVME_P2PDMA_REPRODUCTION.md).

Once the model and PLE data are prepared:

```bash
docker build -t qwen-flash-sm80:0.1.3 .
cp config/example.json config/local.json
```

Edit `config/local.json` to set model paths, the PLE artifact, and GPU IDs. Then validate the generated command and start:

```bash
python3 scripts/serve.py start --config config/local.json --dry-run
python3 scripts/serve.py start --config config/local.json
```

The service binds to localhost only, at `127.0.0.1:18420`, with model name `qwen3.8-flash-next`. The first startup includes compilation and warmup. Wait for `/health` to return HTTP 200 before sending requests.

| Client setting | Value |
|---|---|
| OpenAI-compatible base URL | `http://127.0.0.1:18420/v1` |
| Full chat-completions URL | `http://127.0.0.1:18420/v1/chat/completions` |
| Model | `qwen3.8-flash-next` |
| API key | Leave empty; use a placeholder if the client requires one |

To stop, use the container name printed by the launcher:

```bash
python3 scripts/serve.py stop --name "<container-name>"
```

[Deployment and fallback settings](docs/DEPLOYMENT.en.md) · [Release validation record (Chinese)](docs/发布检查.md).

## Repository layout

- `src/`: runtime source overlay for the pinned upstream version, including GDS, HC, TileLang, and draft INT8.
- `csrc/`: C++ source for the GDS reader extensions, compiled during the image build.
- `src/preload/`: approximately 17 MB of preloaded Triton kernels and SHA256 manifests for this fixed SM80 environment; no model weights.
- `patches/`: upstream source hashes and provenance information to prevent overwriting incompatible versions.
- `config/`, `scripts/`: configuration examples, service control, data preparation, and build tools. CLI messages currently remain in Chinese.
- `docs/`: performance, deployment, and validation records. An English deployment guide is available; detailed historical and driver-reference documents remain in Chinese.

## Acknowledgments and license

Thanks to vLLM, Qwen, the original Qwen3.8-Flash-DGX community project, NVIDIA GDS, Marlin, Triton, TileLang, and their contributors. Existing source notices are preserved. Project code is licensed under Apache-2.0; model weights, CUDA/cuFile, containers, and third-party dependencies retain their respective licenses. See [LICENSE](LICENSE) and [source provenance and licensing details (Chinese)](docs/来源与许可.md).
