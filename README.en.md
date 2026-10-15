# Qwen-Flash-SM80 / CMP 170HX

[简体中文](README.md) | **English**

**A community vLLM runtime for Qwen3.8-Flash-Next on two SM80 GPUs.**

Uses **GDS direct SSD reads**, **TEP2 tensor and expert parallelism**, **MTP6 speculative decoding**, and specialized kernels to run the NVFP4 main model on two CMP 170HX GPUs. The approximately 51.2 GB PLE lookup table stays on SSD, with rows read into VRAM as needed.

**v0.3.0 supports 1–10 concurrent requests; the example configuration defaults to 10.** It uses the NVIDIA NVFP4 checkpoint and fixed MTP6, with per-request draft isolation, compact vocabulary transport, and expanded PCIe communication. History-copy drafting has been removed.

**Measured on two CMP 170HX GPUs: 749.67 tok/s aggregate at 10 concurrent requests, averaging 76.29 tok/s per request.**

[Deployment](docs/DEPLOYMENT.en.md) · [10-request benchmark](#10-request-benchmark) · [API connection](#api-connection) · [Release validation](docs/MULTI_REQUEST.en.md)

Validated hardware: two CMP 170HX GPUs (SM80, approximately 64 GiB VRAM each). GPU P2P and NVMe GDS are host prerequisites. Existing NVIDIA model and matching PLE data can be reused; rebuild the image to upgrade.

## Project highlights

- **Direct SSD-to-GPU PLE reads:** GDS/cuFile reads the required lookup rows without staging the data payload in CPU memory, overlapping reads with model computation. Upstream already supports PLE offload to host RAM; this project adds an SSD-to-GPU path to the pinned upstream version.
- **Avoids approximately 47.68 GiB of full-table VRAM storage:** the **51.2 GB** FP8 PLE table stays on SSD, without keeping the entire table resident in VRAM or host RAM. GPUs retain the required data and working buffers. This figure is for one complete table, not a per-GPU saving.
- **From approximately 38 to 170.826 tok/s:** **4.50× the initial deployment's throughput, or roughly a 350% increase**. Code-heavy outputs have approached **200 tok/s**. These are results from different stages of the project, not a controlled comparison against stock vLLM.

## 10-request benchmark

**October 11, 2026: NVIDIA NVFP4, two CMP 170HX GPUs, TEP2 + MTP6.** Prompt: “写个网站网页” (“Build a web page”). Thinking disabled; `temperature=1`, `top_p=0.95`, `top_k=20`. Each of 10 requests generates 5,000 tokens.

| Metric | Result |
|---|---:|
| Aggregate output throughput | **749.67 tok/s** |
| Mean output speed per request | **76.29 tok/s** |
| Per-request range | 75.19–78.30 tok/s |
| Total output | 50,000 tokens |
| Generation cycle | 72.34 ms |

Output speed is measured from the first to the last nonempty streamed output, excluding time to first token. Aggregate throughput increased from **623.41 to 749.67 tok/s**, approximately **20.3%**, over this optimization series. [Changes and test records](docs/MULTI_REQUEST.en.md)

<details>
<summary>Historical single-request results: 8K–128K input and code output near 200 tok/s</summary>

## Historical 8K–128K benchmark

Historical results from September 8, 2026, using the RadixArk checkpoint. Task prompt: **“写个网站网页”** (“Build a web page”), preceded by web-design reference text to reach each input length. **Thinking enabled, using the template's default `xhigh` (highest supported level), without a separate thinking budget.** TEP2 + MTP6; output limit: 50,000 tokens. One run per length, all ending naturally. Both reasoning and answer tokens count toward output.

| Input length | Prefill throughput* (tok/s) | Decode throughput (tok/s) |
|---|---:|---:|
| 8K | 1986.38 | **168.29** |
| 16K | 2083.10 | **152.99** |
| 32K | 2068.85 | **156.44** |
| 64K | 2036.21 | **159.88** |
| 128K | 2059.93 | **151.05** |

**Outputs with less reasoning have approached 200 tok/s in testing (199 tok/s displayed by the client).**

*Prefill is a client-side approximation: input tokens divided by time to first nonempty output, including input processing and first-output generation. Decode is output tokens divided by the interval from the first to the last nonempty streamed output. It includes reasoning and answer text; empty events and the final completion notification do not extend this interval.

[Full prompt construction, sampling and timing details (Chinese)](docs/性能记录.md).

**By some people's misleading testing standards, it can reach 260 tokens/s.**

![Sequential-number output test: client reports 259 tokens/s and 240 ms time to first token](docs/images/sequential-numbers-259-tok-s.png)

</details>

## Quick start

Follow the [deployment guide](docs/DEPLOYMENT.en.md): prepare the host → download and convert model data → build and start the service. Model deployment commands and settings are collected on that page. Model weights and prebuilt images are not included.

For CMP drivers, use the original community **bayley/cmpunlocker** project, which includes the BAR1/P2P patches. No additional patch bundle from this repository is needed. Hosts with working P2P/GDS can proceed directly to model preparation.

## API connection

| Client setting | Value |
|---|---|
| OpenAI-compatible base URL | `http://127.0.0.1:18420/v1` |
| Full chat-completions URL | `http://127.0.0.1:18420/v1/chat/completions` |
| Model | `qwen3.8-flash-next` |
| API key | Leave empty; use a placeholder if the client requires one |

Use the base URL or full endpoint as required by the client. The default service is accessible only from the host: `127.0.0.1` on a phone or another computer points to that device itself. See the [deployment guide](docs/DEPLOYMENT.en.md) for remote access, request examples, and troubleshooting.

## Integrated optimizations

| Area | Current implementation |
|---|---|
| SSD → GPU | GDS/cuFile PLE reads, C++ planning, 32 persistent readers, 3 reusable staging slots and ordered asynchronous handoff |
| Parallel execution | TEP2 (tensor parallelism + expert parallelism), Marlin and GPU P2P |
| Draft generation | Fixed MTP6 with independent draft probabilities for each request and target-model verification |
| Draft scoring | Full-vocabulary INT8 screening, BF16 candidate reranking and an 8-candidate probability distribution used consistently by drafting and verification |
| Target scoring | Best-token transport at temperature 0; candidate and tie transport at temperature 1 with top-k 20; full-vocabulary paths for other settings |
| Small compute shapes | HC GEMV, TileLang single-row projections, fused seven-row HC/GDN computation and CUDA Graph replay |
| Communication | FlashInfer PCIe IPC AllReduce covering draft and target-verification shapes for 1–10 requests |
| Multi-turn cache | Prefix matching in 36-token units; physical cache blocks remain 3,456 tokens |
| Host memory | Selective CUDA loading, targeted kernel preloading, streaming checkpoint preparation and memory limits |
| Stability | PLE handoff synchronization, context-boundary completion fixes and prewarmed sampling fallbacks |

The configuration supports **two SM80 GPUs, 1–10 active requests, synchronous scheduling**, with 10 requests in the example configuration. Requests share the VRAM cache; 262K is the context limit per request. Maximum combined context is 262,144 tokens; the configured output limit is also 262,144 and remains constrained by available context. Image input is configurable up to 128 images, but this is a limit rather than a claim that every 128-image request fits. Thinking, preserved reasoning history and tool parsing remain enabled by default.

## Repository layout

- `src/`: runtime source overlay for the pinned upstream version, including GDS, HC, TileLang, and draft INT8.
- `csrc/`: C++ source for the GDS reader extensions, compiled during the image build.
- `src/preload/`, `src/memory_extra_preload/`: approximately 24 MB of preloaded Triton kernels and SHA256 manifests for this fixed SM80 environment; no model weights.
- `patches/`: upstream source hashes and provenance information to prevent overwriting incompatible versions.
- `config/`, `scripts/`: configuration examples, service control, data preparation, and build tools. CLI messages currently remain in Chinese.
- `docs/`: performance, deployment, and validation records. English deployment, CMP P2P, and NVMe GDS guides are available; detailed historical records remain in Chinese.

## Acknowledgments and license

Thanks to vLLM, Qwen, the original Qwen3.8-Flash-DGX community project, NVIDIA GDS, the cmpunlocker community, Marlin, Triton, TileLang, and their contributors. Existing source notices are preserved. Inference project code is licensed under Apache-2.0; model weights, CUDA/cuFile, containers, and third-party dependencies retain their respective licenses. See [LICENSE](LICENSE) and [source provenance and licensing details (Chinese)](docs/来源与许可.md).
