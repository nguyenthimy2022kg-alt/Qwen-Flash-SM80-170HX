# v0.3.0 multi-request optimizations

[简体中文](MULTI_REQUEST.md) | **English** · [Home](../README.en.md)

Supports 1–10 active requests with fixed MTP6. GDS, TEP2, prefix caching, tool calls, and image inputs are retained. History-copy drafting has been removed.

## Configuration

Build `qwen-flash-sm80:0.3.0` using the [deployment guide](DEPLOYMENT.en.md), then use these values in your complete `config/local.json`:

```json
{
  "image": "qwen-flash-sm80:0.3.0",
  "max_num_seqs": 10,
  "gpu_memory_utilization": 0.93
}
```

`max_num_seqs` accepts 1–10 and defaults to 1 when omitted; the repository example sets it to 10. Requests share the VRAM cache: the 262,144 context limit does not allocate that capacity to all ten requests simultaneously. Keep synchronous scheduling and `draft_int8=true`.

## Changes and performance

October 11, 2026 PDT: two CMP 170HX GPUs, NVIDIA NVFP4, TEP2, MTP6. Prompt: “写个网站网页” (“Build a web page”). Thinking disabled; temperature=1, top-k=20, top-p=0.95; ten requests of 5,000 output tokens each.

| Stage | Mean tok/s per request | Aggregate tok/s | Generation cycle |
|---|---:|---:|---:|
| Before this optimization series | 63.43 | 623.41 | 87.73 ms |
| Compact vocabulary transport | 71.54 | 700.98 | 76.88 ms |
| Multi-request PCIe transport (this release) | **76.29** | **749.67** | **72.34 ms** |

- **Per-request draft isolation:** each request owns its candidates and probabilities, with request identity preserved through completion, cancellation, and replacement.
- **Vocabulary transport:** exchange the best token at temperature 0, or candidates and boundary ties at temperature 1 with top-k=20. The latter reduces each GPU's 70-row payload from approximately 16.58 MiB to 1.05 MiB while retaining the ordinary global filtering and verification steps. Other settings use the full-vocabulary path.
- **PCIe transport:** extend the existing shared-VRAM communication path to shapes used by 1–10 requests. Local comparisons measured 265.45→227.41 µs for 70 rows and 45.54→34.83 µs for 10 rows, adding approximately 1.63 MiB of workspace per GPU.

Output speed uses the first-to-last nonempty output interval, excluding time to first token. `ignore_eos=true` holds output length fixed. Vocabulary transport was compared within one model load; PCIe results came from the immediately following load. Each setting has one screening run with the same phase instrumentation. Times from the two GPUs are not added together.

## Completed checks

| Scope | Result |
|---|---|
| 1–10 request positions, draft probabilities, and lifecycle | Local and model-request checks passed |
| Temperature-0 vocabulary path | Dual-GPU local checks and 7,140 real forward rows per rank passed best-token comparison |
| Temperature-1 vocabulary path | 45 local cases and 8,318 real forward rows per rank passed score, sampling, and acceptance checks |
| PCIe path | 20 shapes, 5 fallback cases, strided input, independent outputs, and 150 interleaved replays passed |
| Final PCIe model runs | 50 complete text/JSON requests, 2 intentional cancellations, and 13 exact marker/JSON checks passed |
| Changing concurrency | Active requests covered 10→1→0; independent follow-ups, cancellation/refill, and structured recovery completed |
| Runtime files | 595 files in the clean build matched the manifest; diagnostic modules were removed |

One additional response in the final PCIe run returned HTTP 200 and a normal stream ending with `finish_reason=tool_calls`, which did not match the initial text-only harness assertion. That record was retained, and remaining checks were completed. It was not counted as a plain-text speed sample.

Validation covers these short text and request-lifecycle checks. Ten full-context requests, concurrent multimodal requests, and long-duration stress have not been rerun on this release. Publication work on October 14, 2026 used CPU checks and file verification only, without loading a model or accessing a GPU.

[Machine-readable results](release-validation-0.3.0.json) · [Historical v0.2.0 validation](RELEASE_VALIDATION.en.md)
