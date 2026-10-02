# v0.2.0 release validation

[简体中文](发布检查.md) | **English** · [Home](../README.en.md)

Validated on 2026-10-02, on the reference dual CMP 170HX host using the pinned NVIDIA checkpoint, TEP2, one active request and synchronous scheduling. The tested image was rebuilt from the public pinned base and repository sources. It did not mount the development runtime or experimental control directory.

| Check | Result |
|---|---|
| CPU tests | 33 passed, including checkpoint headers, data identity, copy failure cleanup, launch guards and binary hashes |
| NVIDIA loading view | Real checkpoint processed; 3,073 retained payload hashes match the accepted local view; the published PLE mapping also matches |
| GPU communication | Both ranks passed 7 shape/dtype cases, strided fallback, independent outputs and 21 interleaved graph replays |
| Functional/long-context suite | 28 requests passed, including one intentional cancellation followed by recovery |
| Continuous conversation | 20 additional turns in a single growing conversation passed |
| Long input and follow-ups | Initial inputs: 8,188 / 32,764 / 131,068 / 259,996 tokens; two follow-ups after each |
| Images, thinking, tools | Two-image recognition, thinking on/off, tool call/result return, JSON output and stochastic sampling passed |
| Completion and shutdown | 46 `stop`, 1 `tool_calls`, 1 intentional client cancellation; 9,883 reported output tokens; service exit 0, no OOM kill, no new Xid |
| Duplicate startup | Rejected without disturbing the running model |

The accepted local calculation code was retained. Packaging replaces the experimental control/kernel paths, embeds the active draft override and disables request tracing when tracing is off. Final cleanup removes private source-path metadata and trailing whitespace; 596 installed runtime files were checked against their hashes; Python ASTs and all kernel/native binary hashes match the GPU-tested image. The hashes are recorded in `patches/runtime-sha256.json`.

Host container peak memory was about 20.09 GiB and final memory about 16.66 GiB, plus about 2.70 GiB swap. These cgroup figures include file cache; they are neither model-only PSS nor GPU memory. No host-memory protection stop occurred.

These are bounded regression results on one machine, not an indefinite-uptime guarantee, a complete stock-vLLM quality comparison, or a 128-image capacity test. The near-limit input tests generated short verified answers, not 260K output. The history-copy probe passed; its throughput is not a benchmark of generation from scratch. Historical RadixArk speed results remain separate.

## Reproduce against a running service

```bash
python3 -m unittest discover -s tests -v
python3 tests/live_release.py --output runs/live-check --long-context
python3 tests/live_release.py --output runs/multiturn-check --soak-rounds 20
```

Use new output directories. `--base-url` and `--model` can target a custom local endpoint. Tests are serial; most turn thinking off and use temperature 0, while separate checks exercise thinking and sampling. GPU communication checks are in `tests/gpu_communication.py` and require two ranks, `Q38_PCIE_IPC=1` and a writable `/evidence` mount.

[Machine-readable summary](release-validation-0.2.0.json). The service was stopped after validation.
