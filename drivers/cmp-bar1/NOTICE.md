# Sources and license notices

This directory is an optional, separately licensed driver adaptation. It is not
part of the vLLM runtime and is not automatically installed by this project.

- Base driver: [NVIDIA/open-gpu-kernel-modules 610.43.03](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/610.43.03).
  Preserve [NVIDIA-COPYING](NVIDIA-COPYING) and all per-file NVIDIA notices.
- CMP base: [qg19932GH/cmpunlocker at aaddfd4](https://github.com/qg19932GH/cmpunlocker/tree/aaddfd4ce84a2804a7e0cd332acc4c26c79063d9),
  a fork in the cmpunlocker project family, licensed GPL v2. Its sources are
  downloaded with a pinned archive digest, not relabeled as this project's work.
- BAR1 overlays: extracted from the reference host's local cmpunlocker patch
  collection on 2026-09-08. A corresponding local checkout identifies
  [bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) as its origin and
  carries GPL v2. The patch commentary credits the amoghmunikote P2P branch;
  the exact originating commit for each local patch has not been established.
  Do not attribute these algorithms to the vLLM project maintainers.
- Original local patch digests, distributed patch digests, and fixed source
  archive digests are recorded in [sources.json](sources.json).
- `0011`: executable changes retained; stale internal patch references corrected.
- `0013`: executable changes retained; provenance header added.
- `0015`: rebased onto the pinned public base, without a debug-only prerequisite;
  corrected the obsolete Xeon/PLX explanation and log message. The CMP read-cap
  override is retained, but preparation now requires explicit opt-in.
- `prepare.py` and packaging changes dated 2026-09-08 are provided under
  GPL-2.0-only. Distributed CMP-derived patches are under GPL v2; underlying
  NVIDIA code retains its notices. See [LICENSE.GPL-2.0](LICENSE.GPL-2.0).

The root Apache-2.0 license does not replace these third-party terms. No kernel
module binaries or firmware are distributed here. Preparation enables P2P in the downloaded upstream CMP source through its build configuration.
