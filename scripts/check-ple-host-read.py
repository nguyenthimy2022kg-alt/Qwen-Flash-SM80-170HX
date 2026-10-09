#!/usr/bin/env python3
"""Compare Q38_PLE_HOST_READ sync_mt reads byte for byte against the cuFileRead path of the same reader.

Run inside the image, on the PLE GPU, with the service stopped or with enough free GPU memory.
Exits 1 on the first mismatch.
"""
import argparse
import os
import random
import sys

os.environ["Q38_PLE_HOST_READ"] = "1"

import torch  # noqa: E402

from ple_gds.persistent_reader import PersistentGdsPleReader  # noqa: E402


def collect(reader, ids, mode, stream):
    parts = []
    for lease in reader.iter_read(ids, mode=mode, stream=stream):
        parts.append(lease.tensor.view(torch.uint8).clone())
        lease.release(stream)
    stream.synchronize()
    return torch.cat(parts)


def cases(row_count, rng):
    yield "first row", [0]
    yield "last row", [row_count - 1]
    yield "112 random rows", rng.sample(range(row_count), 112)
    yield "113 random rows", rng.sample(range(row_count), 113)
    yield "repeated rows", [rng.randrange(row_count)] * 64 + rng.sample(range(row_count), 64) * 2
    start = rng.randrange(row_count - 8172)
    yield "8172 contiguous rows", list(range(start, start + 8172))
    yield "8172 random rows", rng.sample(range(row_count), 8172)
    yield "40000 random rows", [rng.randrange(row_count) for _ in range(40000)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=os.environ.get("VLLM_PLE_GDS_ARTIFACT", "/ple"))
    parser.add_argument("--gpu-uuid", default=os.environ.get("VLLM_PLE_GDS_GPU_UUID"))
    parser.add_argument("--gpu-bdf", default=os.environ.get("VLLM_PLE_GDS_GPU_BDF"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("VLLM_PLE_GDS_WORKERS", "32")))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.gpu_uuid or not args.gpu_bdf:
        parser.error("--gpu-uuid and --gpu-bdf are required (or VLLM_PLE_GDS_GPU_UUID/_BDF)")

    reader = PersistentGdsPleReader(
        args.artifact,
        expected_gpu_uuid=args.gpu_uuid,
        expected_gpu_bdf=args.gpu_bdf,
        worker_count=args.workers,
        pin_workers=True,
        poll_mode=True,
    )
    if not reader.stats()["host_read"]:
        sys.exit("reader did not enable host read")
    stream = torch.cuda.Stream(device=reader.device)
    rng = random.Random(args.seed)
    rows = 0
    try:
        for round_index in range(args.rounds):
            for name, ids in cases(reader.row_count, rng):
                expected = collect(reader, ids, "sync", stream)
                actual = collect(reader, ids, "sync_mt", stream)
                if not torch.equal(expected, actual):
                    sys.exit(f"MISMATCH round {round_index}: {name}")
                rows += len(ids)
            # Two leases held at once: the second read refills the host lanes while the first copy may be queued.
            first_ids = rng.sample(range(reader.row_count), 2048)
            second_ids = rng.sample(range(reader.row_count), 2048)
            first_expected = collect(reader, first_ids, "sync", stream)
            second_expected = collect(reader, second_ids, "sync", stream)
            first = reader.read(first_ids, mode="sync_mt", stream=stream)
            second = reader.read(second_ids, mode="sync_mt", stream=stream)
            stream.synchronize()
            overlap_ok = (torch.equal(first.tensor.view(torch.uint8), first_expected)
                          and torch.equal(second.tensor.view(torch.uint8), second_expected))
            first.release(stream)
            second.release(stream)
            if not overlap_ok:
                sys.exit(f"MISMATCH round {round_index}: back-to-back reads")
            rows += len(first_ids) + len(second_ids)
    finally:
        reader.close()
    print(f"OK: {args.rounds} rounds, {rows} rows identical between sync (cuFileRead) and sync_mt (host read)")


if __name__ == "__main__":
    main()
