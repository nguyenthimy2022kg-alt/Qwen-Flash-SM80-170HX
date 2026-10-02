"""Opt-in TP2 BF16 PCIe IPC path, scoped to locally screened shapes.

The IPC workspace is shared by all calls in one TP group. Eager operations and
CUDA graph replays must be serialized by vLLM's synchronous, single-lane runner.
Only graph-capture boundaries may rebind its stream, after device synchronization.
No output buffers are shared between calls.
"""
from contextlib import contextmanager
import importlib.util
import torch
import torch.distributed as dist
from flashinfer.comm.cuda_ipc import create_shared_buffer, free_shared_buffer

class PcieIpcComm:
    def __init__(self, group, device):
        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        assert dist.get_world_size(group) == 2
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        assert not getattr(config.parallel_config, 'use_ubatching', False), 'PCIe IPC requires one execution lane'
        assert not getattr(config.scheduler_config, 'async_scheduling', False), 'PCIe IPC requires synchronous scheduling'
        from q38_pcie_kernel import module as kernel
        self.module = kernel.load()
        self.bytes = self.module.pcie_ipc_workspace_size(2, 33 * 2560, 2, 32)
        with torch.cuda.device(self.device):
            self.ptrs = create_shared_buffer(self.bytes, group=group)
            self.handle = self.module.pcie_ipc_init(self.ptrs, self.rank, 33 * 2560, 2, 32)
            self.module.pcie_ipc_set_memop_enabled(self.handle, False)
            torch.cuda.synchronize(self.device)
        dist.barrier(group=group)
        self.stream = None
        self.depth = 0

    def supports(self, x):
        return (self.handle is not None and x.is_cuda and x.device == self.device
                and x.dtype == torch.bfloat16 and x.ndim == 2
                and x.shape[1] == 2560 and x.shape[0] in (1,7,33)
                and x.is_contiguous() and x.data_ptr() % 16 == 0)

    def all_reduce(self, x):
        capturing = torch.cuda.is_current_stream_capturing()
        current = torch.cuda.current_stream(self.device).cuda_stream
        if not capturing:
            if self.stream is None:
                self.stream = current
            elif self.stream != current:
                raise RuntimeError('PCIe IPC used on another stream without an ordered capture boundary')
        elif self.depth == 0:
            raise RuntimeError('PCIe IPC graph capture must use the TP capture context')
        out = torch.empty_like(x)
        self.module.pcie_ipc_all_reduce(self.handle, x, out, 16, 128, 0, False)
        return out

    @contextmanager
    def capture(self):
        if self.depth == 0:
            torch.cuda.synchronize(self.device)
            self.stream = None
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if self.depth == 0:
                torch.cuda.synchronize(self.device)
                self.stream = None

    def destroy(self):
        if self.handle is None:
            return
        torch.cuda.synchronize(self.device)
        # All ranks must reach coordinated teardown; IPC free is collective.
        self.module.pcie_ipc_dispose(self.handle)
        free_shared_buffer(self.ptrs, group=self.group)
        self.handle = None
