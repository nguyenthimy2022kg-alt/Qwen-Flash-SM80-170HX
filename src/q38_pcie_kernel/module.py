from pathlib import Path
from flashinfer.jit import gen_jit_spec
ROOT=Path(__file__).resolve().parent

def spec():
    return gen_jit_spec('pcie_ipc_sm80_screen_6870e3ff',
        [ROOT/'upstream/csrc/pcie_ipc_all_reduce.cu'],
        extra_include_paths=[ROOT/'upstream/include',ROOT/'upstream/csrc'],
        extra_ldflags=['-lcuda'])

def load():
    return spec().build_and_load()

if __name__=='__main__':
    spec().build()
    print('COMPILE COMPLETE',flush=True)
