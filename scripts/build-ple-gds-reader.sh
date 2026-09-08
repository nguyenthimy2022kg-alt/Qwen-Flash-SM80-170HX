#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root=${CUDA_ROOT:-/usr/local/cuda-13.0}
cuda_header_dir=${CUDA_HEADER_DIR:-$cuda_root/targets/x86_64-linux/include}
if [[ ! -f "$cuda_header_dir/cuda.h" ]]; then
    printf 'CUDA Driver API header cuda.h was not found\n' >&2
    exit 1
fi
cufile_include="$cuda_root/targets/x86_64-linux/include"
cufile_lib="$cuda_root/targets/x86_64-linux/lib"
suffix=$(python3-config --extension-suffix)
output="$repo/src/ple_gds/_native$suffix"
persistent_output="$repo/src/ple_gds/_persistent_native$suffix"

printf 'CUDA_HEADER_DIR=%s\n' "$cuda_header_dir"
printf 'CUFILE_INCLUDE=%s\n' "$cufile_include"
printf 'CUFILE_LIB=%s\n' "$cufile_lib"
printf 'OUTPUT=%s\n' "$output"
printf 'PERSISTENT_OUTPUT=%s\n' "$persistent_output"

g++ -O2 -std=c++17 -Wall -Wextra -Werror -shared -fPIC \
    $(python3-config --includes) \
    -I"$cuda_header_dir" -I"$cufile_include" \
    "$repo/csrc/ple_gds/ple_gds_native.cpp" \
    -L"$cufile_lib" -Wl,-rpath,"$cufile_lib" -lcufile -lcuda \
    -o "$output"
sha256sum "$output"

g++ -O2 -std=c++17 -Wall -Wextra -Werror -shared -fPIC \
    $(python3-config --includes) \
    -I"$cuda_header_dir" -I"$cufile_include" \
    "$repo/csrc/ple_gds/ple_gds_persistent.cpp" \
    -L"$cufile_lib" -Wl,-rpath,"$cufile_lib" -lcufile -lcuda \
    -o "$persistent_output"
sha256sum "$persistent_output"
