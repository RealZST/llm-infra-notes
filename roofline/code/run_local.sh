#!/bin/bash
# Build and run the microbenchmarks on the single visible NVIDIA GPU.
#
#   bash run_local.sh OUTPUT_DIR [quick]
#   FP8=1 NVFP4=1 bash run_local.sh OUTPUT_DIR      # with the low-precision GEMMs
#
# OUTPUT_DIR receives bench/samples.csv, bench/device.json, bandwidth/samples.csv,
# summary.csv and roofs.csv (plus build/, compiler.txt and gpu.txt).
# "quick" runs a reduced grid with 3 samples per point (a few seconds) to check
# that everything compiles and validates; omit it for the full grid used in the
# note (7 samples per point, several minutes on a data-centre GPU).
#
# Optional, off by default (each adds fp8/ or fp4/ samples.csv, which
# summarize.py includes in summary.csv and roofs.csv):
#   FP8=1    FP8 E4M3 GEMM through cuBLASLt (fp8.cu); compute capability 8.9 or
#            higher (Ada, Hopper, Blackwell) and CUDA 12.0 or newer, otherwise skipped
#   NVFP4=1  NVFP4 GEMM through cuBLASLt (fp4.cu); compute capability 10.0 or
#            higher (Blackwell) and CUDA 12.8 or newer, otherwise skipped
# Both always run their full grid (14 shapes, 7 samples each), also with "quick".
#
# Requirements: CUDA toolkit (nvcc, cuBLAS, cuBLASLt) and exactly one visible
# GPU, for example CUDA_VISIBLE_DEVICES=0. The kernels and shapes are unchanged
# from the cluster runs; only the scheduler wrapper is gone.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?output dir}; QUICK=${2:-}
mkdir -p "$OUT/build" "$OUT/bench" "$OUT/bandwidth"
nvcc --version | tail -2 > "$OUT/compiler.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv > "$OUT/gpu.txt"
nvcc -O3 -std=c++17 -arch=native "$HERE/bench.cu" -lcublas -o "$OUT/build/bench"
nvcc -O3 -std=c++17 -arch=native "$HERE/bandwidth.cu" -lcublas -o "$OUT/build/bandwidth"
"$OUT/build/bench" "$OUT/bench" $QUICK
"$OUT/build/bandwidth" "$OUT/bandwidth"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
CUDA=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
at_least() { awk -v have="$1" -v need="$2" 'BEGIN { split(have, h, "."); split(need, n, ".");
 exit !(h[1] + 0 > n[1] + 0 || (h[1] + 0 == n[1] + 0 && h[2] + 0 >= n[2] + 0)) }'; }
low_precision() {  # NAME SOURCE MIN_COMPUTE_CAPABILITY MIN_CUDA
 if ! at_least "$CC" "$3"; then echo "skipping $1: compute capability $CC < $3"; return; fi
 if ! at_least "$CUDA" "$4"; then echo "skipping $1: CUDA $CUDA < $4"; return; fi
 mkdir -p "$OUT/$1"
 nvcc -O3 -std=c++17 -arch=native "$HERE/$2" -lcublas -lcublasLt -o "$OUT/build/$1"
 "$OUT/build/$1" "$OUT/$1"
}
[[ "${FP8:-0}" != 1 ]] || low_precision fp8 fp8.cu 8.9 12.0
[[ "${NVFP4:-0}" != 1 ]] || low_precision fp4 fp4.cu 10.0 12.8

python3 "$HERE/summarize.py" "$OUT"
echo "done: $OUT/summary.csv and $OUT/roofs.csv"
