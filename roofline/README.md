# Roofline model, from measured data

Experiments and data behind the note [The Roofline Model, Starting from Measured Data](https://realzst.github.io/notes/roofline/) ([中文](https://realzst.github.io/notes/roofline/zh/)). Every number, table and figure in the note can be recomputed from the data in this folder.

## What is here

| Path | Content |
|---|---|
| `data/roofs.csv` | Measured roofline per GPU and dtype: bandwidth, peak, ridge point, and which case produced each |
| `data/summary.csv` | Every microbenchmark point (1818 rows): stream kernels, FMA sweep, cuBLAS/rocBLAS GEMM warm and cold, FP8/NVFP4 where supported |
| `data/devices.csv` | GPU name, memory, SM count, L2 size, power limit |
| `data/warnings.csv` | Points flagged as unstable (IQR/median > 10%) or above the measured roofline by more than 10% |
| `data/operator_counters.csv` | Nsight Compute counters per module of decoder layer 14 (H100, H200, RTX 4080): DRAM and L2 bytes, kernel time, kernel count, Tensor op counts; nominal 2MNK FLOPs for the seven linear layers |
| `data/decode_performance.csv` | Qwen2.5-7B-Instruct bf16 prefill/decode timings per run, batch and prompt length, H100, H200, RTX 4080 |
| `data/raw/bench/`, `bandwidth/`, `fp8/`, `fp4/` | Per-sample microbenchmark records (`samples.csv`, 7 per point) behind `summary.csv`, with `device.json` and GPU telemetry; one run per GPU |
| `data/raw/decode/`, `data/raw/counters/` | Per-sequence and per-step decode timings behind `decode_performance.csv`; per-kernel Nsight Compute counters behind `operator_counters.csv`. Layout and columns: [data/raw/README.md](data/raw/README.md) |
| `code/bench.cu` | The microbenchmark: copy/add/triad, FMA sweep, GEMM grid; CUDA and HIP |
| `code/bandwidth.cu` | The read-mostly bandwidth calibration kernel |
| `code/fp8.cu`, `code/fp4.cu` | FP8 E4M3 and NVFP4 GEMM through cuBLASLt, behind `data/raw/fp8/` and `data/raw/fp4/`; both include `bench.cu` |
| `code/run_local.sh` | Build and run the microbenchmarks on one NVIDIA GPU; FP8 and NVFP4 on request |
| `code/summarize.py` | `samples.csv` → `summary.csv` and `roofs.csv`, with the definitions used in the note and the same columns as `data/` |
| `code/rebuild_tables.py` | `data/raw/` → `summary.csv`, `roofs.csv`, `warnings.csv`, `decode_performance.csv`, `operator_counters.csv`, compared with `data/` |
| `code/decode/run.py` | The decode harness behind `data/raw/decode/` and `data/raw/counters/`: Qwen2.5-7B-Instruct at a pinned revision; modes `performance` (timing), `trace` (PyTorch profiler), `counters` (one forward pass for Nsight Compute), `pilot` |
| `code/decode/application.json` | Its settings: model revision, batches, prompt lengths, generated tokens, warm-ups, repeats, counter layer and cases, seed |
| `code/decode/prepare_model.py`, `code/decode/model_files.json` | Download the pinned revision, check the SHA256 of every file the runs loaded, write the manifest `run.py` reads |
| `code/decode/run_decode.sh` | Run `run.py` in one mode with GPU identity, `nvidia-smi` telemetry and environment as in the published runs |
| `code/decode/ncu_counters.sh` | The Nsight Compute command for one case and phase (layer 14 only, no cache or clock control) and the CSV export |
| `code/decode/hardware_probe.cu` | GPU name, memory, SM count, compute capability and UUID, for `run_decode.sh` |
| `code/decode/export_raw.py` | Run directories of `run_decode.sh` → the `data/raw/decode/` and `data/raw/counters/` layout |
| `code/plot.py` | The figures; `--numbers` prints the numbers quoted in the text |
| `code/requirements.txt`, `code/requirements-decode.txt` | Python packages for `plot.py`, `summarize.py` and `rebuild_tables.py`, and for `code/decode/` (versions of the published runs) |
| `figures/` | Figures 1–9 of the note, plus `fig4-all-gpus.png`: Figure 4's matrix multiply grid for all eight GPUs of Figure 1 |

All measurement code behind `data/` is here. Left out are the scheduler batch scripts that wrapped it (job directives, module loads, allocation records, a node-local copy of the model); `run_local.sh`, `run_decode.sh` and `ncu_counters.sh` run the same programs with the same kernels, shapes, harness modes and Nsight Compute command, with paths taken from arguments. The original aggregation scripts are also left out; `rebuild_tables.py` rebuilds the processed tables in `data/` from `data/raw/` byte for byte.

## Reproduce

Figures and numbers from the published data (no GPU needed):

```bash
pip install -r code/requirements.txt
python3 code/plot.py            # writes figures/, prints numbers
python3 code/plot.py --numbers  # numbers only
```

The processed tables from `data/raw/` (no GPU needed; writes `out/tables/` and compares each file with `data/`):

```bash
python3 code/rebuild_tables.py
```

The microbenchmarks on a local NVIDIA GPU (CUDA toolkit with nvcc, cuBLAS and cuBLASLt; a few minutes):

```bash
CUDA_VISIBLE_DEVICES=0 bash code/run_local.sh out/                # full grid, 7 samples per point
CUDA_VISIBLE_DEVICES=0 bash code/run_local.sh out/ quick          # smoke test, seconds
CUDA_VISIBLE_DEVICES=0 FP8=1 bash code/run_local.sh out/          # plus FP8 (compute capability 8.9+)
CUDA_VISIBLE_DEVICES=0 FP8=1 NVFP4=1 bash code/run_local.sh out/  # plus NVFP4 (Blackwell, CUDA 12.8+)
```

`out/summary.csv` and `out/roofs.csv` have the same columns as `data/`, with `target` set to `local` (`python3 code/summarize.py out/ NAME` sets another name). FP8 and NVFP4 are skipped when the GPU or CUDA version does not support them. For AMD, compile `bench.cu` with `hipcc -O3 -std=c++17 -DUSE_HIP bench.cu -lrocblas`.

The decode measurement (one NVIDIA GPU with at least 16 GB; CUDA toolkit for `hardware_probe.cu`; Nsight Compute `ncu` for counters). First download and check the model (about 15 GB):

```bash
pip install -r code/requirements-decode.txt
python3 code/decode/prepare_model.py model/application_model.json --cache-dir model/hf
```

Then one run per mode; each `RUN_DIR` must be new:

```bash
export CUDA_VISIBLE_DEVICES=0
bash code/decode/run_decode.sh performance out/decode-1 model/application_model.json   # timing, 3 x 3 grid
bash code/decode/run_decode.sh trace       out/trace-1  model/application_model.json   # PyTorch profiler traces
bash code/decode/run_decode.sh counters    out/ncu-1    model/application_model.json   # Nsight Compute, layer 14
bash code/decode/ncu_counters.sh out/ncu-1 8 4096 decode                              # one counter case again
```

`PYTHON=/path/to/python` selects the interpreter. The published tables used two or more performance runs per GPU. To convert run directories into the published layout and rebuild the tables from them:

```bash
python3 code/decode/export_raw.py decode   out/raw/decode/local   out/decode-1 out/decode-2
python3 code/decode/export_raw.py counters out/raw/counters/local out/ncu-1
python3 code/rebuild_tables.py --raw out/raw --out out/tables-local
```

`export_raw.py` applied to the original run directories reproduces every file in `data/raw/decode/` and `data/raw/counters/` byte for byte, and `rebuild_tables.py` reproduces the five processed tables in `data/` byte for byte.

## Protocol, in short

- Timing: CUDA events around a run of back-to-back calls of the kernel (up to 200, about 20 ms; one call for cold GEMM), time per call = interval / calls; 5 untimed warm-ups; each point 7 samples, median reported; `iqr_ratio` = (p75 − p25)/median, above 0.10 the point is excluded from calibration.
- Bytes and FLOPs are the algorithmic counts: stream kernels count each array once; FMA counts 2 FLOP; GEMM counts `sizeof·(MK+KN+MN)` bytes and `2MNK` FLOP. Intensity on the plots is this algorithmic ratio, except in Figure 6, which divides `2MNK` by the DRAM bytes Nsight Compute measured.
- Roofline bandwidth: the highest bandwidth of a data-moving kernel on a working set far above L2, i.e. the larger of `read_reduce` (≥ 256 MiB) and the FMA sweep at R = 1 (512 MiB read + written). Only on B300 is the FMA sweep higher (6387 vs 5615 GB/s for `read_reduce`). On MI210 the highest `read_reduce` value is at 256 MiB (1334 GB/s; 1323 GB/s at 512 MiB); `bw_case` in `roofs.csv` names the case for every GPU. Roofline peak per dtype: best stable warm GEMM median among squares up to 8192³ and the M sweep (`M ∈ {1..2048}`, `N = K = 8192`).
- Cold GEMM points evict L2 before each sample with a buffer of at least max(128 MiB, 4 × L2); eviction time is outside the measured interval.
- cuBLAS: `cublasGemmEx`, FP32 accumulation for FP16/BF16/TF32, `CUBLAS_COMPUTE_32F_PEDANTIC` for strict FP32, FP64 accumulation for FP64.
- Decode: `transformers` 4.51.3, `torch` 2.6.0, bf16 weights, SDPA attention, TF32 off, greedy, `logits_to_keep=1`; 2 warm-up sequences then 7 timed sequences of 64 generated tokens; `gpu_tpot_ms` is the mean decode-step time of a sequence, and `decode_performance.csv` holds its median over the 7 sequences of each run; the note uses the median of that value over the runs with `reference_eligible` true (2 on H100 and RTX 4080, 3 on H200).
- Microbenchmarks were compiled with CUDA 12.6 on the cluster GPUs, CUDA 13.0 on the RTX 4080 workstation, and ROCm (hipcc, rocBLAS) for the AMD cards. The decode and Nsight Compute runs used the PyTorch 2.6.0 CUDA 12.4 build (`run.json`).

## Software and hardware

GPUs in `data/`: V100 SXM2 16 GB / 32 GB, V100 PCIe 32 GB, A100 SXM4 80 GB, H100 SXM 80 GB HBM3, H200, H200 MIG 3g.71gb, B300, GH200 144 GB, AMD MI100, AMD MI210, GeForce RTX 4080. The note's figures use a subset; `summary.csv` has all of them. Figure 2 leaves out B300: its copy bandwidth keeps rising past its L2 size (126.5 MiB): 2729 GB/s at 32 MiB, 2759 GB/s at 128 MiB, 3636 GB/s at 512 MiB. Warm vs cold GEMM points are in `summary.csv` (`cache` column) but not plotted in the note.
