# Raw measurement records

The per-sample records behind the processed tables in `data/`. Only the runs the published tables were built from are here: one run per GPU and category for the microbenchmarks, every run that appears in `decode_performance.csv`, and the one counter run per GPU behind `operator_counters.csv`.

Host and scheduler identifiers were removed before publishing: job IDs (directories are renamed `run-1`, `run-2`, … per category and GPU, in time order), host and node names, GPU UUIDs, PCI bus IDs, user and file-system paths, process names and IDs, command lines and absolute timestamps. Scheduler records, logs, nvidia-smi dumps, compiler output and source snapshots are not included.

## Layout

```text
raw/
├── bench/<target>/run-1/       samples.csv, device.json, telemetry.csv   copy/add/triad, FMA sweep, GEMM (code/bench.cu)
├── bandwidth/<target>/run-1/   samples.csv, telemetry.csv                read_reduce calibration (code/bandwidth.cu)
├── fp8/<target>/run-1/         samples.csv                               FP8 E4M3 GEMM (code/fp8.cu; H100, H200, H200 MIG, GH200, B300, RTX 4080)
├── fp4/b300/run-1/             samples.csv                               NVFP4 GEMM (code/fp4.cu; B300 only)
├── decode/<target>/run-N/      run.json, sequences.csv, steps.csv, telemetry.csv   Qwen2.5-7B decode timing (code/decode/run.py)
└── counters/<target>/run-1/    run.json, kernels.csv, linear_operators.csv         Nsight Compute, layer 14 (code/decode/ncu_counters.sh)
```

`<target>` uses the same names as the `target` column in `data/`. AMD runs (`mi100`, `mi210`) have no `telemetry.csv`; their monitor output was a text dump, not a table. Two H200 decode runs (`run-1`, `run-2`) have no telemetry table either.

## Microbenchmarks: `samples.csv`

One row per timed sample, exactly as `bench.cu` / `bandwidth.cu` write it (and `run_local.sh` produces in `out/bench/` and `out/bandwidth/`).

| Column | Meaning |
|---|---|
| `family` | `copy`, `add`, `triad`, `fma`, `gemm`, `read_reduce` |
| `dtype`, `cache` | data type; `warm`, `cold` (GEMM with L2 evicted before each sample) or `streaming` (FMA sweep) |
| `m`, `n`, `k` | GEMM shape; for stream kernels `n` is the element count |
| `reps` | FMA iterations per element (FMA sweep only) |
| `bytes`, `flops` | algorithmic bytes and FLOPs per kernel call |
| `footprint_bytes` | working set in bytes |
| `sample` | sample index, 0–6 |
| `loops` | kernel calls between the two CUDA events of this sample |
| `ms` | time per kernel call, ms (event interval / `loops`) |
| `error`, `valid` | numerical error against a CPU reference (GEMM: RMS error over 64 sampled outputs, relative to the reference RMS); `valid` = 1 if within tolerance |

`data/summary.csv` is the per-point aggregate: concatenate `bench`, `bandwidth`, `fp8` and `fp4` samples of a target, group by `family, dtype, cache, m, n, k, reps, bytes, flops, footprint_bytes`, and take the median of `ms` (`median_ms`), the 10th/90th percentiles, and `iqr_ratio`; `max_error` is the largest `error`. This is the grouping in `code/summarize.py`. Every point has 7 samples; recomputing from these files reproduces all 1818 `median_ms` values of `data/summary.csv`; `code/rebuild_tables.py` rebuilds `summary.csv`, `roofs.csv` and `warnings.csv` from them.

`device.json`: `name`, `memory_bytes`, `l2_bytes`, `sm_count`, `major`/`minor` (compute capability, or the HIP equivalent), `arch`, `backend` (`cuda` or `hip`).

`telemetry.csv` (NVIDIA only): nvidia-smi polled about once per second during the run. `t_s` is seconds since the first poll; the other columns are nvidia-smi query fields with units in the header (`[N/A]` where a MIG instance does not report them).

## Decode: `decode/<target>/run-N/`

Qwen2.5-7B-Instruct, bf16, batch ∈ {1, 8, 16} × prompt length ∈ {128, 1024, 4096}, 64 generated tokens, 2 untimed warm-up sequences then 7 timed sequences per case. Cases that did not fit in memory are listed in `run.json` (`oom_cases`) and have no rows.

- `run.json`: GPU name and memory, software versions, protocol settings, and `reference_eligible` / `device_condition`. A run is not reference-eligible when an active thermal slowdown was observed; H200 `run-1` is such a run and is kept as a diagnostic, as in `decode_performance.csv`.
- `sequences.csv`: one row per timed sequence. `gpu_prefill_ms` (CUDA events around the prefill forward + argmax), `wall_prefill_ms`, `wall_sequence_ms` (host wall clock for the whole sequence), `final_cache_length`, `peak_allocated_bytes` / `peak_reserved_bytes` (PyTorch allocator), `input_sha256` (hash of the input token IDs; identical across GPUs) and `generated_sha256` (hash of the generated token IDs).
- `steps.csv`: one row per decode step. The first generated token comes from prefill, so each sequence has 63 steps; `gpu_decode_ms` is the CUDA-event time of the step (forward + argmax for the whole batch), `wall_decode_ms` the host wall time.
- `telemetry.csv`: as above, plus `pstate` and the clock-event (throttle) reasons used for `reference_eligible`.

`data/decode_performance.csv` has one row per run and case, in the order `run-1`, `run-2`, … within each target: `gpu_tpot_ms` is the median over the 7 sequences of the mean of that sequence's `gpu_decode_ms`; `gpu_prefill_ms` is the median of `gpu_prefill_ms`; `*_iqr` columns are p75 − p25 over the 7 sequences; `tokens_per_second` is the median of batch · 64 · 1000 / `wall_sequence_ms`. `code/decode/export_raw.py` writes `run.json`, `sequences.csv`, `steps.csv` and `telemetry.csv` from the harness output (`timings.jsonl`, `metadata.json`, `status.json`, `gpu_identity.json`, nvidia-smi telemetry); `code/rebuild_tables.py` computes `decode_performance.csv` from them.

## Counters: `counters/<target>/run-1/`

Nsight Compute on the kernels inside the NVTX range of decoder layer 14, for (batch, prompt) ∈ {(1, 1024), (8, 1024), (8, 4096)}, prefill and decode. No cache or clock control. On the RTX 4080 only (1, 1024) fits.

- `run.json`: Nsight Compute version, metrics and units, measured and out-of-memory cases.
- `kernels.csv`: one row per profiled kernel. `module` is the innermost NVTX range (`q_proj`, `input_norm`, `attention`, …; `<phase>_layer14` for kernels outside any module range); `query_tokens` / `kv_tokens` are the captured query and KV lengths; `dram__bytes_read.sum`, `dram__bytes_write.sum` in bytes; `gpu__time_duration.sum` in ns; `lts__t_sectors_op_read.sum`, `lts__t_sectors_op_write.sum` in 32-byte L2 sectors; `sm__ops_path_tensor_*` are Tensor-path operation counts (Nsight 2024.3 on H100/H200 reports the BF16 aggregate `sm__ops_path_tensor_src_bf16_dst_fp32.sum`; 2025.3 on the RTX 4080 reports only its `_sparsity_off` / `_sparsity_on` parts).
- `linear_operators.csv`: the runtime input shape, `in_features`, `out_features` and nominal FLOPs (2·tokens·in·out) of the seven linear layers.

`data/operator_counters.csv` sums `kernels.csv` per case and `module`: `dram_bytes` = read + write, `l2_bytes` = 32 × (read + write sectors), `profiled_kernel_ns` = sum of `gpu__time_duration.sum`, `kernel_count`, `bf16_tensor_ops` from the BF16 metric(s) above, and `nominal_flops` from `linear_operators.csv`. `code/decode/export_raw.py` writes the three files from the per-case Nsight Compute exports; `code/rebuild_tables.py` computes `operator_counters.csv`.
