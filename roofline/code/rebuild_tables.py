#!/usr/bin/env python3
"""Rebuild the processed tables in ../data/ from ../data/raw/ and compare.

    python3 rebuild_tables.py [--raw DIR] [--out DIR] [--check DIR]

Writes to --out (default out/tables/ next to data/):
  summary.csv, roofs.csv      microbenchmark points and roofs (summarize.py)
  warnings.csv                points with IQR/median > 0.10, and warm GEMM points
                              more than 10% above min(peak, bandwidth * ai)
  decode_performance.csv      per decode run and case: medians and IQRs over
                              the 7 timed sequences; gpu_tpot_ms is the median
                              over sequences of the mean decode-step time
  operator_counters.csv       Nsight Compute kernels of layer 14 summed per
                              case and module
Parts of --raw that are absent are skipped. Then each file is compared with
the one of the same name in --check (default ../data/ when --raw is the
default, otherwise no comparison): byte for byte, and cell by cell (exact for text and integers,
relative 1e-12 for floats), reporting the largest relative difference. Exit
status 1 if any table differs in value.

The published tables were written after one pandas to_csv/read_csv pass with
the default float parser, which can change the last digit of a float. The
rebuilt tables go through the same pass (roofs.csv is computed from the summary
after that pass and written directly), so they can be compared byte for byte.
"""
import argparse
import csv
import io
import json
import pathlib
import statistics
import sys
from collections import defaultdict

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize import PARTS, load_samples, roofs, summarize  # noqa: E402

# Order of targets in the published tables.
MICRO_TARGETS = ["v100-16", "v100-32", "v100-32-pcie", "a100", "h100", "h200", "h200-mig", "b300",
                 "gh200", "mi100", "mi210", "rtx4080"]
DECODE_TARGETS = ["h200", "h100", "rtx4080"]
CASE_KEYS = ["family", "dtype", "cache", "m", "n", "k", "reps"]
TENSOR = "sm__ops_path_tensor_src_bf16_dst_fp32.sum"
TENSOR_PARTS = [TENSOR.replace(".sum", "_sparsity_off.sum"), TENSOR.replace(".sum", "_sparsity_on.sum")]
LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def ordered(names, order):
    return [t for t in order if t in names] + sorted(set(names) - set(order))


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def runs_in_order(directory):
    return sorted(directory.iterdir(), key=lambda p: int(p.name.split("-")[1]))


# ---------- microbenchmarks ----------

def micro_tables(raw_dir):
    summaries, roof_tables = [], []
    targets = ordered({p.name for part in PARTS for p in (raw_dir / part).glob("*") if p.is_dir()}, MICRO_TARGETS)
    if not targets:
        return {}
    for target in targets:
        files = [raw_dir / part / target / "run-1" / "samples.csv" for part in PARTS]
        device = json.loads((raw_dir / "bench" / target / "run-1" / "device.json").read_text())
        s = summarize(load_samples([f for f in files if f.exists()]), target, device)
        r = roofs(as_published(s), target)
        summaries.append(s)
        roof_tables.append(r)
    summary, roof_table = pd.concat(summaries, ignore_index=True), pd.concat(roof_tables, ignore_index=True)
    return {"summary.csv": summary, "roofs.csv": roof_table, "warnings.csv": warnings_table(summary, roof_table)}


def as_published(table):
    """One pandas to_csv/read_csv pass with the default float parser, as the published
    tables went through; it can change the last digit of a float."""
    return pd.read_csv(io.StringIO(table.to_csv(index=False)))


def case_name(row):
    return ":".join(str(row[k]) if isinstance(row[k], str) else str(float(row[k])) for k in CASE_KEYS)


def warnings_table(summary, roof_table):
    rows = []
    for target, s in summary.groupby("target", sort=False):
        for _, r in s[s.iqr_ratio > 0.10].iterrows():
            rows.append({"target": target, "case": case_name(r), "warning": "IQR/median > 10%", "value": r["iqr_ratio"]})
    roof = {(r.target, r.dtype): r for r in roof_table.itertuples()}
    for _, r in summary[(summary.family == "gemm") & (summary.cache == "warm")].iterrows():
        x = roof.get((r["target"], r["dtype"]))
        if x is None:
            continue
        ratio = r["tflop_s"] / min(x.peak_tflop_s, x.bandwidth_gb_s * r["ai"] / 1000)
        if ratio > 1.1:
            rows.append({"target": r["target"], "case": case_name(r),
                         "warning": "Above empirical reference by >10%; inspect cache/model", "value": ratio})
    return pd.DataFrame(rows, columns=["target", "case", "warning", "value"])


# ---------- decode ----------

def median_iqr(values):
    q = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), q[2] - q[0]


def decode_table(raw_dir):
    fields = ["gpu_prefill_ms", "wall_prefill_ms", "gpu_tpot_ms", "wall_tpot_ms", "wall_sequence_ms",
              "tokens_per_second", "peak_allocated_gib", "peak_reserved_gib"]
    out = []
    root = raw_dir / "decode"
    for target in ordered([p.name for p in root.iterdir() if p.is_dir()], DECODE_TARGETS):
        for run in runs_in_order(root / target):
            info = json.loads((run / "run.json").read_text())
            steps = defaultdict(lambda: ([], []))
            for r in read_rows(run / "steps.csv"):
                gpu, wall = steps[(int(r["batch"]), int(r["prompt_length"]), int(r["repeat"]))]
                gpu.append(float(r["gpu_decode_ms"]))
                wall.append(float(r["wall_decode_ms"]))
            cases = defaultdict(list)
            for r in read_rows(run / "sequences.csv"):
                b, s, rep, t = int(r["batch"]), int(r["prompt_length"]), int(r["repeat"]), int(r["generated_tokens"])
                gpu, wall = steps[(b, s, rep)]
                assert len(gpu) == t - 1, (run, b, s, rep)
                cases[(b, s)].append({
                    "gpu_prefill_ms": float(r["gpu_prefill_ms"]), "wall_prefill_ms": float(r["wall_prefill_ms"]),
                    "gpu_tpot_ms": statistics.mean(gpu), "wall_tpot_ms": statistics.mean(wall),
                    "wall_sequence_ms": float(r["wall_sequence_ms"]),
                    "tokens_per_second": b * t * 1000 / float(r["wall_sequence_ms"]),
                    "peak_allocated_gib": int(r["peak_allocated_bytes"]) / 2**30,
                    "peak_reserved_gib": int(r["peak_reserved_bytes"]) / 2**30,
                    "hash": r["generated_sha256"]})
            for (b, s), seqs in cases.items():
                row = {"target": target, "reference_eligible": info["reference_eligible"],
                       "device_condition": info["device_condition"], "batch": b, "prompt_length": s,
                       "repeats": len(seqs), "distinct_output_hashes": len({x["hash"] for x in seqs})}
                for f in fields:
                    row[f], row[f + "_iqr"] = median_iqr([x[f] for x in seqs])
                out.append(row)
    return pd.DataFrame(out)


# ---------- counters ----------

def counters_table(raw_dir):
    out = []
    root = raw_dir / "counters"
    for target in ordered([p.name for p in root.iterdir() if p.is_dir()], DECODE_TARGETS):
        for run in runs_in_order(root / target):
            kernels = read_rows(run / "kernels.csv")
            tensor = [TENSOR] if TENSOR in kernels[0] else TENSOR_PARTS
            flops = {(int(r["batch"]), int(r["prompt_length"]), r["phase"], r["module"]): int(r["nominal_flops"])
                     for r in read_rows(run / "linear_operators.csv")}
            groups = defaultdict(list)
            for k in kernels:
                groups[(int(k["batch"]), int(k["prompt_length"]), k["phase"], int(k["layer"]), k["module"])].append(k)
            for (b, s, phase, layer, module), items in groups.items():
                dram = sum(float(k["dram__bytes_read.sum"]) + float(k["dram__bytes_write.sum"]) for k in items)
                ns = sum(float(k["gpu__time_duration.sum"]) for k in items)
                f = flops.get((b, s, phase, module))
                assert (f is not None) == (module in LINEARS)
                out.append({"target": target, "batch": b, "prompt_length": s, "phase": phase, "layer": layer,
                            "module": module, "query_tokens": int(items[0]["query_tokens"]),
                            "kv_tokens": int(items[0]["kv_tokens"]), "kernel_count": len(items),
                            "dram_bytes": dram,
                            "l2_bytes": sum(32 * (float(k["lts__t_sectors_op_read.sum"]) + float(k["lts__t_sectors_op_write.sum"]))
                                            for k in items),
                            "profiled_kernel_ns": ns,
                            "nominal_flops": float(f) if f is not None else None,
                            "nominal_flops_per_dram_byte": f / dram if f is not None and dram else None,
                            "nominal_tflops": f / ns / 1000 if f is not None else None,
                            "measured_gb_s": dram / ns,
                            "bf16_tensor_ops": sum(sum(float(k[c]) for c in tensor) for k in items)})
    return pd.DataFrame(out)


# ---------- comparison ----------

def as_number(text):
    try:
        return float(text)
    except ValueError:
        return None


def compare(name, rebuilt_path, published_path):
    a = pd.read_csv(rebuilt_path, dtype=str, keep_default_na=False)
    b = pd.read_csv(published_path, dtype=str, keep_default_na=False)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        print(f"{name}: different columns or row count, {a.shape} vs {b.shape}")
        return False
    worst, mismatches = 0.0, 0
    for col in a.columns:
        for x, y in zip(a[col], b[col]):
            fx, fy = as_number(x), as_number(y)
            if fx is None or fy is None:
                mismatches += x != y
                continue
            rel = abs(fx - fy) / abs(fy) if fy else abs(fx)
            worst = max(worst, rel)
            mismatches += rel > 1e-12
    identical = rebuilt_path.read_bytes() == published_path.read_bytes()
    status = "byte-identical" if identical else ("equal values" if not mismatches else f"{mismatches} cells differ")
    print(f"{name}: {len(a)} rows, {status}; largest relative difference {worst:.2g}")
    return mismatches == 0


def main():
    data = HERE.parent / "data"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=pathlib.Path, default=data / "raw")
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "out" / "tables")
    ap.add_argument("--check", type=pathlib.Path)
    args = ap.parse_args()
    check = args.check or (data if args.raw.resolve() == (data / "raw").resolve() else None)
    args.out.mkdir(parents=True, exist_ok=True)
    tables = micro_tables(args.raw)
    if (args.raw / "decode").is_dir():
        tables["decode_performance.csv"] = decode_table(args.raw)
    if (args.raw / "counters").is_dir():
        tables["operator_counters.csv"] = counters_table(args.raw)
    ok = True
    for name, table in tables.items():
        # roofs.csv was computed from the published summary values and written directly.
        (table if name == "roofs.csv" else as_published(table)).to_csv(args.out / name, index=False)
        print("wrote", args.out / name)
        if check and (check / name).exists():
            ok &= compare(name, args.out / name, check / name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
