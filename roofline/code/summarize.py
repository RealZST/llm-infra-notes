#!/usr/bin/env python3
"""Turn raw samples.csv files from run_local.sh into summary.csv and roofs.csv.

    python3 summarize.py OUTPUT_DIR [TARGET]

OUTPUT_DIR holds bench/ and bandwidth/ (and fp8/, fp4/ if present), each with a
samples.csv; TARGET (default "local") fills the target column. The output has
the same columns and row order as ../data/summary.csv and ../data/roofs.csv.
rebuild_tables.py uses the same functions on ../data/raw/.

Definitions (the same ones behind ../data/):
  * every point is timed 7 times; the reported time is the median;
  * iqr_ratio = (p75 - p25) / median; a point above 0.10 is "unstable";
  * gb_s = bytes / median_ms / 1e6, tflop_s = flops / median_ms / 1e9,
    ai = flops / bytes, with bytes and flops as the kernel reports them
    (GEMM: sizeof * (MK + KN + MN) bytes and 2MNK flops);
  * bandwidth roof = the highest bandwidth any data-moving kernel reaches on a
    working set far larger than L2: read_reduce, or the FMA sweep at R = 1
    (almost no arithmetic). On every GPU but B300 that is read_reduce;
  * compute roof per dtype = best median among stable warm GEMMs;
  * ridge = 1000 * peak_tflop_s / bandwidth_gb_s.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

PARTS = ("bench", "bandwidth", "fp8", "fp4")
KEYS = ["family", "dtype", "cache", "m", "n", "k", "reps", "bytes", "flops", "footprint_bytes"]
DTYPE_ORDER = ["fp64", "fp32", "tf32", "fp16", "bf16", "fp8_e4m3", "nvfp4"]


def load_samples(paths):
    """Concatenate samples.csv files in the given order (bench, bandwidth, fp8, fp4)."""
    raw = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in paths], ignore_index=True)
    assert raw.valid.all(), "a numerical check failed; see samples.csv"
    return raw


def summarize(raw, target, device):
    """One row per point, in the order the points were measured."""
    raw = raw.astype({k: float for k in KEYS[3:]})
    g = raw.groupby(KEYS, sort=False).ms
    s = g.median().rename("median_ms").reset_index()
    s.insert(len(KEYS), "target", target)
    s["p10_ms"] = g.agg(lambda x: np.percentile(x, 10)).values
    s["p90_ms"] = g.agg(lambda x: np.percentile(x, 90)).values
    s["iqr_ratio"] = g.agg(lambda x: (np.percentile(x, 75) - np.percentile(x, 25)) / np.median(x)).values
    s["gb_s"] = s.bytes / s.median_ms / 1e6
    s["tflop_s"] = s.flops / s.median_ms / 1e9
    s["ai"] = s.flops / s.bytes
    s["max_error"] = raw.groupby(KEYS, sort=False).error.max().values
    s["l2_bytes"] = device["l2_bytes"]
    s["device"] = device["name"]
    return s


def roofs(s, target):
    moving = s[((s.family == "read_reduce") & (s.footprint_bytes >= 2**28)) | ((s.family == "fma") & (s.reps == 1))]
    top = moving.loc[moving.gb_s.idxmax()]
    bw = float(top.gb_s)
    bw_case = (f"read_reduce:{int(top['n'])}" if top["family"] == "read_reduce"
               else f"fma:{top['dtype']}:reps={int(top['reps'])}:{int(top['n'])}")
    rows = []
    stable = s[(s.family == "gemm") & (s.cache == "warm") & (s.iqr_ratio <= 0.10)]
    for dtype in sorted(stable.dtype.unique(), key=DTYPE_ORDER.index):
        q = stable[stable.dtype == dtype]
        best = q.loc[q.tflop_s.idxmax()]
        rows.append({"target": target, "dtype": dtype, "bandwidth_gb_s": bw, "peak_tflop_s": best.tflop_s,
                     "ridge_flop_byte": 1000 * best.tflop_s / bw, "bw_case": bw_case,
                     "compute_case": f"{int(best.m)}x{int(best.n)}x{int(best.k)}"})
    return pd.DataFrame(rows)


def main():
    out = pathlib.Path(sys.argv[1])
    target = sys.argv[2] if len(sys.argv) > 2 else "local"
    raw = load_samples([out / d / "samples.csv" for d in PARTS if (out / d / "samples.csv").exists()])
    device = json.loads((out / "bench" / "device.json").read_text())
    s = summarize(raw, target, device)
    s.to_csv(out / "summary.csv", index=False)
    r = roofs(s, target)
    r.to_csv(out / "roofs.csv", index=False)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
