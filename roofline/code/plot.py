#!/usr/bin/env python3
"""Figures and numbers for the roofline note.

Reads ../data/*.csv, writes ../figures/*.png, and prints the numbers quoted
in the note so a reader can check the text against the data:

    python3 plot.py            # figures + numbers
    python3 plot.py --numbers  # numbers only
"""
import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"

# Display names and one colour per GPU, shared by every figure.
NAME = {
    "v100-32": "V100 SXM2 32GB",
    "a100": "A100 80GB",
    "h100": "H100 80GB",
    "h200": "H200",
    "b300": "B300",
    "mi100": "AMD MI100",
    "mi210": "AMD MI210",
    "rtx4080": "RTX 4080",
}
# Okabe-Ito palette, one colour per GPU, shared by every figure.
COLOR = {
    "v100-32": "#D55E00",
    "a100": "#CC79A7",
    "h100": "#E69F00",
    "h200": "#56B4E9",
    "b300": "#882255",
    "mi100": "#009E73",
    "mi210": "#0072B2",
    "rtx4080": "#999933",
}
DTYPE_COLOR = {"fp16": "#E69F00", "tf32": "#56B4E9", "fp32": "#009E73", "fp64": "#332288"}

# Qwen2.5-7B-Instruct, bf16. Parameter count and bytes come from the model's
# metadata; the input embedding is a table lookup, so one decode step reads
# every weight except that table.
PARAM_BYTES = 15_231_233_024
EMBED_BYTES = 152_064 * 3_584 * 2
DECODE_BYTES = PARAM_BYTES - EMBED_BYTES

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "figure.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "lines.linewidth": 1.6,
        "lines.markersize": 3.5,
    }
)


def load():
    s = pd.read_csv(DATA / "summary.csv")
    r = pd.read_csv(DATA / "roofs.csv")
    d = pd.read_csv(DATA / "devices.csv")
    p = pd.read_csv(DATA / "decode_performance.csv")
    w = pd.read_csv(DATA / "warnings.csv")
    o = pd.read_csv(DATA / "operator_counters.csv")
    return s, r, d, p, w, o


def roof(r, target, dtype):
    row = r[(r.target == target) & (r.dtype == dtype)].iloc[0]
    return float(row.bandwidth_gb_s), float(row.peak_tflop_s), float(row.ridge_flop_byte)


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", FIGS / name)



def roofline_over_m(bw, pk, sizeof, n=8192, k=8192):
    """The roofline as a function of M for an M x K by K x N matrix multiply, on a
    dense grid of M so the corner at the ridge point is drawn exactly."""
    m = np.logspace(0, np.log10(8192), 600)
    ai = 2 * m * n * k / (sizeof * (m * k + k * n + m * n))
    return m, np.minimum(bw * ai / 1000, pk)


SIZEOF = {"fp64": 8, "fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8_e4m3": 1}

# ---------------------------------------------------------------- figure 1
def fig_roofline_fp16(r):
    cards = ["v100-32", "a100", "h100", "h200", "b300", "mi100", "mi210", "rtx4080"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, x in zip(axes, [np.logspace(-1, 4, 400), np.linspace(0, 1500, 400)]):
        for t in cards:
            bw, pk, ridge = roof(r, t, "fp16")
            ax.plot(x, np.minimum(bw * x / 1000, pk), color=COLOR[t], label=f"{NAME[t]}: {bw:.0f} GB/s; {pk:.0f} TFLOP/s")
            ax.plot([ridge], [pk], "o", color=COLOR[t], ms=5)
        ax.set_xlabel("Arithmetic intensity  I  (FLOP / byte)")
        ax.set_ylabel("Attainable FP16 throughput  (TFLOP/s)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.1, 1e4)
    axes[0].set_ylim(0.05, 4000)
    axes[0].set_title("log-log: parallel slopes, intercept = bandwidth")
    axes[1].set_title("linear axes: the same eight rooflines")
    fig.suptitle("Measured FP16 rooflines, eight GPUs", y=1.02)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=9)
    save(fig, "fig1-roofline-fp16.png")


# ---------------------------------------------------------------- figure 2
def fig_bandwidth_working_set(s, d, r):
    cards = ["v100-32", "a100", "h100", "h200", "mi100", "mi210", "rtx4080"]
    c = s[(s.family == "copy") & (s.cache == "warm")]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for t in cards:
        q = c[c.target == t].sort_values("footprint_bytes")
        l2 = float(d[d.target == t].l2_mib.iloc[0])
        ax.plot(q.footprint_bytes / 2**20, q.gb_s, "-o", color=COLOR[t], label=f"{NAME[t].replace(' SXM2 32GB', '').replace(' 80GB', '')} (L2 {l2:g} MiB)")
        ax.axvline(l2, color=COLOR[t], ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Working set of the copy kernel  (MiB)")
    ax.set_ylabel("Copy bandwidth  (GB/s)")
    ax.set_title("Bandwidth depends on the working set")
    ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0), frameon=False, fontsize=8, handlelength=1.6, labelspacing=0.3)
    save(fig, "fig2-bandwidth-working-set.png")


# ---------------------------------------------------------------- figure 3
def fig_fma_sweep(s, r):
    cards = ["v100-32", "a100", "h100", "h200", "b300", "mi100", "mi210", "rtx4080"]
    f = s[s.family == "fma"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, dtype in zip(axes, ["fp32", "fp64"]):
        for t in cards:
            q = f[(f.target == t) & (f.dtype == dtype)].sort_values("ai")
            bw, _, _ = roof(r, t, dtype)
            peak = q.tflop_s.max()
            xx = np.logspace(np.log10(q.ai.min()), np.log10(q.ai.max()), 200)
            ax.plot(xx, np.minimum(bw * xx / 1000, peak), ls="--", lw=1.1, color=COLOR[t], alpha=0.5)
            ax.plot(q.ai, q.tflop_s, "-o", ms=3.5, color=COLOR[t], label=NAME[t])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic intensity  I  (FLOP / byte)")
        ax.set_title(f"{dtype.upper()} FMA sweep (solid: measured, dashed: min(B·I, peak))", fontsize=10.5)
    axes[0].set_ylabel("Throughput  (TFLOP/s)")
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=9.5)
    save(fig, "fig3-fma-sweep.png")


# ---------------------------------------------------------------- figure 4
def fig_gemm_grid(s, r, cards, name, title):
    """Rows: GPUs. Columns: FP64, FP32, FP16. Dashed roofline, squares for square
    GEMMs, circles for the M sweep."""
    dtypes = ["fp64", "fp32", "fp16"]
    xx = np.logspace(-1, 4, 400)
    fig, axes = plt.subplots(len(cards), 3, figsize=(12, 3.5 * len(cards)), squeeze=False)
    for i, t in enumerate(cards):
        g = s[(s.family == "gemm") & (s.cache == "warm") & (s.target == t)]
        for j, dtype in enumerate(dtypes):
            ax = axes[i, j]
            bw, pk, ridge = roof(r, t, dtype)
            ax.loglog(xx, np.minimum(bw * xx / 1000, pk), "--", color="#555555", label="Measured roofline")
            q = g[g.dtype == dtype]
            sq = q[(q.m == q.n) & (q.n == q.k)]
            sw = q[(q.n == 8192) & (q.k == 8192) & (q.m != 8192)]
            ax.loglog(sq.ai, sq.tflop_s, "s", color="#0072B2", ms=5, label="Square matmul")
            ax.loglog(sw.ai, sw.tflop_s, "o", color="#D55E00", ms=5, label="M sweep")
            ax.set_xlim(0.1, 1e4)
            pk_s = f"{pk:.0f}" if pk >= 100 else f"{pk:.3g}"
            short = NAME[t].replace(" SXM2 32GB", "").replace(" 80GB", "")
            ax.set_title(f"{short} {dtype.upper()}: roofline {pk_s} TFLOP/s, ridge point {ridge:.3g} FLOP/byte", fontsize=8.5)
            if i == len(cards) - 1:
                ax.set_xlabel("Arithmetic intensity  I  (FLOP / byte)")
            if j == 0:
                ax.set_ylabel("Measured throughput  (TFLOP/s)")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save(fig, name)


def fig_h100_roofline(s, r):
    """Figure 4 in the note: H100, B300 and RTX 4080."""
    fig_gemm_grid(s, r, ["h100", "b300", "rtx4080"], "fig4-h100-roofline.png",
                  "cuBLAS matrix multiply against measured rooflines: three GPUs, three precisions")


def fig_gemm_all(s, r):
    """The same grid for all eight GPUs of Figure 1 (kept in the repository, not in the note)."""
    fig_gemm_grid(s, r, ["v100-32", "a100", "h100", "h200", "b300", "mi100", "mi210", "rtx4080"],
                  "fig4-all-gpus.png",
                  "Matrix multiply (cuBLAS, rocBLAS on AMD) against measured rooflines: eight GPUs, three precisions")


# ---------------------------------------------------------------- figure 5
def fig_m_sweep(s, r):
    cards = ["v100-32", "a100", "h100", "h200", "b300", "mi100", "mi210", "rtx4080"]
    g = s[(s.family == "gemm") & (s.cache == "warm") & (s.dtype == "fp16") & (s.n == 8192) & (s.k == 8192)]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for t in cards:
        q = g[g.target == t].sort_values("m")
        bw, pk, ridge = roof(r, t, "fp16")
        ax.plot(*roofline_over_m(bw, pk, 2), ls="--", lw=1.2, color=COLOR[t])
        m_ridge = ridge * 8192 / (8192 - 2 * ridge)  # M at which I = M*8192/(2M+8192) reaches the ridge
        ax.plot(q.m, q.tflop_s, "-o", color=COLOR[t], label=f"{NAME[t].replace(' SXM2 32GB', '').replace(' 80GB', '')}:  M ≈ {m_ridge:.0f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("M = rows of X  (N = K = 8192; I ≈ M for small M)")
    ax.set_ylabel("FP16 matmul throughput  (TFLOP/s)")
    ax.set_title("FP16 M sweep against the roofline")
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.6, labelspacing=0.3,
              title="solid: measured, dashed: roofline\nridge point at", title_fontsize=8)
    ax.get_legend()._legend_box.align = "left"
    save(fig, "fig5-m-sweep.png")


# ---------------------------------------------------------------- figure 7
def decode_table(p, r):
    q = p[(p.reference_eligible) & (p.prompt_length == 128)]
    rows = []
    for t in ["rtx4080", "h100", "h200"]:
        bw, _, _ = roof(r, t, "fp16")
        bound = DECODE_BYTES / bw / 1e6  # ms
        for b in sorted(q.batch.unique()):
            m = q[(q.target == t) & (q.batch == b)].gpu_tpot_ms.median()
            rows.append({"target": t, "batch": int(b), "measured_ms": m, "bound_ms": bound, "achieved": bound / m})
    return pd.DataFrame(rows)


def fig_decode(p, r):
    tbl = decode_table(p, r)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for t in ["rtx4080", "h100", "h200"]:
        q = tbl[tbl.target == t]
        ax.plot(q.batch, q.measured_ms, "-o", color=COLOR[t], label=f"{NAME[t]} measured")
        ax.axhline(q.bound_ms.iloc[0], color=COLOR[t], ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 16])
    ax.set_xticklabels(["1", "8", "16"])
    ax.set_ylim(0, 33)
    ax.set_xlabel("Batch size  (prompt 128 tokens, 64 generated)")
    ax.set_ylabel("Time per decode step  (ms)")
    ax.set_title("Qwen2.5-7B bf16 decode on RTX 4080, H100, H200")
    ax.legend(loc="center right", frameon=False, title="dashed: 14.1 GB of weights ÷ measured bandwidth", title_fontsize=9)
    save(fig, "fig7-decode.png")


# ---------------------------------------------------------------- figure 6
MODULE_COLOR = {"q_proj": "#0072B2", "k_proj": "#56B4E9", "v_proj": "#44AA99", "o_proj": "#332288",
                "gate_proj": "#D55E00", "up_proj": "#E69F00", "down_proj": "#009E73"}
CASE_MARKER = {(1, 1024): ("o", "batch 1, prompt 1024"), (8, 1024): ("s", "batch 8, prompt 1024"), (8, 4096): ("^", "batch 8, prompt 4096")}


def fig_operators(o, r):
    """Qwen2.5-7B layer 14 on H100, H200 and RTX 4080: the seven linear layers on each
    GPU's own BF16 roof. x = nominal 2MNK / DRAM bytes measured by Nsight;
    y = nominal FLOPs / profiled kernel time. RTX 4080 has the batch-1 case only."""
    cards = ["h100", "h200", "rtx4080"]
    xx = np.logspace(-1, 4, 400)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharey=True)
    from matplotlib.lines import Line2D
    for j, t in enumerate(cards):
        bw, pk, ridge = roof(r, t, "bf16")
        h = o[(o.target == t) & (o.nominal_flops.notna())]
        for i, phase in enumerate(["prefill", "decode"]):
            ax = axes[i, j]
            ax.loglog(xx, np.minimum(bw * xx / 1000, pk), color="#222222", lw=1.6)
            q = h[h.phase == phase]
            for (b, s), (mk, _) in CASE_MARKER.items():
                qq = q[(q.batch == b) & (q.prompt_length == s)]
                for _, row in qq.iterrows():
                    ax.loglog(row.nominal_flops_per_dram_byte, row.nominal_tflops, mk, color=MODULE_COLOR[row.module], ms=6, alpha=0.9)
            ax.set_xlim(0.1, 1e4)
            ax.set_ylim(0.1, 2000)
            ax.set_title(f"{NAME[t]} {phase}: roofline {bw:.0f} GB/s, {pk:.0f} TFLOP/s", fontsize=11)
            if i == 1:
                ax.set_xlabel("2MNK / GPU-memory bytes measured by Nsight")
    axes[0, 0].set_ylabel("2MNK / kernel time  (TFLOP/s)")
    axes[1, 0].set_ylabel("2MNK / kernel time  (TFLOP/s)")
    mods = [Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in MODULE_COLOR.items()]
    cases = [Line2D([], [], marker=mk, ls="", color="#555555", label=lab) for (mk, lab) in CASE_MARKER.values()]
    axes[0, 0].legend(handles=[Line2D([], [], color="#222222", label="BF16 roofline")] + mods + cases, loc="lower right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Qwen2.5-7B (BF16), layer 14: the seven linear layers on each GPU's BF16 roofline", y=1.0)
    fig.tight_layout()
    save(fig, "fig6-operators.png")


# ---------------------------------------------------------------- figure 8
def fig_rtx4080_fp8(s, r):
    """RTX 4080 M sweep, FP16 vs FP8. In FP8 the 8192x8192 matrix is 64 MiB, the
    size of the L2, so repeated runs read part of it from L2 and the points plotted
    with the hand-computed bytes land above the GPU-memory roof."""
    t = "rtx4080"
    g = s[(s.family == "gemm") & (s.cache == "warm") & (s.target == t) & (s.n == 8192) & (s.k == 8192)]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for dtype, c, label in [("fp16", "#E69F00", "FP16: A is 128 MiB"), ("fp8_e4m3", "#CC79A7", "FP8: A is 64 MiB = L2")]:
        q = g[g.dtype == dtype].sort_values("m")
        bw, pk, _ = roof(r, t, dtype)
        ax.loglog(*roofline_over_m(bw, pk, SIZEOF[dtype]), "--", color=c, lw=1.2)
        ax.loglog(q.m, q.tflop_s, "-o", color=c, label=label)
    ax.set_xlabel("M  (N = K = 8192)")
    ax.set_ylabel("Throughput  (TFLOP/s)")
    ax.set_title("RTX 4080: FP8 points above the roofline")
    ax.legend(loc="lower right", frameon=False, title="solid: measured   dashed: roofline", title_fontsize=9)
    save(fig, "fig8-rtx4080-fp8.png")


# ---------------------------------------------------------------- figure 9
def fig_rtx4080_fp64(s, r):
    """RTX 4080: FP64 vs FP16 M sweep. FP64 stalls from M = 2 to 32 (same time, padded tile)."""
    t = "rtx4080"
    g = s[(s.family == "gemm") & (s.cache == "warm") & (s.target == t) & (s.n == 8192) & (s.k == 8192)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for dtype, c in [("fp16", DTYPE_COLOR["fp16"]), ("fp64", DTYPE_COLOR["fp64"])]:
        q = g[g.dtype == dtype].sort_values("m")
        bw, pk, ridge = roof(r, t, dtype)
        axes[0].loglog(*roofline_over_m(bw, pk, SIZEOF[dtype]), "--", color=c, alpha=0.5)
        axes[0].loglog(q.m, q.tflop_s, "-o", ms=4, color=c, label=f"{dtype.upper()}  roofline {pk:.2f} TFLOP/s")
        axes[1].loglog(q.m, q.median_ms, "-o", ms=4, color=c, label=dtype.upper())
    axes[0].set_xlabel("M  (N = K = 8192)")
    axes[0].set_ylabel("Throughput  (TFLOP/s)")
    axes[0].set_title("RTX 4080: useful 2MNK against the roofline")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[1].set_xlabel("M  (N = K = 8192)")
    axes[1].set_ylabel("Kernel time  (ms)")
    axes[1].set_title("The same points as time: FP64 is flat from M = 2 to 32")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig9-rtx4080-fp64.png")


# ---------------------------------------------------------------- numbers
def numbers(s, r, d, p, w, o):
    pd.set_option("display.width", 200)
    print("\n== FP16 roofs (bandwidth GB/s, peak TF/s, ridge FLOP/B)")
    for t in ["v100-32", "a100", "h100", "h200", "b300", "mi100", "mi210", "rtx4080"]:
        bw, pk, ridge = roof(r, t, "fp16")
        print(f"{NAME[t]:16s} {bw:7.0f} {pk:7.1f} {ridge:6.1f}")
    print("\n== H100, B300 and RTX 4080 roofs per dtype")
    for t in ["h100", "b300", "rtx4080"]:
        for dtype in ["fp16", "tf32", "fp32", "fp64"]:
            bw, pk, ridge = roof(r, t, dtype)
            print(f"{t:8s} {dtype:5s} peak {pk:8.2f}  ridge {ridge:7.2f}")
    print("\n== FMA peaks (max over the sweep)")
    f = s[s.family == "fma"]
    for t in ["v100-32", "a100", "h100", "h200", "b300", "mi100", "mi210", "rtx4080"]:
        for dtype in ["fp32", "fp64"]:
            q = f[(f.target == t) & (f.dtype == dtype)]
            print(f"{t:8s} {dtype}: {q.tflop_s.max():6.2f}")
    print("\n== FMA sweep, RTX 4080 FP64 and B300 FP64 (I, TF/s, GB/s)")
    for t in ["rtx4080", "b300"]:
        q = f[(f.target == t) & (f.dtype == "fp64")].sort_values("ai")
        print(t, [(float(a), round(float(x), 3), round(float(g))) for a, x, g in zip(q.ai, q.tflop_s, q.gb_s)])
    print("\n== copy bandwidth vs working set (GB/s)")
    c = s[(s.family == "copy")]
    print(c.pivot_table(index="footprint_bytes", columns="target", values="gb_s").round(0)[["a100", "h100", "b300", "rtx4080"]])
    print("\n== read_reduce bandwidth, largest working set")
    rr = s[(s.family == "read_reduce")]
    print(rr[rr.footprint_bytes == rr.footprint_bytes.max()][["target", "footprint_bytes", "gb_s"]].round(0).to_string(index=False))
    print("\n== H100 FP16 M-sweep (M, I, TF/s, GB/s)")
    g = s[(s.family == "gemm") & (s.cache == "warm") & (s.dtype == "fp16") & (s.n == 8192) & (s.k == 8192)]
    for t in ["h100", "rtx4080"]:
        q = g[g.target == t].sort_values("m")
        print(t)
        print(q[["m", "ai", "tflop_s", "gb_s"]].round(2).to_string(index=False))
    print("\n== H100 FP16 square GEMM")
    sq = s[(s.family == "gemm") & (s.cache == "warm") & (s.dtype == "fp16") & (s.target == "h100") & (s.m == s.n)]
    print(sq[["m", "ai", "tflop_s"]].round(1).to_string(index=False))
    print("\n== decode, prompt 128")
    print(f"weights read per decode step: {DECODE_BYTES/1e9:.2f} GB  (total {PARAM_BYTES/1e9:.2f} GB, embedding {EMBED_BYTES/1e9:.2f} GB)")
    print(decode_table(p, r).round(3).to_string(index=False))
    print("\n== H200 / H100 decode ratio at batch 1, prompt 128")
    q = p[(p.reference_eligible) & (p.prompt_length == 128) & (p.batch == 1)]
    print({t: round(float(q[q.target == t].gpu_tpot_ms.median()), 2) for t in ["h100", "h200", "rtx4080"]})
    print("\n== H100 layer-14 linear operators (Nsight DRAM bytes): I, nominal TFLOP/s, fraction of roof")
    bw, pk, _ = roof(r, "h100", "bf16")
    h = o[(o.target == "h100") & (o.nominal_flops.notna())].copy()
    h["roof"] = np.minimum(bw * h.nominal_flops_per_dram_byte / 1000, pk)
    h["fraction"] = h.nominal_tflops / h.roof
    print(h[["phase", "batch", "prompt_length", "module", "nominal_flops_per_dram_byte", "nominal_tflops", "fraction"]].round(2).sort_values(["phase", "batch", "prompt_length", "module"]).to_string(index=False))
    print("\n== RTX 4080 FP64 M-sweep times (ms)")
    q = s[(s.family == "gemm") & (s.cache == "warm") & (s.target == "rtx4080") & (s.dtype == "fp64") & (s.n == 8192) & (s.k == 8192)].sort_values("m")
    print(q[["m", "median_ms", "tflop_s"]].round(3).to_string(index=False))
    numbers_derived(s, r, p, o)
    print("\n== warnings")
    print(w.warning.value_counts().to_string())
    print(w[w.warning.str.startswith("Above")][["target", "case", "value"]].to_string(index=False))
    print("\n== devices")
    print(d.to_string(index=False))


def numbers_derived(s, r, p, o):
    """Ratios and individual points quoted in the text that the tables above do not show."""
    fp16 = {t: roof(r, t, "fp16") for t in NAME}
    print("\n== FP16 roof ratios (Section 3)")
    print(f"F B300/V100 {fp16['b300'][1] / fp16['v100-32'][1]:.2f}x, B B300/V100 {fp16['b300'][0] / fp16['v100-32'][0]:.2f}x")
    print(f"B H200/H100 {fp16['h200'][0] / fp16['h100'][0]:.3f}; H100 slope meets H200 flat part at I = {1000 * fp16['h200'][1] / fp16['h100'][0]:.1f}")
    print(f"B MI210/MI100 {fp16['mi210'][0] / fp16['mi100'][0]:.3f}; B RTX 4080/V100 {fp16['rtx4080'][0] / fp16['v100-32'][0]:.3f}")
    print(f"F RTX 4080/A100 {fp16['rtx4080'][1] / fp16['a100'][1]:.3f}")
    print(f"H100 BF16 peak {roof(r, 'h100', 'bf16')[1]:.1f}, ridge {roof(r, 'h100', 'bf16')[2]:.1f}; H200 BF16 peak {roof(r, 'h200', 'bf16')[1]:.1f}")
    f = s[s.family == "fma"]
    fma = {(t, dt): f[(f.target == t) & (f.dtype == dt)].tflop_s.max() for t in NAME for dt in ["fp32", "fp64"]}
    print("\n== FMA sweep (Section 4): FP64/FP32 peak ratio; GB/s on the slope (I at most half the ridge point) as % of B")
    for t in NAME:
        bw = fp16[t][0]
        cells = []
        for dt in ["fp32", "fp64"]:
            q = f[(f.target == t) & (f.dtype == dt)].sort_values("ai")
            slope = q[q.ai <= 0.5 * 1000 * fma[(t, dt)] / bw]  # at most half the ridge point: clearly on the slope
            cells.append(f"{dt} {100 * slope.gb_s.min() / bw:.1f}-{100 * slope.gb_s.max() / bw:.1f}%" if len(slope) else f"{dt} no point")
        print(f"{t:8s} fp64/fp32 {fma[(t, 'fp64')] / fma[(t, 'fp32')]:.3f}  " + "  ".join(cells))
    print(f"FP32 FMA RTX 4080/A100 {fma[('rtx4080', 'fp32')] / fma[('a100', 'fp32')]:.2f}x; H100 FP32 FMA ridge {1000 * fma[('h100', 'fp32')] / fp16['h100'][0]:.1f}, FP64 FMA ridge {1000 * fma[('h100', 'fp64')] / fp16['h100'][0]:.1f}")
    g = s[(s.family == "gemm") & (s.cache == "warm")]
    print("\n== individual matrix multiply times (Sections 5 and 7), ms")
    sq = g[(g.target == "h100") & (g.dtype == "fp16") & (g.m == g.n) & (g.n == g.k) & (g.m <= 512)]
    print("H100 FP16 square n = 256, 512:", [round(float(x) * 1000, 2) for x in sq.sort_values("m").median_ms], "us")
    sw = g[(g.n == 8192) & (g.k == 8192)]
    for t, dt, ms in [("h100", "fp16", [1, 2, 4, 8]), ("b300", "fp16", [1, 2, 4, 8]), ("b300", "fp32", [1, 2, 4, 8, 16, 32, 64])]:
        q = sw[(sw.target == t) & (sw.dtype == dt) & sw.m.isin(ms)].sort_values("m")
        print(f"{t} {dt} M = {ms}:", [round(float(x), 4) for x in q.median_ms])
    print("\n== RTX 4080 FP8 M sweep against the FP8 roofline (Section 7)")
    bw, pk, _ = roof(r, "rtx4080", "fp8_e4m3")
    q = sw[(sw.target == "rtx4080") & (sw.dtype == "fp8_e4m3")].sort_values("m").copy()
    q["roof"] = np.minimum(bw * q.ai / 1000, pk)
    q["above_roof"] = q.tflop_s / q.roof
    print(q[["m", "ai", "tflop_s", "gb_s", "above_roof"]].round(2).to_string(index=False))
    print("\n== decode, reference runs, median gpu_tpot_ms by prompt length and batch (Section 6)")
    tb = p[p.reference_eligible].pivot_table(index=["target", "prompt_length"], columns="batch", values="gpu_tpot_ms", aggfunc="median")
    tb["b16/b1"] = tb[16] / tb[1]
    print(tb.round(3).to_string())
    dt = decode_table(p, r)
    b1 = dt[dt.batch == 1].set_index("target")
    print("measured / bound at batch 1:", {t: round(float(x), 2) for t, x in (b1.measured_ms / b1.bound_ms).items()})
    print("measured - bound at batch 1 (ms):", {t: round(float(x), 2) for t, x in (b1.measured_ms - b1.bound_ms).items()})
    w_flop = DECODE_BYTES  # 2 FLOP per BF16 weight of 2 bytes
    wf = w_flop / (fp16["h100"][1] * 1e12) * 1e3
    print(f"batch 1: {DECODE_BYTES / 2 / 1e9:.2f}e9 weights, W = {w_flop / 1e9:.2f}e9 FLOP, H100 W/F = {wf:.4f} ms = 1/{b1.bound_ms['h100'] / wf:.0f} of the bound")
    print("\n== layer-14 linear operators, fraction of each GPU's BF16 roof (Section 6)")
    for t in ["h100", "h200", "rtx4080"]:
        bw, pk, _ = roof(r, t, "bf16")
        h = o[(o.target == t) & (o.nominal_flops.notna())].copy()
        h["fraction"] = h.nominal_tflops / np.minimum(bw * h.nominal_flops_per_dram_byte / 1000, pk)
        print(t, h.pivot_table(index="module", columns=["phase", "batch", "prompt_length"], values="fraction").round(3).to_string(), sep="\n")
        pre = h[h.phase == "prefill"]
        rest = pre[~((pre.batch == 1) & pre.module.isin(["k_proj", "v_proj"]))]
        print(f"{t} prefill I {pre.nominal_flops_per_dram_byte.min():.1f}-{pre.nominal_flops_per_dram_byte.max():.1f}; "
              f"TFLOP/s without k, v at batch 1: {rest.nominal_tflops.min():.1f}-{rest.nominal_tflops.max():.1f}")
    print("kernels per decoder layer (decode, Nsight):", o[o.phase == "decode"].groupby(["target", "batch", "prompt_length"]).kernel_count.sum().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", action="store_true", help="print numbers only")
    a = ap.parse_args()
    s, r, d, p, w, o = load()
    if not a.numbers:
        fig_roofline_fp16(r)
        fig_bandwidth_working_set(s, d, r)
        fig_fma_sweep(s, r)
        fig_h100_roofline(s, r)
        fig_gemm_all(s, r)
        fig_m_sweep(s, r)
        fig_decode(p, r)
        fig_operators(o, r)
        fig_rtx4080_fp8(s, r)
        fig_rtx4080_fp64(s, r)
    numbers(s, r, d, p, w, o)


if __name__ == "__main__":
    main()
