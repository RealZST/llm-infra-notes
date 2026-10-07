#!/usr/bin/env python3
"""Convert run directories of run_decode.sh into the layout of data/raw/.

    python3 export_raw.py decode   OUT_DIR RUN_DIR [RUN_DIR ...]
    python3 export_raw.py counters OUT_DIR RUN_DIR [RUN_DIR ...]

Each RUN_DIR (in the order given) becomes OUT_DIR/run-1, run-2, ...

decode (mode performance): run.json, sequences.csv and steps.csv from
timings.jsonl, metadata.json, status.json and gpu_identity.json; telemetry.csv
with timestamps turned into seconds since the first poll and the GPU UUID
dropped. reference_eligible is false when telemetry.csv, or an `nvidia-smi -q`
capture saved as live_device_condition.txt during the run, shows an active SW or
HW thermal slowdown.

counters (mode counters): run.json, kernels.csv (one row per profiled kernel,
module = innermost NVTX range from nvtx_counters.csv) and linear_operators.csv
from the per-case directories.

Host names, UUIDs, PCI bus IDs, paths and absolute timestamps are not copied.
"""
import csv, io, json, pathlib, re, sys
from datetime import datetime

DEVICE_KEYS = ["name", "memory_bytes", "sm_count", "major", "minor"]
SOFTWARE_KEYS = ["model_id", "revision", "dtype", "torch", "torch_cuda", "transformers",
                 "attention_implementation", "cache", "logits_to_keep", "seed"]
UNIT = re.compile(r"^(-?[0-9.]+) (W|MHz|%)$")
THERMAL_QUERY = re.compile(r"(?:SW|HW) Thermal Slowdown\s*:\s*Active\b")
BASE_METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum",
                "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum"]
BASE_UNITS = ["byte", "byte", "ns", "sector", "sector"]
SEQUENCE_COLUMNS = ["batch", "prompt_length", "repeat", "generated_tokens", "gpu_prefill_ms", "wall_prefill_ms",
                    "wall_sequence_ms", "final_cache_length", "peak_allocated_bytes", "peak_reserved_bytes",
                    "input_sha256", "generated_sha256"]


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def device(run):
    identity = read_json(run / "gpu_identity.json")
    return {k: identity[k] for k in DEVICE_KEYS}


def software(metadata):
    return {k: metadata[k] for k in SOFTWARE_KEYS if k in metadata}


def parse_time(text):
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(text)


def export_telemetry(src, dst):
    rows = list(csv.reader(src.open()))
    header = [h.strip() for h in rows[0]]
    body = [[c.strip() for c in r] for r in rows[1:] if r]
    time_index = header.index("timestamp")
    keep = [i for i, h in enumerate(header) if h not in ("timestamp", "uuid")]
    t0 = parse_time(body[0][time_index])
    out = []
    for r in body:
        values = [UNIT.match(r[i]).group(1) if UNIT.match(r[i]) else r[i] for i in keep]
        out.append([f"{(parse_time(r[time_index]) - t0).total_seconds():.3f}"] + values)
    write_csv(dst, ["t_s"] + [header[i] for i in keep], out)


def thermal_slowdown(run):
    """Same test as the analysis of the published runs."""
    found = False
    query = run / "live_device_condition.txt"
    if query.exists():
        found = bool(THERMAL_QUERY.search(query.read_text()))
    telemetry = run / "telemetry.csv"
    if telemetry.exists():
        for row in csv.DictReader(telemetry.open()):
            found |= any(str(v).strip().lower() in ("active", "true", "1") for k, v in row.items()
                         if k and k.strip().endswith(("sw_thermal_slowdown", "hw_thermal_slowdown")))
    return found


def export_decode(run, out):
    config = read_json(run / "application.json")
    status = read_json(run / "status.json")
    metadata = read_json(run / "metadata.json")
    assert metadata["mode"] == "performance", f"{run} is not a performance run"
    thermal = thermal_slowdown(run)
    info = {"mode": "performance", "device": device(run), "software": software(metadata),
            "timing_protocol": metadata.get("timing_protocol"),
            "generated_tokens": config["generated_tokens"], "warmup_sequences": config["warmup_sequences"],
            "repeats": config["repeats"], "batches": config["batches"], "prompt_lengths": config["prompt_lengths"],
            "oom_cases": status.get("oom_cases", []),
            "reference_eligible": not thermal,
            "device_condition": ("Observed active thermal slowdown; diagnostic only" if thermal
                                 else "No active thermal slowdown observed in available records")}
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")
    rows = [json.loads(line) for line in (run / "timings.jsonl").read_text().splitlines() if line.strip()]
    write_csv(out / "sequences.csv", SEQUENCE_COLUMNS, [[r[c] for c in SEQUENCE_COLUMNS] for r in rows])
    steps = []
    for r in rows:
        assert len(r["gpu_decode_ms"]) == len(r["wall_decode_ms"])
        for step, (gpu, wall) in enumerate(zip(r["gpu_decode_ms"], r["wall_decode_ms"]), 1):
            steps.append([r["batch"], r["prompt_length"], r["repeat"], step, gpu, wall])
    write_csv(out / "steps.csv", ["batch", "prompt_length", "repeat", "step", "gpu_decode_ms", "wall_decode_ms"], steps)
    if (run / "telemetry.csv").exists():
        export_telemetry(run / "telemetry.csv", out / "telemetry.csv")


def nsight_table(path):
    """Kernel rows of an `ncu --csv --page raw --print-units base` table, keyed by ID."""
    text = path.read_text()
    rows = list(csv.DictReader(io.StringIO(text[text.find('"ID",'):])))
    units = rows[0]
    assert not units["ID"] and [units[k] for k in BASE_METRICS] == BASE_UNITS, path
    return {int(r["ID"]): r for r in rows[1:] if r["ID"]}, units


def number(text):
    v = float(str(text).replace(",", ""))
    return int(v) if v.is_integer() else v


def export_counters(run, out):
    config = read_json(run / "application.json")
    version = re.search(r"Version (\S+)", (run / "profiler_version.txt").read_text()).group(1)
    kernels, linears, oom, measured, tensor_columns, metadata, command = [], [], [], [], None, None, None
    for case in sorted((run / "cases").iterdir()):
        b, s, phase = case.name.split("_")
        b, s = int(b[1:]), int(s[1:])
        if (case / "OOM").exists():
            oom.append([b, s, phase])
            continue
        metadata = read_json(case / "metadata.json")
        record = read_json(case / "counter_case.json")
        command = read_json(case / "command.json")
        assert (record["batch"], record["prompt_length"], record["phase"]) == (b, s, phase)
        original, units = nsight_table(case / "counters.csv")
        renamed, _ = nsight_table(case / "nvtx_counters.csv")
        assert original.keys() == renamed.keys()
        columns = [k for k in units if k.endswith(".sum") and "ops_path_tensor" in k]
        tensor_columns = tensor_columns or columns
        assert columns == tensor_columns
        measured.append([b, s, phase])
        for kid in sorted(original):
            r, named = original[kid], renamed[kid]
            assert all(r[k] == named[k] for k in BASE_METRICS)
            kernels.append({"batch": b, "prompt_length": s, "phase": phase, "layer": record["layer"],
                            "query_tokens": record["input_query_tokens"], "kv_tokens": record["attention_kv_tokens"],
                            "kernel_id": kid, "module": named["Kernel Name"].split("/", 1)[0],
                            "kernel": r["Kernel Name"], "block_size": r["Block Size"], "grid_size": r["Grid Size"],
                            **{k: number(r[k]) for k in BASE_METRICS + columns}})
        for o in record["linear_operators"]:
            linears.append({"batch": b, "prompt_length": s, "phase": phase, "layer": o["layer"], "module": o["module"],
                            "input_shape": "x".join(map(str, o["input_shape"])), "in_features": o["in_features"],
                            "out_features": o["out_features"], "nominal_flops": o["nominal_flops"]})
    info = {"mode": "counters", "device": device(run), "software": software(metadata), "nsight_compute": version,
            "layer": config["counter_layer"], "counter_cases": config["counter_cases"],
            "metrics": BASE_METRICS, "units": dict(zip(BASE_METRICS, BASE_UNITS)),
            "tensor_metric_columns": tensor_columns, "nvtx_include": f"<phase>_layer{config['counter_layer']}/",
            "cache_control": command["cache_control"], "clock_control": command["clock_control"],
            "measured_cases": measured, "oom_cases": oom}
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")
    for name, rows in [("kernels.csv", kernels), ("linear_operators.csv", linears)]:
        write_csv(out / name, list(rows[0]), [list(r.values()) for r in rows])


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("decode", "counters"):
        sys.exit(__doc__)
    kind, out_dir, runs = sys.argv[1], pathlib.Path(sys.argv[2]), [pathlib.Path(p) for p in sys.argv[3:]]
    for i, run in enumerate(runs, 1):
        out = out_dir / f"run-{i}"
        out.mkdir(parents=True, exist_ok=True)
        (export_decode if kind == "decode" else export_counters)(run, out)
        print(run, "->", out)


if __name__ == "__main__":
    main()
