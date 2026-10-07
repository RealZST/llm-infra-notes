#!/bin/bash
# Run the decode harness (run.py) in one mode on the single visible NVIDIA GPU.
#
#   bash run_decode.sh MODE RUN_DIR MANIFEST
#
# MODE      performance | trace | counters | pilot
# RUN_DIR   new output directory (refused if it already holds a run)
# MANIFEST  application_model.json written by prepare_model.py
#
# performance: the 3 x 3 batch/prompt grid, 2 warm-ups + 7 timed sequences each
#   (timings.jsonl, per-case inputs/outputs/validation JSON, status.json).
# trace: one PyTorch profiler trace per counter case in RUN_DIR/cases/b*_s*/.
# counters: Nsight Compute on decoder layer 14 per counter case and phase, via
#   ncu_counters.sh, in RUN_DIR/cases/b*_s*_<phase>/.
# pilot: B1/S128 check plus one prefill counter case; not used for any table.
#
# All modes also write gpu_identity.json, nvidia-smi -q before and after
# (device.txt, device_final.txt) and telemetry.csv (nvidia-smi once per second).
# export_raw.py turns a performance or counters RUN_DIR into the data/raw/ layout.
#
# Environment: PYTHON (default python3) is the interpreter with torch and
# transformers; nvcc and nvidia-smi on PATH; ncu on PATH for counters and pilot.
# Expose exactly one GPU, for example CUDA_VISIBLE_DEVICES=0.
#
# This is the scheduler batch script of the published runs with the scheduler
# directives, allocation records, module loads and the node-local model copy
# removed; the run.py calls and the telemetry query are unchanged.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?mode}; RUN=${2:?run dir}; MANIFEST=${3:?model manifest}
PYTHON=${PYTHON:-python3}
[[ ! -e "$RUN/start.txt" ]] || { echo 'Refusing to overwrite an existing run.' >&2; exit 2; }
mkdir -p "$RUN/build"
RUN=$(cd "$RUN" && pwd)
cp "$HERE/run.py" "$HERE/hardware_probe.cu" "$HERE/ncu_counters.sh" "$RUN/"
cp "$0" "$RUN/run_decode.sh"
cp "$HERE/application.json" "$RUN/application.json"
cp "$MANIFEST" "$RUN/application_model.json"
sha256sum "$RUN/run.py" "$RUN/run_decode.sh" "$RUN/ncu_counters.sh" "$RUN/hardware_probe.cu" "$RUN/application.json" "$RUN/application_model.json" > "$RUN/source.sha256"
echo "$MODE" > "$RUN/mode.txt"
date -Is > "$RUN/start.txt"
OWN_LOCAL=""
if [[ -z ${APP_LOCAL_DIR:-} ]]; then APP_LOCAL_DIR=$(mktemp -d); OWN_LOCAL=1; fi
export APP_LOCAL_DIR
SAMPLER=""
finish() {
 local rc=$?
 if [[ -n "$SAMPLER" ]]; then
  kill "$SAMPLER" 2>/dev/null || true
  wait "$SAMPLER" 2>/dev/null || true
  nvidia-smi -i "$GPU_UUID" -q > "$RUN/device_final.txt" || true
 fi
 date -Is > "$RUN/end.txt";echo "$rc" > "$RUN/exit_code.txt"
 [[ -z "$OWN_LOCAL" ]] || rm -rf "$APP_LOCAL_DIR"
}
trap finish EXIT
nvcc --version > "$RUN/compiler.txt"
nvcc -O2 "$RUN/hardware_probe.cu" -o "$RUN/build/identity" > "$RUN/compile.log" 2>&1
"$RUN/build/identity" > "$RUN/gpu_identity.json"
GPU_UUID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["uuid"])' "$RUN/gpu_identity.json")
nvidia-smi -i "$GPU_UUID" -q > "$RUN/device.txt"
nvidia-smi -i "$GPU_UUID" --query-gpu=timestamp,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown --format=csv -l 1 > "$RUN/telemetry.csv" 2> "$RUN/telemetry.stderr" &
SAMPLER=$!
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$APP_LOCAL_DIR/pycache"
export TRITON_CACHE_DIR="$APP_LOCAL_DIR/triton"
export PYTHON
"$PYTHON" -m pip freeze > "$RUN/python_packages.txt"
if [[ "$MODE" == counters || "$MODE" == pilot ]]; then
 ncu --version > "$RUN/profiler_version.txt"
 ncu --list-sections > "$RUN/profiler_sections.txt"
fi
child() {
 local DEST=$1
 mkdir -p "$DEST"
 cp "$RUN/application.json" "$RUN/application_model.json" "$RUN/gpu_identity.json" "$DEST/"
}
case "$MODE" in
 pilot)
  "$PYTHON" -u "$RUN/run.py" --run "$RUN" --mode pilot > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
  bash "$HERE/ncu_counters.sh" "$RUN" 1 128 prefill
  ;;
 performance)
  "$PYTHON" -u "$RUN/run.py" --run "$RUN" --mode performance > "$RUN/stdout.txt" 2> "$RUN/stderr.txt"
  ;;
 trace)
  for PAIR in '1 1024' '8 1024' '8 4096'; do
   read -r B S <<< "$PAIR"
   DEST="$RUN/cases/b${B}_s${S}"
   child "$DEST"
   RC=0
   "$PYTHON" -u "$RUN/run.py" --run "$DEST" --mode trace --batch "$B" --length "$S" > "$DEST/stdout.txt" 2> "$DEST/stderr.txt" || RC=$?
   if [[ "$RC" == 3 && -f "$DEST/oom.json" ]]; then touch "$DEST/OOM"; continue; fi
   [[ "$RC" == 0 ]] || exit "$RC"
   touch "$DEST/COMPLETE"
  done
  ;;
 counters)
  for PAIR in '1 1024' '8 1024' '8 4096'; do
   read -r B S <<< "$PAIR"
   bash "$HERE/ncu_counters.sh" "$RUN" "$B" "$S" prefill
   bash "$HERE/ncu_counters.sh" "$RUN" "$B" "$S" decode
  done
  ;;
 *) echo "Unknown mode: $MODE" >&2; exit 2;;
esac
printf 'Decode harness completed: %s\n' "$MODE" > "$RUN/COMPLETE"
