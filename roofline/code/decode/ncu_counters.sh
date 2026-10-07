#!/bin/bash
# Nsight Compute counters for decoder layer 14 of one (batch, prompt length, phase).
#
#   bash ncu_counters.sh RUN_DIR BATCH PROMPT_LENGTH PHASE
#
# RUN_DIR   a directory prepared by run_decode.sh (run.py, application.json,
#           application_model.json, gpu_identity.json); run_decode.sh counters
#           calls this script for (1, 1024), (8, 1024), (8, 4096) x prefill, decode.
# PHASE     prefill | decode
#
# Writes RUN_DIR/cases/b<B>_s<S>_<PHASE>/:
#   command.json        the capture settings below
#   counters.csv        ncu --csv --page raw output of the profiled run (kernel names)
#   nvtx_counters.csv   the same report exported with NVTX-renamed kernels, which
#                       carry the innermost module range (q_proj/..., input_norm/...)
#   counters.ncu-rep.gz the report; report_storage.json has its size and SHA256
#   counter_case.json   run.py's record: captured shapes and nominal linear FLOPs
# or, if the case does not fit in memory, oom.json and an OOM marker.
#
# Only kernels inside the NVTX range "<PHASE>_layer14" are profiled
# (--nvtx-include), with no cache flushing and no clock locking between kernels
# (--cache-control none --clock-control none). run.py calls cudaProfilerStart/Stop
# around the single captured forward pass (--profile-from-start off).
#
# Environment: PYTHON (default python3), NCU (default ncu), APP_LOCAL_DIR
# (directory for the uncompressed report; default a temporary directory).
set -euo pipefail
RUN=$(cd "${1:?run dir}" && pwd); B=${2:?batch}; S=${3:?prompt length}; PHASE=${4:?phase}
PYTHON=${PYTHON:-python3}; NCU=${NCU:-ncu}
LOCAL=${APP_LOCAL_DIR:-}
if [[ -z "$LOCAL" ]]; then LOCAL=$(mktemp -d); trap 'rm -rf "$LOCAL"' EXIT; fi
DEST="$RUN/cases/b${B}_s${S}_${PHASE}"
REPORT="$LOCAL/b${B}_s${S}_${PHASE}.ncu-rep"
mkdir -p "$DEST"
cp "$RUN/application.json" "$RUN/application_model.json" "$RUN/gpu_identity.json" "$DEST/"
[[ -f "$RUN/profiler_sections.txt" ]] || "$NCU" --list-sections > "$RUN/profiler_sections.txt"
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum"
EXTRA=()
# Adds the Tensor-pipe operation counts (sm__ops_path_tensor_*) where this ncu version has the section.
if grep -q 'SpeedOfLight_HierarchicalTensorRooflineChart' "$RUN/profiler_sections.txt"; then
 EXTRA+=(--section SpeedOfLight_HierarchicalTensorRooflineChart)
fi
python3 - "$DEST/command.json" "$B" "$S" "$PHASE" "$METRICS" ${EXTRA[@]+"${EXTRA[@]}"} <<'PY'
import json,sys
json.dump({'batch':int(sys.argv[2]),'prompt_length':int(sys.argv[3]),'phase':sys.argv[4],
'metrics':sys.argv[5].split(','),'extra_section_args':sys.argv[6:],
'nvtx_include':sys.argv[4]+'_layer14/','cache_control':'none','clock_control':'none'},open(sys.argv[1],'w'),indent=2)
PY
rm -f "$DEST/oom.json"
set +e
"$NCU" --profile-from-start off --cache-control none --clock-control none --nvtx \
 --nvtx-include "${PHASE}_layer14/" --metrics "$METRICS" ${EXTRA[@]+"${EXTRA[@]}"} \
 --export "$REPORT" --csv --page raw --print-units base \
 "$PYTHON" -u "$RUN/run.py" --run "$DEST" --mode counters --batch "$B" --length "$S" --phase "$PHASE" \
 > "$DEST/counters.csv" 2> "$DEST/counters.stderr"
RC=$?
set -e
echo "$RC" > "$DEST/exit_code.txt"
# Exceeding device memory is recorded per case (oom.json), not retried or skipped silently.
if [[ "$RC" != 0 && -f "$DEST/oom.json" ]]; then touch "$DEST/OOM"; exit 0; fi
[[ "$RC" == 0 ]] || exit "$RC"
# Export step: the same report as CSV with each kernel renamed to its innermost NVTX range.
"$NCU" --import "$REPORT" --csv --page raw --print-units base --print-nvtx-rename kernel > "$DEST/nvtx_counters.csv" 2> "$DEST/nvtx_export.stderr"
python3 - "$REPORT" "$DEST" <<'PY'
import gzip,hashlib,json,pathlib,shutil,sys
src,dest=map(pathlib.Path,sys.argv[1:])
with src.open('rb') as f,gzip.open(dest/'counters.ncu-rep.gz','wb',compresslevel=6) as out:shutil.copyfileobj(f,out)
h=hashlib.sha256()
with src.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
(dest/'report_storage.json').write_text(json.dumps({'file':'counters.ncu-rep.gz','decompressed_bytes':src.stat().st_size,'decompressed_sha256':h.hexdigest()},indent=2)+'\n')
PY
touch "$DEST/COMPLETE"
