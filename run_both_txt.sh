#!/usr/bin/env bash
# Run spectrum pipeline on the two 77K sample cubes in parallel (WSL/Docker).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${ROOT}/src/au_region_analysis.py" "${ROOT}/40_532_250uW_1s_sample_77K_.txt.txt" --pixel-pngs &
pid1=$!
python "${ROOT}/src/au_region_analysis.py" "${ROOT}/40_SC_20per_1s_sample_77K_Au_region.txt.txt" --pixel-pngs &
pid2=$!

fail=0
wait "$pid1" || fail=1
wait "$pid2" || fail=1
exit "$fail"
