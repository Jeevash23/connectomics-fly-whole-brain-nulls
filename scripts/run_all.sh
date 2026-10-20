#!/usr/bin/env bash
# Run the whole benchmark for one wiring version: the measured wiring, then draws 0..DRAWS-1 of every null family,
# then the analysis. Finished runs are skipped and interrupted runs resume, so the script can be restarted.
#
# Usage, from the repository root:  scripts/run_all.sh v783 [DRAWS] [JOBS]
#   DRAWS  draws per null family (default 100, as published)
#   JOBS   runs at once (default 1). Each run uses 10 Numba threads, as in the published runs, and about 4 GB of
#          memory, so JOBS=4 needs about 16 GB of memory and 40 threads.
# Families: N1-N4 on v783; N1 and N2 on v630 (N3 and N4 need v783's cell types and super classes).
# The analysis rewrites results/<version>/analysis.json; the published file is first kept as analysis.published.json.
# If any run fails, the analysis is not run; run the script again, and it skips the runs that finished.
set -euo pipefail
VERSION="${1:?usage: scripts/run_all.sh v783|v630 [DRAWS] [JOBS]}"
DRAWS="${2:-100}"
JOBS="${3:-1}"
case "$VERSION" in
  v783) FAMILIES="N1 N2 N3 N4" ;;
  v630) FAMILIES="N1 N2" ;;
  *) echo "version must be v783 or v630" >&2; exit 2 ;;
esac
PY="${PYTHON:-python}"
cd "$(dirname "$0")/../code"
mkdir -p ../logs
export NUMBA_NUM_THREADS=10
FAILURES="../logs/${VERSION}_failures.txt"
: > "$FAILURES"

"$PY" run_wiring.py --version "$VERSION" --wiring M > "../logs/${VERSION}_M.log" 2>&1 || echo "failed: $VERSION M" >> "$FAILURES"
for draw in $(seq 0 $((DRAWS - 1))); do
  for family in $FAMILIES; do
    echo "$family $draw"
  done
done | xargs -P "$JOBS" -L 1 sh -c \
  '"$0" run_wiring.py --version "$1" --wiring "$2" --draw "$3" > "../logs/$1_$2_$3.log" 2>&1 || echo "failed: $1 $2 $3"' \
  "$PY" "$VERSION" >> "$FAILURES"

if [ -s "$FAILURES" ]; then
  echo "$(wc -l < "$FAILURES") run(s) failed (listed in logs/${VERSION}_failures.txt); the analysis was not run." >&2
  echo "Run this script again: finished runs are skipped." >&2
  exit 1
fi
if [ -f "../results/$VERSION/analysis.json" ] && [ ! -f "../results/$VERSION/analysis.published.json" ]; then
  cp "../results/$VERSION/analysis.json" "../results/$VERSION/analysis.published.json"
fi
"$PY" analyse.py "$VERSION"
