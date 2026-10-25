#!/usr/bin/env bash
# Sync sources + reindex every Trawl pool, smallest first.
# Designed for overnight runs. Logs to the logs folder in Mullet's state dir (`mullet paths state`).
#
# Usage:
#   ./scripts/overnight_reindex_all.sh              # run in foreground
#   nohup ./scripts/overnight_reindex_all.sh &      # run detached
#
# Flags:
#   --graph        also rebuild knowledge graph (slower, uses LLM tokens)
#   --rebuild      full rebuild instead of incremental
#   --dry-run      print what would run, don't execute
#   --pools p1,p2  comma-separated pool list, overrides auto-discovery
#
# Pool discovery (default): every dir under `mullet paths pools` containing a
# config.yaml. Ordered smallest → largest by `du -sb` so the run yields fast
# wins early. Pools whose machine_name starts with `test_init_` are skipped.

set -u

GRAPH=""
REBUILD=""
DRY_RUN=""
POOLS_OVERRIDE=""
while (( $# )); do
  case "$1" in
    --graph)    GRAPH="--graph"; shift ;;
    --rebuild)  REBUILD="--rebuild"; shift ;;
    --dry-run)  DRY_RUN="1"; shift ;;
    --pools=*)  POOLS_OVERRIDE="${1#--pools=}"; shift ;;
    --pools)    POOLS_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

command -v mullet >/dev/null || { echo "mullet is not on PATH" >&2; exit 2; }
POOLS_DIR="$(mullet paths pools)"

if [[ -n "$POOLS_OVERRIDE" ]]; then
  # Comma-separated explicit list, in order given.
  IFS=',' read -r -a POOLS <<< "$POOLS_OVERRIDE"
else
  # Auto-discover. List dirs under POOLS_DIR that contain a config.yaml,
  # skip test_init_*, sort by total size (smallest first) so the run yields
  # fast wins early. POOLS_DIR is usually a symlink — glob handles that.
  if [[ ! -d "$POOLS_DIR" ]]; then
    echo "Pools dir not found: $POOLS_DIR" >&2
    exit 2
  fi
  POOLS=()
  declare -A POOL_SIZES=()
  for d in "$POOLS_DIR"/*/; do
    name=$(basename "$d")
    case "$name" in test_init_*) continue ;; esac
    [[ -f "$d/config.yaml" ]] || continue
    size=$(du -sb "$d" 2>/dev/null | awk '{print $1}')
    POOL_SIZES["$name"]="${size:-0}"
  done
  if (( ${#POOL_SIZES[@]} == 0 )); then
    echo "No pools with config.yaml found under $POOLS_DIR" >&2
    exit 2
  fi
  # Sort pool names by their size ascending.
  while IFS= read -r line; do POOLS+=("${line#* }"); done < <(
    for n in "${!POOL_SIZES[@]}"; do
      printf '%012d %s\n' "${POOL_SIZES[$n]}" "$n"
    done | sort -n
  )
fi

LOG_DIR="$(mullet paths state)/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/overnight-reindex-$(date +%Y%m%d-%H%M).log"

run() {
  if [[ -n "$DRY_RUN" ]]; then
    echo "DRY-RUN: $*"
  else
    "$@"
  fi
}

{
  echo "=== Overnight reindex starting $(date -Iseconds) ==="
  echo "Log: $LOG"
  echo "Pools: ${#POOLS[@]}"
  echo "Options: graph=${GRAPH:-no} rebuild=${REBUILD:-no} dry_run=${DRY_RUN:-no}"
  echo

  FAILED=()
  START_ALL=$SECONDS

  for p in "${POOLS[@]}"; do
    START=$SECONDS
    echo "--- $(date -Iseconds) START $p ---"
    if run mullet reindex --pool "$p" --sync-sources $GRAPH $REBUILD; then
      DUR=$((SECONDS - START))
      echo "--- $(date -Iseconds) DONE  $p (${DUR}s) ---"
    else
      DUR=$((SECONDS - START))
      echo "!!! $(date -Iseconds) FAILED $p (${DUR}s) — continuing"
      FAILED+=("$p")
    fi
    echo
  done

  TOTAL=$((SECONDS - START_ALL))
  echo "=== Overnight reindex finished $(date -Iseconds) — total ${TOTAL}s ==="
  if (( ${#FAILED[@]} )); then
    echo "FAILED pools (${#FAILED[@]}): ${FAILED[*]}"
    exit 1
  else
    echo "All ${#POOLS[@]} pools reindexed cleanly."
  fi
} 2>&1 | tee -a "$LOG"
