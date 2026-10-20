#!/bin/bash
# Resubmit the cells of an array that are neither done nor queued. Safe to run as often as
# you like, from the login node.
#
#   resubmit_incomplete.sh [--dry-run] SBATCH_SCRIPT MANIFEST DONE_MARKER JOB_NAME [sbatch options...]
#
# MANIFEST has one cell directory per line. Line t+1 is array task t, the same mapping that
# assets/array.sbatch uses. A cell is done when <cell dir>/DONE_MARKER exists. Extra
# options go to sbatch, for example --time=6:00:00.
#
# Prints one line: DONE, IN QUEUE, WOULD RUN, or SUBMITTED. If any task with JOB_NAME is
# still in the queue, it submits nothing, so it never queues a cell twice.
set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi
if [ $# -lt 4 ]; then
    sed -n '2,12p' "$0" >&2
    exit 2
fi

SCRIPT=$1
MANIFEST=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
MARKER=$3
NAME=$4
shift 4

todo=()
n=0
while IFS= read -r cell || [ -n "$cell" ]; do
    if [ -z "$cell" ]; then
        echo "ERROR: blank line $((n + 1)) in $MANIFEST" >&2
        exit 2
    fi
    [ -e "$cell/$MARKER" ] || todo+=("$n")
    n=$((n + 1))
done < "$MANIFEST"

done_count=$((n - ${#todo[@]}))
if ! queue=$(squeue -u "$USER" -h -r -n "$NAME"); then
    echo "ERROR: squeue failed, nothing submitted" >&2
    exit 1
fi
queued=$(printf '%s' "$queue" | grep -c . || true)

if [ "${#todo[@]}" -eq 0 ]; then
    echo "DONE       $NAME  ($done_count/$n cells)"
    exit 0
fi
if [ "$queued" -gt 0 ]; then
    echo "IN QUEUE   $NAME  ($done_count/$n cells done, $queued task(s) queued)"
    exit 0
fi

tasks=$(IFS=,; echo "${todo[*]}")
if [ "$DRY_RUN" -eq 1 ]; then
    echo "WOULD RUN  $NAME  ($done_count/$n cells done, tasks $tasks)"
    exit 0
fi

id=$(sbatch --parsable --job-name="$NAME" --array="$tasks" \
            --export=ALL,MANIFEST="$MANIFEST",DONE_MARKER="$MARKER" "$@" "$SCRIPT")
echo "SUBMITTED  $NAME  ($done_count/$n cells done) -> job $id, tasks $tasks"
