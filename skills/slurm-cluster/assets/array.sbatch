#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --account=<account>
#SBATCH --partition=<partition>[,<partition with the same GPU model>]
#SBATCH --array=0-<N-1>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/%u/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x_%A_%a.err

# One array task per cell. Copy this file next to the experiment and replace every <...>.
#
# MANIFEST is a text file with one cell directory per line. Each directory holds that
# cell's config. The task writes its outputs and DONE_MARKER into the same directory.
# Write the manifest once, before you submit, and do not edit it while tasks are pending:
# task t always runs line t+1. Set the array size from it: N = $(wc -l < cells.txt).
#
#   mkdir -p /scratch/$USER/logs      # Slurm does not create the log directory
#   sbatch --array=0-$(( $(wc -l < cells.txt) - 1 )) \
#          --export=ALL,MANIFEST=$PWD/cells.txt,DONE_MARKER=test_metrics.csv this.sbatch
#
# Resubmitting is safe: a cell with DONE_MARKER exits at once, and an unfinished cell
# resumes from its newest checkpoint (the training code must do this). To resubmit only
# unfinished cells, use scripts/resubmit_incomplete.sh.
#
# #SBATCH lines cannot read shell variables. %u is the user name, %x the job name,
# %A the array job id, %a the task id. Change /scratch/%u if scratch is elsewhere.

set -euo pipefail

MANIFEST=${MANIFEST:?pass --export=ALL,MANIFEST=/abs/path/cells.txt}
DONE_MARKER=${DONE_MARKER:-DONE}
PROJECT=<absolute path of this experiment's own code checkout>
TASK=${SLURM_ARRAY_TASK_ID:?run this with sbatch --array}

CELL=$(sed -n "$((TASK + 1))p" "$MANIFEST")
if [ -z "$CELL" ]; then
    echo "no cell on line $((TASK + 1)) of $MANIFEST" >&2
    exit 2
fi
if [ -e "$CELL/$DONE_MARKER" ]; then
    echo "done already: $CELL"
    exit 0
fi

export PYTHONUNBUFFERED=1
export TMPDIR=${SCRATCH:-/scratch/$USER}/tmp
export HF_HOME=${SCRATCH:-/scratch/$USER}/.cache/huggingface
export TORCH_HOME=${SCRATCH:-/scratch/$USER}/.cache/torch
mkdir -p "$TMPDIR"

# ENV is the command prefix that enters the job's software environment. Keep it a prefix
# array, not a shell function: a function started with & runs in an extra subshell, and
# the USR1 below would reach that subshell instead of the program.
# Plain conda:
#   ENV=(conda run --no-capture-output -n <env>)
# Singularity with a read-only overlay (NYU torch):
#   IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
#   OVERLAY=<path>.ext3
#   ENV=(singularity exec --nv --overlay "$OVERLAY":ro "$IMAGE"
#        bash -c 'source /ext3/env.sh && conda activate base && cd "$PROJECT" && exec "$@"' _)
ENV=(env)
export PROJECT
cd "$PROJECT"

# Provenance: when two cells disagree later, this says whether they ran the same code
# on the same hardware.
echo "job ${SLURM_ARRAY_JOB_ID}_${TASK} | host $(hostname) | $(date -u +%FT%TZ) | code $(git -C "$PROJECT" rev-parse --short HEAD)$(git -C "$PROJECT" diff --quiet || echo ' (dirty)')"
echo "cell $CELL"
md5sum "$CELL"/*.yaml 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail fast: check the cell's config before GPU time is spent. A bad config that fails
# here costs seconds. The same config that fails after 4 hours costs every task's slot.
# "${ENV[@]}" python <check script> "$CELL"

# The work runs in the background so that the USR1 trap can fire. Bash runs a trap only
# after the foreground command returns. The program must save a checkpoint and exit on
# USR1. Slurm sends USR1 300 s before the wall time (see --signal above).
"${ENV[@]}" python <train script> "$CELL" &
PID=$!
trap 'echo "[$(date -u +%FT%TZ)] wall time near: USR1 to $PID"; kill -USR1 "$PID" 2>/dev/null; wait "$PID" || true; exit 124' USR1

RC=0
wait "$PID" || RC=$?
if [ "$RC" -eq 0 ] && [ ! -e "$CELL/$DONE_MARKER" ]; then
    touch "$CELL/$DONE_MARKER"
fi
echo "[$(date -u +%FT%TZ)] exit $RC"
exit "$RC"
