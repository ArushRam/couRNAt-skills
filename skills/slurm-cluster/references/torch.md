# NYU torch

These facts are specific to the NYU torch cluster. Each fact carries the date of its measurement. Limits change, so re-measure with the command shown before you plan a large submission.

This file holds no user, account, or project names. Find them at run time with the commands below.

## Your account and paths

```bash
echo "$USER" "$SCRATCH"                                       # user name and scratch directory
sacctmgr -P show assoc user=$USER format=Account,Partition,QOS  # Slurm accounts you can charge
```

Put the account in every script with `#SBATCH --account=<account>`. If you hold more than one account, ask the user which one to charge.

## Limits

The per-user limit is QOS `gpu48`: 24 GPUs at once and 48 hours of wall time per job (measured 2026-07-31). This limit almost never blocks you.

```bash
sacctmgr -P show qos gpu48 format=Name,MaxTRESPU,MaxWall
# gpu48|gres/gpu=24|2-00:00:00
```

## Partitions

| Partition | GPUs | Pool for all users | Notes |
|---|---|---|---|
| `l40s_courant` | L40S 48 GB, 4 per node | 52 GPUs | Often 52/52 in use. Jobs then wait with `QOSGrpGRES`. |
| `l40s_public` | L40S 48 GB, same `gl[001-068]` nodes | 208 GPUs | `AllowAccounts=ALL`, so a courant account works. |
| `h200_courant` | H200 141 GB, 8 per node | | For jobs that do not fit on 48 GB. |
| `a100_courant` | A100 80 GB, 4 per node | | |

The `_courant` partitions need a Courant account. Run `scontrol show partition <name> | grep AllowAccounts` to check yours.

For L40S work, ask for both L40S partitions:

```bash
#SBATCH --partition=l40s_courant,l40s_public
```

They use the same hardware, and `PreemptMode=OFF` is set on both, so the choice changes only the pool. Slurm starts the task on whichever partition frees a GPU first.

`l40s_plus` advertises `AllowAccounts=ALL`, but `sbatch` rejected it for every Courant account tested. Run `sbatch --test-only` before you rely on it.

If the other cells of a paired comparison ran on L40S, leave out `h200_courant` and `a100_courant`. A cell on another GPU model is not comparable on run time.

## Measured wait times

- A 2-GPU canary waited 25.6 hours. Single-GPU jobs submitted at the same time started in about 2 minutes.
- An 8-task array (1 GPU, 4 hours, 2 CPUs, 32G, both L40S partitions) never stuck PENDING. All 8 tasks completed. Four took about 3h23m, and two warm-started tasks took about 2h50m.
- 16 one-GPU shards on both partitions replaced 2 jobs with 2 GPUs each. Those 2 jobs had needed about 47 and 107 hours.

## Cancellations

`CANCELLED by 0` in `sacct` means an administrator cancelled the job (seen 2026-07-27). Submitting another job never cancels your own running job.

## Paths

| What | Path |
|---|---|
| Scratch (large, purged) | `$SCRATCH`, which is `/scratch/$USER` |
| Archive (long-term) | `/archive/$USER` |
| Home (small quota) | `$HOME` |

Keep logs, temporary files, caches, code checkouts, and results on scratch. A common layout is `$SCRATCH/logs`, `$SCRATCH/tmp`, `$SCRATCH/.cache/huggingface`, and `$SCRATCH/projects/<project>/`.

## One checkout per experiment

A task reads the code at its start, not at submission. A `git pull` in a checkout therefore changes the code for every task of that checkout that is still pending. Give each experiment its own checkout:

```bash
cd "$SCRATCH/projects/<project>"
git clone <main-checkout> <experiment-checkout>    # or git worktree add
```

Do not pull in a checkout while its array has pending tasks.

## Container

Torch jobs run inside Singularity with a read-only conda overlay. The base image is shared:

```bash
IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/environments/<env>.ext3     # ask the user, or run: ls $SCRATCH/environments
singularity exec --nv --overlay "$OVERLAY":ro "$IMAGE" bash -c '
    source /ext3/env.sh && conda activate base
    cd "$PROJECT" && python train.py "$CONFIG"'
```

Use `--overlay ...:ro` for every job. More than one job can mount an overlay read-only at the same time, but a read-write mount locks it.

Keep HuggingFace uploads in a separate overlay and process, not in the training process.

The compute node's own `python3` outside the container can lack `pyyaml` and the project packages. Run every Python step that imports project code inside the container.

## HuggingFace and mail

Read the HuggingFace org from `$HF_ORG`. If it is not set, ask the user. Set `HF_HOME=$SCRATCH/.cache/huggingface`.

For a failure email, add these lines with the user's address:

```bash
#SBATCH --mail-user=<address>
#SBATCH --mail-type=FAIL
```

For an array, `--mail-type=FAIL` sends one email for the whole array. Adding `ARRAY_TASKS` sends one email for each task. For a large array, leave off `ARRAY_TASKS`.
