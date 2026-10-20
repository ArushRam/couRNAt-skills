---
name: slurm-cluster
description: Write, submit, monitor, and resubmit Slurm GPU jobs from a shell on the cluster itself, shaped to start fast on a shared pool. Covers job arrays, one-GPU-per-task packing, listing several partitions, short resumable wall times, done-marker skips, resubmitting only unfinished tasks, and reading pending reasons. Use this skill for any sbatch script, any sweep or grid of training runs, and any choice of GPUs, CPUs, memory, or hours. Also use it for a job stuck PENDING, for squeue or sacct output, and for rerunning failed or timed-out tasks. The user can say only "run this on the cluster" or "launch the grid". On NYU torch, also read references/torch.md.
---

# Slurm on the cluster

This skill is for a Claude that runs in a shell on the cluster login node. You call `sbatch`, `squeue`, `sacct`, and `sacctmgr` directly.

Wait time is usually the largest cost of a job, more than run time. Most of this skill is about the shape of the request. The shape decides whether a job starts in two minutes or in a day.

If the cluster is NYU torch, read `references/torch.md` now. It has the account, the partitions, the pool sizes, and the container paths.

## 1. Know which limit blocks you

Two different limits get confused:

- The per-user limit. This is the most GPUs that you can hold at once.
- The partition pool limit. This is the most GPUs that all users together can hold on one partition. This is the limit that usually blocks you.

Measure both before you plan a large submission. Always pass `-P` to `sacctmgr`. Without it, the column-aligned output is easy to misread, and earlier notes on torch got the per-user limit wrong twice because of this.

```bash
sacctmgr -P show assoc user=$USER format=Account,Partition,QOS
sacctmgr -P show qos format=Name,MaxTRESPU,GrpTRES,MaxWall
scontrol show partition <name> | grep -E 'QoS|AllowAccounts|MaxTime|Nodes='
squeue -p <partition> -t PD -h -o '%r' | sort | uniq -c    # why other jobs wait
```

## 2. Shape the request

Use these defaults unless the job needs something else:

| Setting | Default | Reason |
|---|---|---|
| GPUs per task | 1 | When a pool is full, the free GPUs are usually one per node. A 2-GPU request waits for two free GPUs on one node, and Slurm holds no reservation for a job blocked on a pool limit. On torch, a 2-GPU canary waited 25.6 hours. 1-GPU tasks started in about 2 minutes. |
| Partitions | All partitions with the same GPU model, comma-separated | Slurm places the job on whichever partition frees a GPU first. Some partitions front the same nodes with a larger pool. |
| Wall time | 4 hours | Short jobs backfill into gaps left by long jobs. The job must be resumable for this to work (section 4). |
| CPUs and memory | 2 CPUs, 32G, then raise from measurement | Large requests shrink the set of nodes that can take the task. |
| Many cells | One job array, one cell per task | Each cell queues by itself and reports at its end. |

Do not run a whole grid inside one job that uses one GPU. The grid runs one cell at a time, goes past the wall time, and returns nothing until the end. Do not submit N separate jobs either. An array is one submission that you can watch, cancel, and resubmit as a unit.

If you compare cells on run time or need bit-level parity, do not mix GPU models. For a paired comparison, keep every cell on one GPU model and leave out partitions with other models.

To inspect the first cells before the rest start, use a throttle (`--array=0-31%8`). This catches divergence early. After you trust the run, remove the throttle.

Put more than one cell on one GPU only after you measure. Run one cell with `nvidia-smi dmon -s u -d 60` in the background. If GPU use and memory stay low, run two or more cells as background processes in one task and `wait` for all of them.

## 3. Map tasks to cells with a frozen list

Every array task must find its cell from `$SLURM_ARRAY_TASK_ID` alone. Write the cell list to a file once, before you submit, and read line `TASK + 1`:

```bash
CELL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
```

Do not build the list at run time with `ls configs/*.yaml | sort`. If someone adds a config while tasks are still pending, every later task moves to a different cell. A manifest file also gives you the array size (`wc -l`) and is the input for the resubmit script.

If the factors are fixed, arithmetic decoding (`config = t / n_seeds`, `seed = t % n_seeds`) is fine. Write the mapping in a comment at the top of the script.

## 4. Make every task resumable and safe to run twice

A task can stop because of the wall time, a node failure, or a requeue. Design for this:

1. Save a checkpoint often enough that a restart loses less than about 15 minutes.
2. On start, look for a checkpoint. If one exists, resume from the newest.
3. Write a done marker last, after all outputs exist. Use a file that the code writes as its final step, for example `test_metrics.csv`.
4. On start, look for the done marker. If it exists, exit 0 at once. Then you can resubmit the whole array, and finished cells cost a few seconds.
5. Add `#SBATCH --requeue` so that Slurm can restart a task after a node failure.

To save state before the wall time ends, ask Slurm for an early signal and trap it. Bash runs a trap only after the foreground command returns, so run the work in the background and `wait`:

```bash
#SBATCH --signal=B:USR1@300        # USR1 to the batch shell 5 minutes before the limit
python train.py "$CELL" &
PID=$!
trap 'echo "[$(date -u)] wall time near, final save"; kill -USR1 "$PID"; wait "$PID" || true; exit 124' USR1
RC=0
wait "$PID" || RC=$?
```

The program must handle `USR1` (save a checkpoint and exit). If it cannot, the trap can still upload or copy what exists before the job ends.

Start the program as a plain command with `&`, not through a shell function. A function started with `&` runs in an extra subshell, and `kill -USR1 "$PID"` then reaches the subshell instead of the program.

Test the trap on the canary instead of waiting for the wall time: `scancel --batch --signal=USR1 <jobid>_<task>`. The log must show the final save, and the task must exit with code 124.

## 5. Write the script

Start from `assets/array.sbatch`. It already has the parts below. Each part prevents a failure that happened on a real run:

- `set -euo pipefail`, so that a failed step stops the task.
- `export PYTHONUNBUFFERED=1`. Without it, the `.out` file stays empty until the task ends.
- Log names `%x_%A_%a.out` (job name, array id, task id), so the logs of one array sort together.
- A provenance line at task start: hostname, UTC date, git commit, `md5sum` of the config and the model file, and the GPU name from `nvidia-smi`. When two cells disagree later, this line tells you whether they ran the same code on the same hardware.
- Caches and `TMPDIR` on scratch, not home. Home quotas fill without a warning.
- A fail-fast check before GPU work starts. For example, load the config, build the model, and assert the values that the experiment depends on. A wrong config that fails in 10 seconds costs nothing. The same config that fails after 4 hours costs a full slot, and in an array it costs every task.
- Config derivation inside the container or environment that runs the job. A bare compute node's `python3` can lack `pyyaml`, and in an array that failure takes out every task at once.

Give each experiment its own code checkout. A task reads the code at its start, not at submission, so a `git pull` changes the code for every pending task of that checkout. Do not pull in a checkout while its array has pending tasks.

Before you submit, run `sbatch --test-only <script>`. This catches a wrong account, partition, or time limit without queuing anything.

For vLLM jobs, set `export VLLM_WORKER_MULTIPROC_METHOD=spawn` before any Python runs. Never write `python -c "import X" || pip install X` in an sbatch script. The import can initialize CUDA on the GPU node and hang. Use `pip install --quiet X`. If the package is present, it does nothing.

## 6. Submit, then watch

```bash
sbatch --parsable my.sbatch                    # prints the job id, keep it
squeue -u $USER -r -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'   # -r lists each array task
sacct -j <jobid> -X -o JobID,State,ExitCode,Elapsed,NodeList  # finished tasks
tail -n 50 /path/to/logs/<name>_<jobid>_<task>.out
```

Give every status report a UTC timestamp. The reader then knows the time of your last check.

When a job is PENDING, read the reason in the last column and act on it:

| Reason | Meaning | Action |
|---|---|---|
| `QOSGrpGRES` | The partition pool is full across all users. | Add another partition with the same GPU model. Resubmitting to the same partition does nothing. |
| `QOSMaxGRESPerUser` | You hold your own GPU limit. | Wait. Your running jobs keep running. |
| `Resources` | No node has the CPUs, memory, or GPUs that you asked for. | Lower CPUs or memory, or ask for 1 GPU. |
| `Priority` | Other jobs are ahead of you. | Wait, or shorten the wall time so that the task can backfill. |
| `JobArrayTaskLimit` | Your own `%N` throttle. | Raise it with `scontrol update JobId=<id> ArrayTaskThrottle=<N>`. |
| `PartitionTimeLimit` | The wall time is longer than the partition allows. | The job never starts. Cancel it and resubmit with a shorter `--time`. |
| `ReqNodeNotAvail`, `Reservation` | Maintenance or a reservation. | Run `scontrol show reservation` and check whether your wall time crosses it. |

Submitting more jobs does not cancel your running jobs. A job over your limit stays PENDING, and nothing evicts the running one. Do not run work one job at a time because of this fear. If `sacct` shows `CANCELLED by 0`, an administrator cancelled the job (uid 0 is root).

## 7. Resubmit only what is unfinished

After tasks time out or fail, resubmit only the cells that are neither done nor queued. Use `scripts/resubmit_incomplete.sh`:

```bash
bash scripts/resubmit_incomplete.sh --dry-run my.sbatch cells.txt test_metrics.csv myjob
bash scripts/resubmit_incomplete.sh           my.sbatch cells.txt test_metrics.csv myjob
```

The script reads the manifest and checks each cell directory for the done marker. If any task of the job is still queued, it submits nothing. Otherwise it submits `--array=<list of unfinished task ids>`. Run it as often as you like. Repeat until it prints `DONE`.

Before a resubmit, read the log of one failed task. A timeout with a checkpoint is normal: resubmit. A Python traceback, `CUDA out of memory`, or NaN loss is a bug: fix it first, or all the tasks fail again. For out-of-memory errors, report to the user before you change batch size or sequence length, because that changes the experiment.

## 8. Work that runs beside the job

- Upload or sync results from one task only, not from every task. Many tasks pushing to one remote repository at once race each other. Pass a role argument (for example `sync`) to one task, or use a separate small job.
- If the cell directories are disjoint, several tasks can write into one results folder. Any summary file that each task rewrites is last-writer-wins. Rebuild it from the cell directories at the end.
- Never run training or heavy preprocessing on the login node. Use `srun --pty` or an sbatch job.
