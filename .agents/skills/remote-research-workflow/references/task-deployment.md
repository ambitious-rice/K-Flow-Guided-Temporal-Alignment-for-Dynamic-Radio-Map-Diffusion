# Task deployment and execution

Read this reference when turning a local RMDM change into a remote computation, or when deciding whether an operation needs a persistent run record. It complements `remote-connection.md`, `code-sync.md`, and `artifact-sync.md`; it does not replace their connection, Git, or storage boundaries.

## Required task inputs and interaction rules

Before deployment, collect the task's goal, intended split/scale, command or configuration, required inputs, expected output directory, and whether it is a brief check or significant computation. Read `AGENTS.md` and `.agents/config.yaml`, then select the configured server and a named environment from `remote.servers.<server>.environments.items`.

- Use the server's `environments.default` only when no task-specific environment is stated.
- If the named environment, model/data path, execution checkout, output path, or destructive overwrite policy is unknown, ask the user instead of inventing it.
- Do not ask for routine SSH, Git, environment, or status checks that configuration and the remote machine can answer.
- A code-only deployment, a login check, or a short diagnostic does not need a run record. A task that is important, long-running, GPU-backed, or worth reproducing must have `.agents/runs/<run-id>.yaml` before or immediately after it is started.

## Deployment decision flow

```text
local change and validation
        |
        v
commit exact source -> push configured Git branch
        |
        v
connect and verify backend identity -> inspect remote worktree
        |
        +-- clean and at a known revision -> fast-forward to GitHub commit
        |
        +-- unexplained source changes -> stop; do not overwrite or run there
        |
        v
choose configured named environment and task-specific output directory
        |
        +-- code-only or brief diagnostic -> run foreground if needed; no run record
        |
        +-- important/long computation -> start detached, verify process/log/GPU,
                                      then persist true state in .agents/runs/
        |
        v
inspect completion on remote -> select small necessary artifacts -> one-time cloud sync
        |
        v
update the existing run record with status, outputs, and next action
```

Source always travels `local source -> GitHub -> remote checkout`. Models, datasets, checkpoints, and results use the configured external storage only when needed. Do not use cloud storage to deploy Git-tracked source, and do not use SHA/checksum manifests in artifact transfer.

## Remote checkout selection

After connection and identity verification, inspect the exact intended work directory:

```bash
cd "<workdir>"
git status --short --branch
git rev-parse HEAD
git remote -v
```

Only when its tracked source tree is understood and safe, deploy with a fast-forward-only update:

```bash
git fetch origin main
git merge --ff-only origin/main
git rev-parse HEAD
```

If the default remote project directory has unexplained modifications, do not reset it. Use an already verified clean execution checkout, or create a separate clean checkout through Git after recording its path in the task's run record. Preserve prior result-only directories unless the user authorizes their removal.

## Environment selection and command construction

Resolve the selected environment from the chosen server's configuration. For a non-interactive command, use its configured `python` executable directly:

```bash
PYTHON="<configured-environments.items.<environment-id>.python>"
cd "<workdir>"
CUDA_VISIBLE_DEVICES="<gpu-list>" "$PYTHON" -m accelerate.commands.launch \
  --num_processes "<processes>" --main_process_port "<free-port>" \
  <script-and-arguments>
```

Do not assume that `conda activate` works in a detached shell. The Conda name remains useful for interactive diagnosis, while the configured Python path is the reproducible execution selector. Record the environment ID, Conda name, and actual Python path in a significant task's run record. When a task requires another environment, add a new named item to the applicable server only after remotely verifying it; do not replace the default merely because one task used another environment.

## Execution modes and persistent state

Use foreground execution only for a short, observed diagnostic. For significant or potentially disconnected work, create the output directory and start a detached session (for example, `tmux`) with a unique run ID/session name. Redirect stdout and stderr to the task log, then verify that the process actually started by checking the session, PID, initial log output, and GPU use when applicable.

```bash
mkdir -p "<output-dir>"
tmux new-session -d -s "<session>" \
  'cd "<workdir>" && exec "<python>" <script-and-arguments> > "<log>" 2>&1'
tmux has-session -t "<session>"
ps -fp "<pid>"
tail -n 40 "<log>"
```

The shell quoting above is schematic: construct the final command so `<workdir>`, `<python>`, script arguments, and `<log>` are resolved before starting the session. Never report a task as started merely because `tmux new-session` returned success.

Persist only durable, reproducing facts in `.agents/runs/<run-id>.yaml`: goal; source commit/branch/dirty state; server/workdir; full command; environment ID/name/Python; session/PID/log/timestamps; status; artifact locations; key result; and next action. Do not record transient shell mistakes or routine diagnostics.

## Completion and artifact handling

When checking a task, remote state is authoritative. Inspect the process/session, log tail, output files, and application exit status. Update the existing run record when the task enters a material new state such as completed or failed.

For results, stage and sync only the useful small artifacts—configuration, metrics, summary, report, and selected log or previews—using the one-time `aliyunpan sync` workflow. Confirm command success and the expected files at the destination; do not create, compare, or require SHA/checksum files. Do not automatically synchronize checkpoints, datasets, caches, intermediate tensors, or complete run trees.

## Verified RMDM cases

### W16 noise-estimation smoke (computation task)

- Goal: verify the frozen W16 no-Tx estimator on three validation scenes, one video/window per scene, for clean and `sigma=0.01` inputs.
- Source: commit `6a193154d82d382e0877a2fa4689e7f1fe950f58` on `main`.
- Server and checkout: `zjlab`, clean checkout `/data/fzj/RMDM_smoke_20260831` because the default project tree had unexplained changes.
- Environment: `rmdm_hvdit_v2` (`RMDM_HVDIT_V2`), using `/data/fzj/conda_envs/RMDM_HVDIT_V2/bin/python` with one visible GPU.
- Execution: detached session `w16_noise_estimation_smoke_20260831`; evaluation followed by `scripts/summarize_w16_noise_smoke.py`; output `/data/fzj/RMDM_smoke_20260831/runs/w16_noise_estimation_20260831/smoke`.
- State and result: persisted in `.agents/runs/20260831_w16_noise_estimation_smoke.yaml`; six units completed. All scenes detected an increase, but the smoke criterion failed because injected-noise MAE was `0.015468719052713789`, above `0.01`.
- Synced artifacts: run configuration, log, summary JSON, and report Markdown to `/fzj/RMDM/w16_noise_estimation_smoke_20260831`, then to the configured local results directory. No checkpoint or full run tree was synchronized.

### SHA-removal deployment (code-only task)

- Goal: remove SHA/checksum logic and standardize the project-control directory as `.agents`.
- Source: commit `6dfd7d1e6d7aaea29576f290f0cee1a5e8c9ee29` on `main`.
- Local work: update source, configuration template, policy, and workflow documentation; run Python compilation and skill validation; push `main`.
- Remote work: inspect the same clean checkout, confirm its only untracked content was the known W16 result directory, then run `git fetch origin main` and `git merge --ff-only origin/main`.
- Result: clean checkout advanced from `6a19315` to `6dfd7d1`; no computation, artifact transfer, or run record was created.

These examples are historical deployment patterns, not defaults for later task parameters. Later runs must use their own run IDs, source commits, output directories, resource limits, and verified environment selection.
