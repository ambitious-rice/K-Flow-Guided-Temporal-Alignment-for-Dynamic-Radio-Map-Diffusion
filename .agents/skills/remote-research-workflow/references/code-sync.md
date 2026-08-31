# Source synchronization through GitHub

Read this reference for first-time local setup, source changes, GitHub pushes, or remote deployment. Source code is synchronized through GitHub only; direct SSH transfer and external-storage source copies are not part of this workflow.

## Configuration fields

```yaml
project:
  local_root: /data_p6/fzj/projects/RMDM

remote:
  servers:
    zjlab:
      project_root: /data/fzj/RMDM
      environments:
        default: rmdm_hvdit_v2
        items:
          rmdm_hvdit_v2:
            manager: conda
            conda_name: RMDM_HVDIT_V2
            python: /data/fzj/conda_envs/RMDM_HVDIT_V2/bin/python

git:
  repository: https://github.com/ambitious-rice/K-Flow-Guided-Temporal-Alignment-for-Dynamic-Radio-Map-Diffusion.git
  branch: main
  remote_name: origin
```

`repository` is the GitHub HTTPS address used locally and remotely. `branch` is the source branch to verify and deploy. `remote_name` is normally `origin`. `local_root` and `project_root` are distinct and must not be inferred from the current shell directory.

`remote.servers.<server>.environments` is a per-server registry, so one project can use multiple isolated Conda environments. A task selects an ID from `items`; `default` is only a fallback when the task has no specific environment requirement. Each entry must state its manager, Conda environment name, and the exact Python executable verified on that server. Prefer the configured Python path in non-interactive task commands instead of relying on shell activation. Add an environment only after checking it remotely; never guess a path or copy an environment definition between unlike servers.

## First-time local setup

When `project.local_root` has no `.git` directory, initialize the repository directly in that directory rather than cloning into a nested folder. Fetch the configured repository and branch, inspect possible conflicts with existing local project-control files, then check out the branch in place.

For a code-only checkout, set `GIT_LFS_SKIP_SMUDGE=1`. This preserves LFS objects as pointers and avoids downloading models or checkpoints as part of source setup. Do not use an archive, a cloud drive, SCP, rsync, or SSH forwarding to bootstrap source code.

The verified RMDM bootstrap result is a flat layout:

```text
/data_p6/fzj/projects/RMDM/
├── configs/
├── docs/
├── scripts/
├── src/
└── tests/
```

## Normal deployment

1. Make and validate source changes locally.
2. Check the local Git status, then commit or otherwise preserve an exact recoverable local state.
3. Push the configured branch to GitHub.
4. Connect to the configured remote node and inspect its repository, branch, revision, and working tree.
5. If the remote tree has unexplained changes, stop and report them. Do not reset, force-checkout, delete, or merge them into a new experiment.
6. If it is safe to update, fetch the configured GitHub branch and perform a fast-forward-only update.
7. Verify the resulting remote commit before launching work; record that commit in any important run record.

The required direction is:

```text
local source → GitHub → remote execution checkout
```

## Artifact and storage boundary

`resources.model_root` is a model root, not a catalogue of individual checkpoint files. Specific checkpoints and result paths belong in `.agents/runs/` records.

Read [artifact-sync.md](artifact-sync.md) before transferring any artifact. External storage must never be used to deploy source code.

```yaml
resources:
  dataset_root:
  model_root:

artifacts:
  remote_results_root: /data/fzj/RMDM/runs
  local_results_root: /data_p6/fzj/projects/RMDM/results

external_storage:
  name: aliyun-drive
  command: aliyunpan
  credential_profile: default
  drive: resource
  server_roots:
    zjlab: /fzj
  sync_defaults:
    policy: increment
    cycle: onetime
```

`external_storage.server_roots` maps each remote node to its cloud-storage root. It is for large files only. Select and synchronize the necessary metrics, metadata, previews, logs, or checkpoints for a task; do not blindly transfer caches, intermediate tensors, or complete source trees.

## Cloud-client login

When an artifact transfer is requested, first verify the client with `aliyunpan who`. If it reports that no account is logged in, run `aliyunpan login`. The client prints a five-minute authorization URL and waits for Enter.

Present that temporary URL to the user and ask them to complete the browser authorization and required scan. After the user confirms completion, send Enter to the waiting command and verify `aliyunpan who` again. If the URL expired or the command failed, cancel it and start a fresh login; do not attempt to automate the scan, pass credentials on the command line, or copy login configuration between local and remote machines. Each machine maintains its own login state.
