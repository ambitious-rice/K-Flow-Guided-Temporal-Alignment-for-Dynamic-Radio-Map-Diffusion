---
name: remote-research-workflow
description: "Manage this project's local-led research workflow: configure a remote compute target, synchronize source through GitHub, run remote research tasks, and preserve run state. Use for RMDM remote development and execution; do not use for unrelated SSH hosts or generic file transfer."
---

# Remote Research Workflow

Use this skill for the project's local-development and remote-execution workflow. `AGENTS.md` is the governing project policy; read it and `.agents/config.yaml` before remote work.

## Core boundaries

- Develop, validate, manage Git, retain experiment state, and analyze results locally.
- Use remote nodes for environments, computation, long-running tasks, logs, checkpoints, and raw results.
- Treat the local Git repository as the source of truth. Do not modify Git-tracked source on a remote node.
- Synchronize source code only through GitHub. Use external storage for datasets, models, checkpoints, and results; do not use it for source deployment.
- Record significant or long-running jobs in `.agents/runs/`; verify their current state on the remote machine before reporting it.

## Configuration

`config.yaml` stores stable project facts: local and remote roots, connection data, repository and branch, named per-server execution environments, resource roots, artifact roots, and the external-storage mapping. It may contain connection credentials at the user's direction. Keep it local-only, mode `600`, and excluded from Git; never reveal credentials in a command line, log, commentary, or final response.

Select a task environment by its configured ID. Do not assume that one Conda environment fits every task or server; use the configured Python executable for non-interactive execution and verify a new environment before adding it to the registry.

Support two entry modes:

- **Config-first:** Read the existing config and validate only the fields needed for the requested operation.
- **Prompt-first:** Write user-provided and safely discovered stable facts to a local draft config, then ask only for values that cannot be discovered safely. Do not invent paths, credentials, or storage roots.

## References

- For remote connection, identity checks, and the no-forwarding boundary, read [references/remote-connection.md](references/remote-connection.md).
- For local initialization, GitHub synchronization, remote code deployment, and LFS handling, read [references/code-sync.md](references/code-sync.md).
- For task deployment decisions, named-environment selection, detached execution, state updates, and verified RMDM examples, read [references/task-deployment.md](references/task-deployment.md).
- For cloud-storage login or transfers of datasets, models, checkpoints, or results, read [references/artifact-sync.md](references/artifact-sync.md).

Keep environment, dataset, model, execution, and result-transfer procedures grounded in verified project configuration and exercised workflows.
