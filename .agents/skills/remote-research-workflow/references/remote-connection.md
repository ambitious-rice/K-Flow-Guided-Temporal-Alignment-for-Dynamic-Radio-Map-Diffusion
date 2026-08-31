# Remote connection

Read this reference for remote inspection, environment work, job control, result operations, or any other operation that needs a backend session.

## Configuration fields

The connection reference reads the selected node from `.agents/config.yaml`:

```yaml
remote:
  default: zjlab
  servers:
    zjlab:
      connection:
        type: gateshell
        host: 192.168.10.10
        port: 60022
        username: "..."
        password: "..."
        target: zjlab
      gateshell_backend:
        selector_command: ssh zjlab@10.107.207.47:22
        expected_address: 10.107.207.47
        expected_hostname: zjlab-ESC8000A-E11
        expected_username: zjlab
        expected_initial_directory: /home/zjlab
      project_root: /data/fzj/RMDM
```

- `default` selects the node when the user has not named one.
- `connection` describes the GateShell gateway, not the backend's direct SSH service.
- `target` selects the GateShell backend.
- `gateshell_backend` records the verified selection and identity facts used by the persistent-session helper. Add all of these fields for every new GateShell server; do not put a new backend identity only in a script or chat message.
- `project_root` is the remote project directory after identity verification.

The current verified targets are `zjlab` and `wzy`. Add another server as a named entry under `remote.servers`; do not hard-code an additional machine into the skill.

## Connection procedure

1. Confirm Tailscale is running, `RouteAll` is enabled, and `ip route get <gateway-host>` uses `tailscale0`. If routes are not accepted, use `sudo tailscale set --accept-routes=true`; if local sudo authority is unavailable, ask the user to run only that command and recheck. A route through another interface is not valid for this gateway.
2. Optionally test the configured gateway port without credentials. Then open an interactive TTY SSH session to the configured GateShell gateway. Supply the configured password only after its password prompt is visible. If there is no banner or password prompt and input is echoed locally, stop and diagnose the route instead of entering it again.
3. Inspect GateShell's displayed rows and explicitly select the configured backend; never accept a default row, navigate with `j`/`k`, filter the list, or send a bare Enter.

## Persistent operator session

For repeated interaction with one backend, do not repeatedly create fresh SSH logins. Maintain one **local** tmux session named `rmdm-gateway-<server>` that contains the nested GateShell-to-backend terminal. The provided helper is the standard interface:

```bash
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py open <server>
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py authenticate <server>
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py backend <server>
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py verify <server>
```

`authenticate` refuses to submit a configured password unless a password prompt is currently visible. It passes the password to tmux through standard input, never as a command-line argument or displayed output. `backend` refuses to select a target unless the configured backend address is visible in GateShell first, and preserves the required three-write input discipline.

After identity verification, use the same session for short remote commands:

```bash
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py exec <server> -- 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py capture <server>
```

The operator session is deliberately serialized. To run operations concurrently, launch separate **remote job** tmux sessions from this one operator terminal; do not create extra interactive gateway logins merely for parallelism. A second local operator session is only for a genuinely independent interactive investigation, must have an explicit suffix, and should be closed promptly (for example, `open zjlab --suffix diagnostics`).

Before opening a session, reuse a live session after recapturing and, when necessary, re-verifying its backend identity. When the work is done, or before handing off a task, inspect and recycle idle operator sessions:

```bash
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py sessions
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py gc --max-idle-hours 12
python3 .agents/skills/remote-research-workflow/scripts/remote_session.py close <server>
```

Closing a local operator session never stops remote job tmux sessions. Do not kill a session just because an SSH network path is stale: first check whether it is an operator session and preserve/inspect the separately named remote jobs.

### Verified GateShell target commands

For the two configured RMDM targets, first confirm that the displayed label and backend address match this table, then use the corresponding command. These are fixed, verified GateShell backend identities; a new target belongs in `.agents/config.yaml` and must be verified before it is added here.

| Config target | GateShell command | Expected hostname | Expected user | Expected initial directory |
| --- | --- | --- | --- | --- |
| `zjlab` | `ssh zjlab@10.107.207.47:22` | `zjlab-ESC8000A-E11` | `zjlab` | `/home/zjlab` |
| `wzy` | `ssh wzy@10.119.1.24:22` | `ubuntu-SYS-420GP-TNR` | `wzy` | `/home/wzy` |

### GateShell input discipline

When GateShell shows its full-screen list, enter the selected command in **three separate writes**, with a short output check between them:

1. Send only `:` and wait until the command line visibly shows it.
2. Send only the selected `ssh user@host:22` text, without Enter, and wait until it is visibly echoed.
3. Send only carriage return and wait for the backend banner or shell prompt.

Do not send `:ssh ...` in one write: GateShell can consume only the mode-switch character and discard the rest. If GateShell instead presents its ordinary `[GateShell]$` prompt, send the selected SSH command normally without a leading `:`. If a command is partial or terminal state is ambiguous, close the session and reconnect instead of guessing.

### Backend identity verification

Before operating on the project, verify the selected backend with a read-only command that prints `hostname`, `id -un`, the working directory, and current time. Reject the session if any observed identity differs from the table above.

## No direct transfer, forwarding, or assumed jump host

The compute backends do not accept direct port-22 access or SSH forwarding from this local machine. Therefore `ProxyJump` / `-J`, `ProxyCommand`, and OpenSSH ControlMaster do **not** replace the GateShell selector in the current deployment. Do not attempt SCP, rsync-over-SSH, SSH `-L`/`-R`/`-D`, `ProxyCommand`, port forwarding, or direct backend SSH.

The persistent tmux operator terminal is the supported performance optimization: it keeps the one real gateway connection and the nested backend login alive, while `ServerAliveInterval` detects a dead path. A non-visual standard SSH jump would require the infrastructure owner to expose a verified backend SSH route or a GateShell-supported non-interactive command interface; add that route and its identity checks to `.agents/config.yaml` before using it.

Use GitHub for source synchronization. Use the configured external-storage client for models, checkpoints, datasets, and results once that transfer workflow is requested and verified.

## Operating boundaries

- Inspect remote state before claiming a job is running, complete, or failed.
- Never edit Git-tracked project source remotely. Diagnose remotely, then fix and validate locally before deploying through Git.
- A login or brief diagnostic check needs no run record. For important computation, create or update `.agents/runs/<run-id>.yaml` with its purpose, source revision, remote location, command, PID/session, log, artifacts, status, and next action.
