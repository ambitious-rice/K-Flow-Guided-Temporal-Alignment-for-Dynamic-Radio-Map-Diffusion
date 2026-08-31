#!/usr/bin/env python3
"""Safely maintain one local tmux-backed GateShell operator terminal per server.

This helper intentionally manages only the local terminal which holds an SSH login.
It never starts, stops, or edits work on the remote backend.  Long-running remote
jobs must use their own remote tmux sessions and project run records.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on the invoking machine
    raise SystemExit("remote_session.py requires PyYAML (python3-yaml).") from exc


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / ".agents" / "config.yaml"
PREFIX = "rmdm-gateway-"
PASSWORD_PROMPT = re.compile(r"(?:password|密码)\s*:", re.IGNORECASE)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def command(*args: str, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, input=stdin, text=True, capture_output=True, check=check)


def load_server(server_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text())
        server = data["remote"]["servers"][server_name]
        connection = server["connection"]
    except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError) as exc:
        fail(f"cannot load configured server {server_name!r}: {exc}")
    if connection.get("type") != "gateshell":
        fail(f"{server_name!r} is not a GateShell connection")
    required = ("host", "port", "username", "password", "target")
    missing = [key for key in required if not connection.get(key)]
    if missing:
        fail(f"connection for {server_name!r} is missing: {', '.join(missing)}")
    return server, connection


def session_name(server_name: str, suffix: str = "") -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", server_name):
        fail("server name contains unsupported tmux characters")
    if suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", suffix):
        fail("session suffix contains unsupported tmux characters")
    return f"{PREFIX}{server_name}" + (f"-{suffix}" if suffix else "")


def has_session(name: str) -> bool:
    return command("tmux", "has-session", "-t", name, check=False).returncode == 0


def capture(name: str, lines: int = 160) -> str:
    result = command("tmux", "capture-pane", "-p", "-J", "-t", name, "-S", f"-{lines}")
    return result.stdout


def show_capture(name: str) -> None:
    print(capture(name), end="")


def final_visible_line(pane: str) -> str:
    """Return the last non-empty line from the current terminal display."""
    for line in reversed(pane.splitlines()):
        if line.strip():
            return line
    return ""


def require_route(host: str) -> None:
    route = command("ip", "route", "get", host)
    if "tailscale0" not in route.stdout:
        fail(f"route to {host} does not use tailscale0; inspect Tailscale before connecting")


def open_session(server_name: str, suffix: str = "") -> None:
    _, connection = load_server(server_name)
    name = session_name(server_name, suffix)
    if has_session(name):
        print(f"reusing local operator session {name}")
        show_capture(name)
        return
    require_route(str(connection["host"]))
    ssh = [
        "ssh",
        "-tt",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "ControlMaster=no",
        "-p",
        str(connection["port"]),
        f"{connection['username']}@{connection['host']}",
    ]
    command("tmux", "new-session", "-d", "-s", name, "--", *ssh)
    time.sleep(1)
    if not has_session(name):
        fail("SSH exited before presenting a prompt; inspect the Tailscale route and gateway")
    print(f"opened local operator session {name}")
    show_capture(name)


def authenticate(server_name: str, suffix: str = "") -> None:
    _, connection = load_server(server_name)
    name = session_name(server_name, suffix)
    if not has_session(name):
        fail(f"{name} is not open; run the open command first")
    pane = capture(name)
    if not PASSWORD_PROMPT.search(final_visible_line(pane)):
        fail("no password prompt is visible; refusing to send a credential")
    buffer_name = f"{name}-credential"
    loaded = command("tmux", "load-buffer", "-b", buffer_name, "-", stdin=str(connection["password"]))
    if loaded.returncode != 0:
        fail("could not stage credential in tmux")
    try:
        command("tmux", "paste-buffer", "-d", "-b", buffer_name, "-t", name)
        command("tmux", "send-keys", "-t", name, "Enter")
    finally:
        command("tmux", "delete-buffer", "-b", buffer_name, check=False)
    time.sleep(1)
    show_capture(name)


def send_literal(name: str, text: str, enter: bool = False) -> None:
    command("tmux", "send-keys", "-l", "-t", name, text)
    if enter:
        command("tmux", "send-keys", "-t", name, "Enter")


def select_backend(server_name: str, suffix: str = "") -> None:
    server, _ = load_server(server_name)
    name = session_name(server_name, suffix)
    if not has_session(name):
        fail(f"{name} is not open")
    backend = server.get("gateshell_backend") or {}
    selector = backend.get("selector_command")
    expected_address = backend.get("expected_address")
    if not selector or not expected_address:
        fail("gateshell_backend.selector_command and expected_address must be configured")
    pane = capture(name)
    if expected_address not in pane:
        fail(f"expected backend address {expected_address} is not visible in GateShell; inspect with capture")
    if "[GateShell]$" in pane:
        send_literal(name, str(selector), enter=True)
        time.sleep(1)
        show_capture(name)
        return
    send_literal(name, ":")
    time.sleep(0.25)
    send_literal(name, str(selector))
    time.sleep(0.25)
    send_literal(name, "", enter=True)
    time.sleep(1)
    show_capture(name)


def verify_backend(server_name: str, suffix: str = "") -> None:
    server, _ = load_server(server_name)
    name = session_name(server_name, suffix)
    if not has_session(name):
        fail(f"{name} is not open")
    backend = server.get("gateshell_backend") or {}
    expected = (backend.get("expected_hostname"), backend.get("expected_username"), backend.get("expected_initial_directory"))
    if not all(expected):
        fail("expected hostname, username, and initial directory must be configured")
    marker = "__RMDM_IDENTITY__"
    send_literal(name, f"printf '{marker}\\n'; hostname; id -un; pwd; date -Is", enter=True)
    time.sleep(1)
    pane = capture(name)
    marker_at = pane.rfind(marker)
    observed = pane[marker_at:] if marker_at >= 0 else pane
    if not all(value in observed for value in expected):
        print(observed, end="")
        fail("backend identity did not match configured hostname/user/directory")
    print(observed, end="")
    print("backend identity verified")


def execute(server_name: str, remote_command: str, suffix: str = "") -> None:
    name = session_name(server_name, suffix)
    if not has_session(name):
        fail(f"{name} is not open")
    send_literal(name, remote_command, enter=True)


def list_sessions() -> None:
    result = command("tmux", "list-sessions", "-F", "#{session_name}\t#{session_activity}", check=False)
    if result.returncode != 0:
        return
    for line in result.stdout.splitlines():
        if line.startswith(PREFIX):
            print(line)


def gc_sessions(max_idle_hours: float) -> None:
    if max_idle_hours <= 0:
        fail("max idle hours must be positive")
    now = time.time()
    result = command("tmux", "list-sessions", "-F", "#{session_name}\t#{session_activity}", check=False)
    if result.returncode != 0:
        return
    for line in result.stdout.splitlines():
        try:
            name, activity = line.split("\t", 1)
            idle_seconds = now - float(activity)
        except ValueError:
            continue
        if name.startswith(PREFIX) and idle_seconds >= max_idle_hours * 3600:
            command("tmux", "kill-session", "-t", name)
            print(f"recycled idle operator session {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("open", "authenticate", "backend", "verify", "capture", "close"):
        item = subparsers.add_parser(action)
        item.add_argument("server")
        item.add_argument("--suffix", default="")
    execute_parser = subparsers.add_parser("exec")
    execute_parser.add_argument("server")
    execute_parser.add_argument("--suffix", default="")
    execute_parser.add_argument("remote_command", nargs=argparse.REMAINDER)
    subparsers.add_parser("sessions")
    gc_parser = subparsers.add_parser("gc")
    gc_parser.add_argument("--max-idle-hours", type=float, required=True)
    args = parser.parse_args()

    if args.action == "open":
        open_session(args.server, args.suffix)
    elif args.action == "authenticate":
        authenticate(args.server, args.suffix)
    elif args.action == "backend":
        select_backend(args.server, args.suffix)
    elif args.action == "verify":
        verify_backend(args.server, args.suffix)
    elif args.action == "capture":
        name = session_name(args.server, args.suffix)
        if not has_session(name):
            fail(f"{name} is not open")
        show_capture(name)
    elif args.action == "close":
        name = session_name(args.server, args.suffix)
        if has_session(name):
            command("tmux", "kill-session", "-t", name)
            print(f"closed local operator session {name}")
    elif args.action == "exec":
        remote_command = " ".join(args.remote_command).strip()
        if not remote_command:
            fail("exec needs a remote command after --")
        execute(args.server, remote_command, args.suffix)
    elif args.action == "sessions":
        list_sessions()
    elif args.action == "gc":
        gc_sessions(args.max_idle_hours)


if __name__ == "__main__":
    main()
