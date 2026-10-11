"""Poll the designated remote pilot and collect its completed results locally."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

RECORDS = Path('/data_p6/fzj/projects/RMDM/.agents/runs')
POLL_SECONDS = 30
TIMEOUT_SECONDS = 24 * 60 * 60


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.monitor-{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def log(event, **fields):
    print(json.dumps(dict(time=now(), event=event, **fields)), flush=True)


def update_record(path, **changes):
    # Re-read for every update so deployment metadata added after startup survives.
    record = json.loads(path.read_text())
    record.update(changes)
    write(path, record)


def snapshot(remote, timeout):
    script = '''import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def read(path):
    if not path.exists():
        return None
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"unparsed_text": text}
workers = {}
for phase in ("calibration", "evaluation"):
    for path in sorted((root / phase).glob("worker*_status.json")):
        workers[str(path.relative_to(root))] = read(path)
print(json.dumps({"export": read(root / "export.json"),
                  "jobs": read(root / "jobs.json"),
                  "export_failure": read(root / "export_failure.json"),
                  "export_failure_marker": read(root / "export_failure"),
                  "workers": workers}))
'''
    command = shlex.join([remote.get('python', 'python3'), '-c', script, remote['root']])
    result = subprocess.run([*remote['ssh'], command], capture_output=True, text=True,
                            timeout=timeout, check=True)
    return json.loads(result.stdout)


def collect(root, remote, exported, remaining):
    destination = root/'remote_results'
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/'results.tar.gz'
    cloud = remote['cloud'].rstrip('/') + '/results.tar.gz'
    log('download_started', cloud=cloud, destination=str(destination))
    # A previous interrupted download must not be mistaken for the complete file.
    if archive.exists():
        archive.rename(destination/f'results.interrupted-{time.time_ns()}.tar.gz')
    with (root/'download_results.log').open('a') as output:
        subprocess.run(['aliyunpan', 'download', '--np', '--saveto', str(destination), cloud],
                       stdout=output, stderr=subprocess.STDOUT, check=True,
                       timeout=max(1, remaining))
    if not archive.is_file():
        raise FileNotFoundError(f'Download did not produce {archive}')
    expected_bytes = exported.get('bytes')
    if expected_bytes is not None and archive.stat().st_size != int(expected_bytes):
        raise ValueError(f'Archive size differs: {archive.stat().st_size} != {expected_bytes}')
    log('download_complete', archive=str(archive), bytes=archive.stat().st_size)
    with tarfile.open(archive, 'r:gz') as bundle:
        bundle.extractall(destination, filter='data')
    # Import only after collection; summarize uses CPU metrics and no GPU worker.
    from .run import summarize
    log('summarize_started', root=str(destination))
    summarize(destination)
    report = destination/'tables/report.json'
    if json.loads(report.read_text()).get('state') != 'complete':
        raise ValueError('Local report did not reach complete state')
    return report


def monitor(root):
    record_path = RECORDS/f'{root.name}.yaml'
    record = json.loads(record_path.read_text())
    remote = record['remote']
    if not isinstance(remote['ssh'], list) or not remote['ssh']:
        raise ValueError('Run record remote.ssh must contain an SSH argument list')
    deadline = time.monotonic() + TIMEOUT_SECONDS
    log('monitor_started', root=str(root), remote_root=remote['root'], timeout_hours=24)
    while time.monotonic() < deadline:
        try:
            live = snapshot(remote, timeout=min(60, max(1, deadline-time.monotonic())))
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            log('poll_error', error=str(error))
            write(root/'live_remote.json', dict(checked_at=now(), state='poll_error', error=str(error)))
        else:
            live['checked_at'] = now()
            write(root/'live_remote.json', live)
            jobs = live.get('jobs') or {}
            exported = live.get('export') or {}
            failure = live.get('export_failure')
            marker = live.get('export_failure_marker')
            if (failure is not None or marker is not None or jobs.get('state') == 'failed'
                    or exported.get('state') in ('failed', 'export_failure')):
                detail = dict(export_failure=failure, export_failure_marker=marker, jobs=jobs, export=exported)
                update_record(record_path, status='failed', finished_at=now(), failure=detail,
                              next='Inspect remote launch/export logs before retrying.')
                log('remote_failed', detail=detail)
                return 1
            log('poll', phase=jobs.get('phase'), state=jobs.get('state'),
                export_state=exported.get('state'), workers=live.get('workers'))
            if exported.get('state') == 'uploaded':
                try:
                    report = collect(root, remote, exported, deadline-time.monotonic())
                except Exception as error:
                    update_record(record_path, status='failed', finished_at=now(),
                                  failure=dict(stage='local_collection', error=str(error)),
                                  next='Inspect download_results.log and local remote_results; rerun monitor after repair.')
                    log('collection_failed', error=str(error))
                    raise
                update_record(record_path, status='complete', finished_at=now(), report=str(report),
                              remote_export=exported, next='Review local pilot metrics and estimator comparisons.')
                log('complete', report=str(report))
                return 0
        time.sleep(min(POLL_SECONDS, max(0, deadline-time.monotonic())))
    update_record(record_path, status='failed', finished_at=now(),
                  failure=dict(stage='monitor', error='24 hour monitor timeout'),
                  next='Check remote task state; monitor timeout did not stop remote workers.')
    log('timeout', timeout_hours=24)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(monitor(args.root.resolve()))


if __name__ == '__main__':
    main()
