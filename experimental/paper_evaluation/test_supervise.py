import json
import os
import subprocess
import sys
from .supervise import run_workers


def test_adopt_healthy_worker_and_resume_failed_worker(tmp_path):
    good=tmp_path/'good';bad=tmp_path/'bad';marker=tmp_path/'once'
    healthy_command=[sys.executable,'-c',f'import time; from pathlib import Path; time.sleep(.3); Path({str(good)!r}).touch()']
    healthy=subprocess.Popen(healthy_command)
    retry_code=f'from pathlib import Path; import sys; p=Path({str(marker)!r}); first=not p.exists(); p.touch(); sys.exit(7) if first else Path({str(bad)!r}).touch()'
    jobs=[dict(shard=0,pid=healthy.pid,command=healthy_command,env=dict(os.environ),log=str(tmp_path/'good.log')),
          dict(shard=1,command=[sys.executable,'-c',retry_code],env=dict(os.environ),log=str(tmp_path/'bad.log'))]
    run_workers(jobs,state_path=tmp_path/'state.json',event_path=tmp_path/'events.jsonl',stage='test',completion=lambda j:(good if j['shard']==0 else bad).exists(),poll_seconds=.02)
    healthy.wait()
    state=json.loads((tmp_path/'state.json').read_text())
    assert state['state']=='complete'
    assert state['workers'][0]['pid']==healthy.pid
    assert state['workers'][0]['restarts']==0
    assert state['workers'][1]['restarts']==1


def test_permanent_failure_has_bounded_retries(tmp_path):
    import pytest
    job=dict(shard=0,command=[sys.executable,'-c','raise SystemExit(9)'],env=dict(os.environ),log=str(tmp_path/'fail.log'))
    with pytest.raises(RuntimeError,match='after 2 restarts'):
        run_workers([job],state_path=tmp_path/'state.json',event_path=tmp_path/'events.jsonl',stage='test',completion=lambda j:False,poll_seconds=.02)
    state=json.loads((tmp_path/'state.json').read_text())
    assert state['state']=='failed'
    assert state['workers'][0]['restarts']==2
    events=[json.loads(x) for x in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert len([x for x in events if x['action']=='started'])==3
