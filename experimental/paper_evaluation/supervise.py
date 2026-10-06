"""Monitor every shard, adopt existing processes, and resume bounded failures."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError,ProcessLookupError):return None

def run_workers(jobs,*,state_path,event_path,stage,completion,poll_seconds=2,max_restarts=2):
    """A job may specify pid to adopt; only dead, incomplete jobs are restarted."""
    records=[]
    def event(value):
        with Path(event_path).open('a') as f:f.write(json.dumps(dict(at=now(),**value))+'\n')
    def spawn(record):
        handle=Path(record['log']).open('a')
        process=subprocess.Popen(record['command'],env=record['env'],stdout=handle,stderr=subprocess.STDOUT)
        record.update(pid=process.pid,process=process,identity=process_identity(process.pid),handle=handle,state='running')
        event(dict(action='started',shard=record['shard'],pid=process.pid,restarts=record['restarts']))
    for job in jobs:
        record=dict(job,process=None,handle=None,restarts=0,state='running')
        pid=record.get('pid');identity=process_identity(pid) if pid else None
        if identity is not None:
            argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if argv!=record['command']:raise ValueError(f'Cannot adopt PID {pid}: command mismatch')
            record['identity']=identity
            event(dict(action='adopted',shard=record['shard'],pid=pid))
        elif completion(record):record['state']='complete'
        else:spawn(record)
        records.append(record)
    while True:
        failure=None
        for record in records:
            if record['state']=='complete':continue
            process=record['process'];code=process.poll() if process else None
            alive=(code is None and process_identity(record['pid'])==record['identity'])
            if alive:continue
            if record['handle']:record['handle'].close();record['handle']=None
            if completion(record):
                record['state']='complete';event(dict(action='complete',shard=record['shard'],pid=record['pid'],exit_code=code));continue
            event(dict(action='incomplete_exit',shard=record['shard'],pid=record['pid'],exit_code=code))
            if record['restarts']>=max_restarts:
                record['state']='failed';failure=RuntimeError(f"{stage} shard {record['shard']} failed after {max_restarts} restarts");break
            record['restarts']+=1;spawn(record)
        state='failed' if failure else 'complete' if all(r['state']=='complete' for r in records) else 'running'
        write_json(state_path,dict(state=state,stage=stage,pid=os.getpid(),updated_at=now(),workers=[{k:r.get(k) for k in ('shard','pid','command','log','state','restarts')} for r in records]))
        if failure:raise failure
        if state=='complete':return
        time.sleep(poll_seconds)
