"""CPU-only remote queue: wait for an idle GPU, then smoke/benchmark/evaluate.

Does not preempt other users. Stops its own child if competing compute appears;
outputs are then explicitly tainted and must not be reported as clean timing.
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def query(fields, kind='gpu'):
    p=subprocess.run(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],
                     check=True,capture_output=True,text=True,timeout=15)
    return [[s.strip() for s in row] for row in csv.reader(io.StringIO(p.stdout)) if row]


def occupied():
    return {uuid for uuid,pid in query('gpu_uuid,pid','compute-apps')}


def foreign_processes(uuid, child_pid):
    result=[]
    for gpu,pid in query('gpu_uuid,pid','compute-apps'):
        if gpu!=uuid: continue
        try:
            if os.getpgid(int(pid)) != child_pid:
                result.append(int(pid))
        except ProcessLookupError:
            pass
    return result


def stop(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wait-hours',type=float,default=24.)
    parser.add_argument('--run-hours',type=float,default=4.)
    parser.add_argument('--cloud')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'tainted.json').exists():
        raise RuntimeError('Existing timing is tainted; review before selecting a fresh output directory')
    lock=(args.out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    state=dict(pid=os.getpid(),started_at=now(),source_revision=revision,command=sys.argv,
               parent=str(args.parent),out=str(args.out))
    def status(name,**kw):
        state.update(state=name,updated_at=now(),**kw);write(args.out/'queue_status.json',state)
    status('waiting_for_idle_gpu')
    deadline=time.monotonic()+args.wait_hours*3600
    consecutive={};selected=None;gpu_lock=None
    while time.monotonic()<deadline:
        busy=occupied();gpus=query('uuid,name,utilization.gpu,memory.used')
        # Prefer Ada when both are available. Once chosen, every stage stays on that GPU.
        for uuid,name,util,memory in sorted(gpus,key=lambda x:'5880' not in x[1]):
            idle=uuid not in busy and int(util)<=5 and int(memory)<1024
            consecutive[uuid]=consecutive.get(uuid,0)+1 if idle else 0
            if consecutive[uuid]>=3:
                candidate=Path('/tmp',f'rmdm_efficiency_{uuid}.lock').open('w')
                try: fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close();continue
                selected=(uuid,name);gpu_lock=candidate;break
        status('waiting_for_idle_gpu',resources=gpus,idle_checks=consecutive)
        if selected: break
        time.sleep(30)
    if not selected:
        status('resource_wait_timeout');return
    uuid,name=selected
    status('starting',gpu_uuid=uuid,gpu_name=name)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH='src:.',OMP_NUM_THREADS='4',
             MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    started=time.monotonic()
    for stage in ('smoke','benchmark','run'):
        cmd=[sys.executable,'-u','-m','experimental.noise_adaptive.efficiency',stage,
             '--parent',str(args.parent),'--out',str(args.out)]
        if uuid in occupied():
            status('resource_conflict_before_stage',stage=stage);return
        with (args.out/f'{stage}.log').open('a') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status('running',stage=stage,child_pid=child.pid,stage_command=cmd)
            try:
                while child.poll() is None:
                    foreign=foreign_processes(uuid,child.pid)
                    if foreign:
                        stop(child)
                        write(args.out/'tainted.json',dict(reason='competing GPU compute',pids=foreign,at=now()))
                        status('stopped_resource_contention',foreign_pids=foreign);return
                    if time.monotonic()-started>args.run_hours*3600:
                        stop(child);status('run_timeout');return
                    status('running',stage=stage,child_pid=child.pid)
                    time.sleep(5)
            except BaseException:
                stop(child);raise
            if child.returncode:
                status('failed',stage=stage,exit_code=child.returncode);return
    assert (args.out/'complete.json').exists()
    status('complete',child_pid=None)
    if args.cloud:
        archive=args.out/'w4_efficiency_results.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in args.out.iterdir():
                if path.name in (archive.name,'queue.lock','queue.log'): continue
                tar.add(path,arcname=path.name)
        command=['aliyunpan','upload',str(archive),args.cloud]
        with (args.out/'upload.log').open('a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        write(args.out/'export.json',dict(state='uploaded' if result.returncode==0 else 'failed',
            command=command,cloud=args.cloud+'/'+archive.name,bytes=archive.stat().st_size,at=now()))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if '--out' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--out')+1])/'queue_status.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            state.update(state='failed',error=repr(exc),updated_at=now())
            write(path,state)
        raise
