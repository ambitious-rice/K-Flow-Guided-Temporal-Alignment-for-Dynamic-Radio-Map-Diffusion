"""Four local GPUs with bounded training/evaluation and persistent supervision."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from experimental.noise_loss_screen.common import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve();record=Path('.agents/runs')/(root.name+'.yaml')
    start=time.time();spec=json.loads((root/'config.json').read_text());spec.update(started_at=start,train_deadline=start+spec['train_hours']*3600,hard_deadline=start+spec.get('total_hours',8.75)*3600);write(root/'config.json',spec)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=spec['gpus'],PYTHONPATH='src:.',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','experimental.noise_scratch_full.train','--root',str(root)]
    with (root/'train.log').open('a') as f:proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    job=dict(pid=proc.pid,gpus=spec['gpus'],command=command,log=str(root/'train.log'));write(root/'job.json',job)
    d=json.loads(record.read_text());d.update(status='running',started_at_epoch=start,training_config=spec,job=job);write(record,d)
    while proc.poll() is None and time.time()<spec['hard_deadline']:
        write(root/'pipeline.json',dict(state='running',started_at=start,elapsed=time.time()-start,job=job));time.sleep(15)
    reason='complete' if proc.poll()==0 else 'failed' if proc.poll() is not None else 'time_limit'
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        time.sleep(5)
        if proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        proc.wait(timeout=30)
        if (root/'training'/'last.pt').exists():
            reason='ready_to_resume'
            state_path=root/'training'/'status.json'
            state=json.loads(state_path.read_text())
            state.update(state=reason,reason='hard_time_limit',resume_checkpoint=str(root/'training'/'last.pt'),note='Resume at saved checkpoint step, which may precede the last logged step')
            write(state_path,state)
    if reason=='complete':
        training_status=json.loads((root/'training'/'status.json').read_text())
        reason=training_status['state']
        try:
            result=subprocess.run([sys.executable,'-m','experimental.noise_scratch_full.analyze','--root',str(root)],timeout=max(1,spec['hard_deadline']-time.time()))
            if result.returncode:reason='analysis_failed'
        except subprocess.TimeoutExpired:reason='analysis_time_limit'
    write(root/'pipeline.json',dict(state=reason,started_at=start,finished_at=time.time(),elapsed=time.time()-start,exit_code=proc.poll(),job=job))
    d=json.loads(record.read_text());d.update(status=reason,finished_at_epoch=time.time(),next='Inspect loss components, raw/EMA sigma-intervention curves, best alignment and final confirmation. Completion of budget does not imply convergence.');write(record,d)
if __name__=='__main__':main()
