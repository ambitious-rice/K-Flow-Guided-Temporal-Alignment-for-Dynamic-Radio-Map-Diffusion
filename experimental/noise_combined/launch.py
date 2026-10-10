"""Supervise 3 independent 2-GPU torchrun jobs with bounded resource use."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from experimental.noise_loss_screen.common import write
from .model import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();record=Path('.agents/runs')/(root.name+'.yaml');jobs={};handles=[];start=time.time()
    for variant in VARIANTS:
        path=root/variant;spec=json.loads((path/'config.json').read_text())
        spec.update(started_at=start,train_deadline=start+spec['train_hours']*3600,hard_deadline=start+spec['total_hours']*3600);write(path/'config.json',spec)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=spec['gpus'],PYTHONPATH='src:.',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','experimental.noise_combined.train','--root',str(path)]
        if a.smoke:command.append('--smoke')
        log=path/('smoke.log' if a.smoke else 'train.log');handle=log.open('a');handles.append(handle)
        proc=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[variant]=dict(proc=proc,spec=spec,command=command,log=str(log),state='running')
    def snapshot():
        return {name:dict(pid=j['proc'].pid,gpus=j['spec']['gpus'],command=j['command'],log=j['log'],state=j['state'],exit_code=j['proc'].poll()) for name,j in jobs.items()}
    write(root/('smoke_jobs.json' if a.smoke else 'jobs.json'),snapshot())
    d=json.loads(record.read_text());d.update(status='smoke_running' if a.smoke else 'running',jobs=snapshot(),started_at=start);write(record,d)
    while any(j['state']=='running' for j in jobs.values()):
        for name,j in jobs.items():
            if j['state']!='running':continue
            proc=j['proc'];path=root/name
            if proc.poll() is not None:
                j['state']='complete' if proc.returncode==0 else 'failed'
                if proc.returncode==0 and not a.smoke:
                    state=json.loads((path/'training/status.json').read_text());j['state']=state['state']
                    result=subprocess.run([sys.executable,'-m','experimental.noise_paper_injection.analyze','--root',str(path)],timeout=60)
                    if result.returncode:j['state']='analysis_failed'
            elif time.time()>(start+1800 if a.smoke else j['spec']['hard_deadline']):
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=30)
                j['state']='ready_to_resume' if (path/'training/last.pt').exists() else 'time_limit'
                status_path=path/'training/status.json'
                if not a.smoke and status_path.exists():
                    saved=json.loads(status_path.read_text());saved.update(state=j['state'],reason='hard_time_limit',resume_checkpoint=str(path/'training/last.pt'),note='Saved checkpoint may precede last logged step; final evaluation may be partial');write(status_path,saved)
        write(root/('smoke_pipeline.json' if a.smoke else 'pipeline.json'),dict(started_at=start,updated_at=time.time(),jobs=snapshot()))
        if any(j['state']=='running' for j in jobs.values()):time.sleep(10)
    for h in handles:h.close()
    if not a.smoke:
        subprocess.run([sys.executable,'-m','experimental.noise_combined.summarize','--root',str(root)],timeout=60)
    d=json.loads(record.read_text());d.update(status='smoke_finished' if a.smoke else 'finished',finished_at=time.time(),jobs=snapshot(),next='Inspect each arm status and paired confirmation; completion alone is not success');write(record,d)


if __name__=='__main__':main()
