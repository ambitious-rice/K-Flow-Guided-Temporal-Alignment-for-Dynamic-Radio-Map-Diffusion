"""Four two-GPU jobs; stop training at7.25h, hard-stop own jobs at8.75h."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from experimental.noise_loss_screen.common import write
from .model import VARIANTS

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root).resolve();record=Path('.agents/runs/noise_feature_retrain_20261008.yaml')
    start=time.time();spec=json.loads((root/'config.json').read_text());spec.update(started_at=start,train_deadline=start+7.25*3600,hard_deadline=start+8.75*3600);write(root/'config.json',spec)
    jobs=[];procs=[]
    for index,variant in enumerate(VARIANTS):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=f'{index*2},{index*2+1}',PYTHONPATH='src:.',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','experimental.noise_feature_retrain.train','--root',str(root),'--variant',variant]
        log=root/f'{variant}.log'
        with log.open('a') as handle:proc=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(proc);jobs.append(dict(variant=variant,pid=proc.pid,gpus=env['CUDA_VISIBLE_DEVICES'],command=command,log=str(log)))
    write(root/'jobs.json',jobs)
    if record.exists():
        d=json.loads(record.read_text());d.update(status='running',started_at_epoch=start,jobs=jobs,training=spec);write(record,d)
    reason=None
    while True:
        states=[p.poll() for p in procs]
        write(root/'pipeline.json',dict(state='running',started_at=start,elapsed=time.time()-start,exit_codes=states,jobs=jobs,train_deadline=spec['train_deadline'],hard_deadline=spec['hard_deadline']))
        if all(x is not None for x in states):
            reason='complete' if all(x==0 for x in states) else 'failed';break
        if any(x not in (None,0) for x in states):reason='failed';break
        if time.time()>=spec['hard_deadline']:reason='time_limit';break
        time.sleep(15)
    if reason!='complete':
        for proc in procs:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        time.sleep(5)
        for proc in procs:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
    else:
        try:
            result=subprocess.run([sys.executable,'-m','experimental.noise_feature_retrain.analyze','--root',str(root)],timeout=max(1,spec['hard_deadline']-time.time()))
            if result.returncode:reason='analysis_failed'
        except subprocess.TimeoutExpired:
            reason='analysis_time_limit'
    write(root/'pipeline.json',dict(state=reason,started_at=start,finished_at=time.time(),elapsed=time.time()-start,exit_codes=[p.poll() for p in procs],jobs=jobs))
    if record.exists():
        d=json.loads(record.read_text());d.update(status=reason,finished_at_epoch=time.time(),next='Review matched training outcomes, actual-sigma response, best fixed input and blind baseline; no success assumed',analysis=str(root/'analysis.json'));write(record,d)
if __name__=='__main__':main()
