"""Bounded screen -> frozen candidate -> matched confirmation, all local GPUs."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experimental.noise_loss_screen.common import write
from .run import ROOT,OLD,CANDIDATES


def spawn(args,gpu,log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:.',OMP_NUM_THREADS='4')
    command=[sys.executable,'-u','-m','experimental.noise_fusion_ddim.run']+args
    handle=open(log,'a');proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env);handle.close()
    return proc,dict(pid=proc.pid,gpu=gpu,command=command,log=str(log))

def main():
    started=time.time();jobs=[];procs=[]
    try:
        for i,c in enumerate(CANDIDATES):
            p,j=spawn(['--candidate',str(i)],i,ROOT/f'{c["name"]}.log');procs.append(p);jobs.append(j)
        write(ROOT/'pipeline.json',dict(state='screening',started_at=started,jobs=jobs))
        for p in procs:
            if p.wait()!=0:raise RuntimeError(f'Screen worker{p.pid} failed; inspect logs')
        scores=[]
        for c in CANDIDATES:
            d=json.loads((ROOT/c['name']/'matrix.json').read_text());assert d['complete'] and len(d['rows'])==128
            r=d['rows'];correct=[x for x in r if x['input_sigma']==x['sigma']]
            fixed={str(s):float(np.mean([x['metrics']['unobserved_mse'] for x in r if x['input_sigma']==s])) for s in [0,.03,.05,.09]}
            scores.append(dict(candidate=c,correct_mse=float(np.mean([x['metrics']['unobserved_mse'] for x in correct])),fixed_mse=fixed,best_fixed_sigma=float(min(fixed,key=fixed.get))))
        chosen=min(scores,key=lambda x:x['correct_mse']);write(ROOT/'selected.json',dict(**chosen,selection='Minimum tune8 actual-sigma unseen MSE; all candidate configurations predeclared; no confirmation used',all_candidates=scores))
        procs=[];jobs=[]
        for gpu in range(8):
            p,j=spawn(['--stage','confirm','--part',str(gpu),'--parts','8'],gpu,ROOT/f'confirm{gpu}.log');procs.append(p);jobs.append(j)
        write(ROOT/'pipeline.json',dict(state='confirming',started_at=started,candidate=chosen,jobs=jobs))
        for p in procs:
            if p.wait()!=0:raise RuntimeError(f'Confirmation worker{p.pid} failed; inspect logs')
        subprocess.run([sys.executable,'-m','experimental.noise_fusion_ddim.analyze'],check=True)
        write(ROOT/'pipeline.json',dict(state='complete',started_at=started,finished_at=time.time()))
    except Exception as exc:
        write(ROOT/'pipeline.json',dict(state='failed',error=str(exc),jobs=jobs,updated_at=time.time()));raise
if __name__=='__main__':main()
