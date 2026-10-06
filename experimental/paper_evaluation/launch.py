"""Eight GPU local stages with checked subprocess exits and persistent progress."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .worker import ROOT,fit,now
from .prepare import write_json

def stage(mode,bank,output):
    procs=[]
    for gpu in range(8):
        command=[sys.executable,'-u','-m','experimental.paper_evaluation.worker','--mode',mode,'--bank',str(bank),'--output',str(output),'--shard',str(gpu),'--shards','8']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',PYTHONPATH='src:.')
        log=ROOT/'logs'/f'{mode}_{gpu}.log';handle=log.open('a');process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
        procs.append((process,handle,command,str(log)))
    write_json(ROOT/'local_pipeline.json',dict(state='running',stage=mode,updated_at=now(),pid=os.getpid(),workers=[dict(pid=p.pid,command=c,log=l) for p,h,c,l in procs]))
    codes=[p.wait() for p,h,c,l in procs]
    for p,h,c,l in procs:h.close()
    if any(codes):raise RuntimeError(f'{mode} worker exit codes {codes}; see logs')

def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    try:
        stage('calibration',ROOT/'calibration_bank/bank.json',ROOT/'calibration')
        print(json.dumps(fit()),flush=True)
        # Same saved inputs and starts for all comparison rows.
        stage('main',ROOT/'bank/bank.json',ROOT/'main')
        stage('w1',ROOT/'bank/bank.json',ROOT/'w1')
        stage('original',ROOT/'bank/bank.json',ROOT/'original')
        write_json(ROOT/'local_pipeline.json',dict(state='complete',updated_at=now(),pid=os.getpid()))
    except BaseException as error:
        write_json(ROOT/'local_pipeline_error.json',dict(state='failed',error=repr(error),updated_at=now(),pid=os.getpid()));raise
if __name__=='__main__':main()
