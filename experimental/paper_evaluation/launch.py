"""Eight GPU stages; continuously monitor workers and preserve frozen calibration."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
from .worker import ROOT,fit,now
from .prepare import write_json
from .supervise import run_workers


def stage(mode,bank,output,adopted=None):
    adopted={i:w for i,w in enumerate(adopted or [])};jobs=[]
    entries=json.loads(Path(bank).read_text())['entries']
    for gpu in range(8):
        command=[sys.executable,'-u','-m','experimental.paper_evaluation.worker','--mode',mode,'--bank',str(bank),'--output',str(output),'--shard',str(gpu),'--shards','8']
        jobs.append(dict(shard=gpu,command=command,pid=adopted.get(gpu,{}).get('pid'),
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',PYTHONPATH='src:.'),
            log=str(ROOT/'logs'/f'{mode}_{gpu}.log')))
    def complete(job):
        for entry in entries[job['shard']::8]:
            path=Path(output)/'cases'/f"{Path(entry['file']).stem}.json"
            if not path.exists():return False
            record=json.loads(path.read_text())
            if record['entry']!=entry:raise ValueError('Saved case identity differs from manifest')
            if mode=='calibration':
                if not (Path(output)/'units'/f"{Path(entry['file']).stem}.pt").exists():return False
            else:
                expected=5 if mode=='main' else 1
                if len(record['rows'])!=expected or any(not Path(r['prediction']).exists() for r in record['rows']):return False
        return True
    run_workers(jobs,state_path=ROOT/'local_pipeline.json',event_path=ROOT/'worker_events.jsonl',stage=mode,completion=complete)


def main():
    p=argparse.ArgumentParser();p.add_argument('--adopt-main');p.add_argument('--legacy-controller',type=int);args=p.parse_args()
    (ROOT/'logs').mkdir(exist_ok=True)
    try:
        if args.adopt_main:
            if not (ROOT/'calibration.json').exists():raise ValueError('Frozen calibration missing')
            stage('main',ROOT/'bank/bank.json',ROOT/'main',json.loads(Path(args.adopt_main).read_text())['workers'])
            if args.legacy_controller:
                # Its original GPU children have now finished; close only the paused controller.
                for sig in (signal.SIGTERM,signal.SIGCONT):
                    try:os.kill(args.legacy_controller,sig)
                    except ProcessLookupError:pass
        else:
            if not (ROOT/'calibration.json').exists():
                stage('calibration',ROOT/'calibration_bank/bank.json',ROOT/'calibration');print(json.dumps(fit()),flush=True)
            stage('main',ROOT/'bank/bank.json',ROOT/'main')
        stage('w1',ROOT/'bank/bank.json',ROOT/'w1')
        stage('original',ROOT/'bank/bank.json',ROOT/'original')
        write_json(ROOT/'local_pipeline.json',dict(state='complete',updated_at=now(),pid=os.getpid()))
    except BaseException as error:
        write_json(ROOT/'local_pipeline_error.json',dict(state='failed',error=repr(error),updated_at=now(),pid=os.getpid()));raise
if __name__=='__main__':main()
