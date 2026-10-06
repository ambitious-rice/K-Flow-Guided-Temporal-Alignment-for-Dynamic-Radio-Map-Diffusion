"""Two remote workers, bounded retries, frozen calibration before test inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from .run import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    (root/'logs').mkdir(exist_ok=True)
    def stage(name):
        def worker(gpu):
            cmd=[sys.executable,'-u','-m','experimental.noise_estimation_iterative.run','--root',str(root),
                 '--stage',name,'--shard',str(gpu),'--shards','2']
            for attempt in range(3):
                with (root/'logs'/f'{name}_{gpu}.log').open('a') as log:
                    process=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                    write(root/f'gpu{gpu}_job.json',dict(stage=name,pid=process.pid,command=cmd,attempt=attempt,started_at=time.time()))
                    code=process.wait()
                if code==0:return
            raise RuntimeError(f'{name} GPU {gpu} failed after three attempts')
        write(root/'pipeline.json',dict(state='running',stage=name,pid=os.getpid(),updated_at=time.time()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,g) for g in range(2)]
            for future in futures:future.result()
    try:
        if not (root/'calibration.json').exists():
            stage('calibration')
            subprocess.run([sys.executable,'-m','experimental.noise_estimation_iterative.run','--root',str(root),'--stage','fit'],check=True)
        stage('test')
        expected=json.loads((root/'inputs/test_bank.json').read_text())
        for e in expected:
            case=json.loads((root/'test/cases'/f'{e["id"]}.json').read_text())
            if case['entry']!=e or len(case['trajectories'])!=3:raise ValueError('Incomplete coverage')
            for trajectory in case['trajectories']:
                if len(trajectory['rows'])!=4:raise ValueError('Incomplete trajectory')
                for row in trajectory['rows']:
                    if not (root/row['prediction']).is_file():raise ValueError('Missing prediction')
        archive=root/'metrics.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for base in ('test','inputs','logs'):
                for path in (root/base).rglob('*'):
                    if path.is_file() and path.suffix in ('.json','.log'):t.add(path,arcname=str(path.relative_to(root)))
            t.add(root/'calibration.json',arcname='calibration.json')
        subprocess.run(['aliyunpan','upload','--norapid','--skip','--np',str(archive),'/fzj/noise_iterative_remote_20261007'],check=True)
        write(root/'pipeline.json',dict(state='complete',cases=len(expected),finished_at=time.time()))
    except BaseException as error:
        write(root/'pipeline_error.json',dict(error=repr(error),at=time.time()));raise


if __name__=='__main__':main()
