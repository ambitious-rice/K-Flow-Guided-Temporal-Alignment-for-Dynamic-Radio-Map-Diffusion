"""Two GPU work queue; small baselines first, RadioDiff in balanced chunks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tarfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline-source',required=True);p.add_argument('--weights-root',required=True);args=p.parse_args()
    root=Path(args.root);(root/'logs').mkdir(exist_ok=True);jobs=queue.Queue()
    checkpoints={'radiounet':'radiounet_second','rmegan':'rmegan_local','radiodiff':'radiodiff'}
    for method,n in [('radiounet',2),('rmegan',2),('radiodiff',16)]:
        for shard in range(n):jobs.put((method,shard,n))
    def worker(gpu):
        while True:
            try:method,shard,n=jobs.get_nowait()
            except queue.Empty:return
            cmd=[sys.executable,'-u','experimental/paper_evaluation/remote_baselines.py','--bank',str(root/'remote_inputs/bank.json'),'--output',str(root/method),'--method',method,'--checkpoint',str(Path(args.weights_root)/checkpoints[method]/'best.pt'),'--baseline-source',args.baseline_source,'--shard',str(shard),'--shards',str(n),'--frame-batch','16']
            with (root/'logs'/f'{method}_{shard}.log').open('a') as log:
                process=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                (root/f'gpu{gpu}_job.json').write_text(json.dumps(dict(state='running',method=method,shard=shard,pid=process.pid,command=cmd)))
                code=process.wait()
            if code:raise RuntimeError(f'{method} shard {shard} failed, exit {code}')
        
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=[pool.submit(worker,g) for g in range(2)]
        for result in results:result.result()
    # Metrics and logs are small; full predictions stay on the compute server.
    with tarfile.open(root/'remote_metrics.tar.gz','w:gz') as archive:
        for method in checkpoints:
            for path in (root/method).rglob('*.json'):archive.add(path,arcname=str(path.relative_to(root)))
        archive.add(root/'logs',arcname='logs')
    subprocess.run(['aliyunpan','upload','--norapid','--skip','--np',str(root/'remote_metrics.tar.gz'),'/fzj/paper_test200_20261006'],check=True)
    (root/'remote_complete.json').write_text(json.dumps(dict(state='complete',finished_at=time.time(),methods=list(checkpoints))))
if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if '--root' in sys.argv:
            error_root=Path(sys.argv[sys.argv.index('--root')+1]);error_root.mkdir(parents=True,exist_ok=True)
            (error_root/'remote_error.json').write_text(json.dumps(dict(state='failed',error=repr(error),at=time.time())))
        raise
