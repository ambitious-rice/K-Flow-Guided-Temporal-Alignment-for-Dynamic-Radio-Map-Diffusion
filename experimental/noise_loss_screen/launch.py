import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from .common import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--initial',required=True);p.add_argument('--where',choices=['local','remote'],required=True);a=p.parse_args()
    root=Path(a.root);(root/'logs').mkdir(exist_ok=True)
    names=['k1','k1_nosigma','k2','k2_nosigma','k3','k3_nosigma','off','off_nosigma'] if a.where=='local' else ['k4','k6']
    write(root/f'{a.where}_pipeline.json',dict(state='running',pid=os.getpid(),variants=names,started_at=time.time()))
    def run(gpu,name,mode='train'):
        status=root/('baseline' if mode=='baseline' else name)/'status.json'
        if status.exists() and json.loads(status.read_text()).get('state')=='complete':return
        command=[sys.executable,'-u','-m','experimental.noise_loss_screen.run','--root',str(root),'--initial',a.initial,'--variant',name,'--mode',mode]
        for attempt in range(2):
            with (root/'logs'/f'{mode}_{name}.log').open('a') as log:
                process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                write(root/f'gpu{gpu}_job.json',dict(pid=process.pid,command=command,variant=name,mode=mode,attempt=attempt,started_at=time.time()))
                code=process.wait()
            if code==0:
                result=json.loads(status.read_text())
                if result['state']!='complete':raise RuntimeError('Worker exited without completion')
                return
        raise RuntimeError(f'{mode}/{name} failed; inspect worker log')
    def worker(gpu,name):
        if a.where=='local' and gpu==0:run(gpu,name,'baseline')
        run(gpu,name)
    errors=[]
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures={pool.submit(worker,g,n):n for g,n in enumerate(names)}
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:
                errors.append(dict(variant=futures[future],error=repr(error)));write(root/f'{a.where}_errors.json',errors)
    if errors:raise RuntimeError(errors)
    if a.where=='remote':
        with tarfile.open(root/'metrics.tar.gz','w:gz') as archive:
            for name in names:
                for path in (root/name).rglob('*'):
                    if path.is_file() and path.suffix in ('.json','.jsonl'):archive.add(path,arcname=str(path.relative_to(root)))
            archive.add(root/'logs',arcname='logs')
        subprocess.run(['aliyunpan','upload','--norapid','--skip','--np',str(root/'metrics.tar.gz'),'/fzj/noise_loss_screen_20261007'],check=True)
    write(root/f'{a.where}_pipeline.json',dict(state='complete',variants=names,finished_at=time.time()))

if __name__=='__main__':main()
