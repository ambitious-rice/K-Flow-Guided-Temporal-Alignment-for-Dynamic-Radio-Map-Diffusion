"""Run two prespecified auxiliary-weight checks after existing workers finish."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--initial',required=True);a=p.parse_args();root=Path(a.root)
    write(root/'supplement_pipeline.json',dict(state='waiting',pid=os.getpid(),started_at=time.time()))
    def worker(gpu,previous,name):
        while True:
            path=root/previous/'status.json'
            done=path.exists() and json.loads(path.read_text()).get('state')=='complete'
            if done:
                used=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
                if not used.stdout.strip():break
            time.sleep(10)
        cmd=[sys.executable,'-u','-m','experimental.noise_loss_screen.run','--root',str(root),'--initial',a.initial,'--variant',name]
        for attempt in range(2):
            with (root/'logs'/f'train_{name}.log').open('a') as log:
                child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                write(root/f'supplement_gpu{gpu}.json',dict(pid=child.pid,command=cmd,variant=name,attempt=attempt,started_at=time.time()))
                code=child.wait()
            if code==0:return
        raise RuntimeError(f'{name} failed')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(worker,6,'off','k2_aux01'),pool.submit(worker,7,'off_nosigma','k2_aux01_nosigma')]
            for f in fs:f.result()
        write(root/'supplement_pipeline.json',dict(state='complete',finished_at=time.time()))
    except Exception as error:
        write(root/'supplement_pipeline.json',dict(state='failed',error=repr(error),updated_at=time.time()));raise

if __name__=='__main__':main()
