"""Local detached result retrieval and analysis after remote completion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/noise_iterative_remote_20261007');a=p.parse_args();root=Path(a.root)
    remote='/data_16T_137/fzj/RMDM/runs/noise_iterative_remote_20261007'
    ssh=['ssh','-p','2137','-o','BatchMode=yes','-o','ConnectTimeout=10','fzj@10.11.113.168']
    while True:
        try:
            result=subprocess.run(ssh+[f'if test -f {remote}/pipeline_error.json; then cat {remote}/pipeline_error.json; exit 2; fi; cat {remote}/pipeline.json'],capture_output=True,text=True,timeout=25)
            if result.returncode==2:raise RuntimeError(result.stdout)
            state=json.loads(result.stdout) if result.returncode==0 else {'state':'unreachable'}
        except (subprocess.TimeoutExpired,json.JSONDecodeError):state={'state':'unreachable'}
        (root/'finalizer_status.json').write_text(json.dumps(dict(state='waiting',remote=state,updated_at=time.time()),indent=2)+'\n')
        if state.get('state')=='complete':break
        time.sleep(30)
    destination=root/'download';destination.mkdir(exist_ok=True)
    subprocess.run(['/data_p6/fzj/bin/aliyunpan','download','--nocheck','--np','--saveto',str(destination),'/fzj/noise_iterative_remote_20261007/metrics.tar.gz'],check=True)
    archives=list(destination.rglob('metrics.tar.gz'))
    if len(archives)!=1:raise ValueError('Expected metrics archive')
    output=root/'results';output.mkdir(exist_ok=True)
    with tarfile.open(archives[0]) as t:t.extractall(output,filter='data')
    subprocess.run([sys.executable,'-m','experimental.noise_estimation_iterative.summarize','--root',str(output)],check=True)
    (root/'finalizer_status.json').write_text(json.dumps(dict(state='complete',output=str(output),updated_at=time.time()),indent=2)+'\n')
    recordpath=Path('.agents/runs/noise_iterative_remote_20261007.yaml')
    record=json.loads(recordpath.read_text());record.update(status='complete',finished_at=time.time(),next_action='Inspect paired accuracy, reconstruction and runtime by iteration; do not select hyperparameters on these test results.')
    record['artifacts']['results']=str(output);recordpath.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
