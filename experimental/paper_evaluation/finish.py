"""Detached finalizer: wait for real completion, fetch remote metrics, build tables."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from .prepare import write_json
from .worker import ROOT,now

SSH=['ssh','-p','2137','-o','BatchMode=yes','-o','ConnectTimeout=10','fzj@10.11.113.168']
REMOTE='/data_16T_137/fzj/RMDM/runs/paper_test200_20261006'
def main():
    while True:
        local=json.loads((ROOT/'local_pipeline.json').read_text())
        remote=subprocess.run(SSH+['test -f '+REMOTE+'/remote_complete.json'],capture_output=True,timeout=30)
        if (ROOT/'local_pipeline_error.json').exists():raise RuntimeError('Local pipeline failed; inspect logs')
        if local['state']=='complete' and remote.returncode==0:break
        write_json(ROOT/'finalizer_status.json',dict(state='waiting',local_stage=local.get('stage',local['state']),remote_complete=remote.returncode==0,updated_at=now()))
        time.sleep(60)
    download=ROOT/'remote_download';download.mkdir(exist_ok=True)
    subprocess.run(['aliyunpan','download','--nocheck','--np','--saveto',str(download),'/fzj/paper_test200_20261006/remote_metrics.tar.gz'],check=True)
    candidates=list(download.rglob('remote_metrics.tar.gz'))
    if len(candidates)!=1:raise RuntimeError('Expected one remote metrics archive')
    remoteout=ROOT/'remote_results';remoteout.mkdir(exist_ok=True)
    with tarfile.open(candidates[0]) as archive:archive.extractall(remoteout,filter='data')
    subprocess.run([sys.executable,'-m','experimental.paper_evaluation.summarize','--root',str(ROOT),'--remote',str(remoteout)],check=True)
    write_json(ROOT/'finalizer_status.json',dict(state='complete',updated_at=now(),tables=str(ROOT/'tables')))
    path=Path('.agents/runs/paper_test200_20261006.yaml');record=json.loads(path.read_text());record.update(status='complete',last_verified_at=now(),next_action='Review the three reconstruction tables and paired noise-condition results; no claim of superiority until analysis.')
    record['artifacts']['tables']=str(ROOT/'tables');path.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
