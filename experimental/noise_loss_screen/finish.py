import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

def main():
    root=Path('runs/noise_loss_screen_20261007');remote='/data_16T_137/fzj/RMDM/runs/noise_loss_screen_20261007'
    ssh=['ssh','-p','2137','-o','BatchMode=yes','-o','ConnectTimeout=10','fzj@10.11.113.168']
    while True:
        local=json.loads((root/'local_pipeline.json').read_text()) if (root/'local_pipeline.json').exists() else {}
        try:
            r=subprocess.run(ssh+[f'cat {remote}/remote_pipeline.json'],capture_output=True,text=True,timeout=25)
            distant=json.loads(r.stdout) if r.returncode==0 else {}
        except (subprocess.TimeoutExpired,json.JSONDecodeError):distant={}
        (root/'finalizer_status.json').write_text(json.dumps(dict(state='waiting',local=local,remote=distant,updated_at=time.time()),indent=2)+'\n')
        if local.get('state')=='complete' and distant.get('state')=='complete':break
        time.sleep(30)
    destination=root/'download';destination.mkdir(exist_ok=True)
    subprocess.run(['/data_p6/fzj/bin/aliyunpan','download','--nocheck','--np','--saveto',str(destination),'/fzj/noise_loss_screen_20261007/metrics.tar.gz'],check=True)
    archive=list(destination.rglob('metrics.tar.gz'));assert len(archive)==1
    with tarfile.open(archive[0]) as t:t.extractall(root/'remote_results',filter='data')
    subprocess.run([sys.executable,'-m','experimental.noise_loss_screen.analyze','--root',str(root)],check=True)
    (root/'finalizer_status.json').write_text(json.dumps(dict(state='complete',updated_at=time.time()),indent=2)+'\n')
    p=Path('.agents/runs/noise_loss_screen_20261007.yaml');record=json.loads(p.read_text());record.update(status='complete',finished_at=time.time(),next_action='Review all validation variants and confirmation results; no test tuning.');record['analysis']=str(root/'analysis');p.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
