"""Export supplemental results via cloud and create an independent V2 locally."""
import argparse,json,shlex,subprocess,sys,tarfile,time
from pathlib import Path
from experimental.noise_loss_screen.common import write
from .summarize import summarize


def export(root,cloud):
    spec=json.loads((root/'protocol.json').read_text())
    jobs=json.loads((root/'jobs.json').read_text());assert all(j['state']=='complete' for j in jobs.values())
    files=list((root/'results').glob('worker*/cases/*.json'));assert len(files)==spec['expected_cases']==7200
    dest=root/'supplement_results.tar.gz'
    with tarfile.open(dest,'w:gz') as tar:
        for name in ['protocol.json','manifest.json','jobs.json','results']:
            tar.add(root/name,arcname=name)
    subprocess.run(['aliyunpan','upload',str(dest),cloud],check=True)
    write(root/'remote_export.json',dict(state='uploaded',cloud=cloud+'/supplement_results.tar.gz',cases=len(files),bytes=dest.stat().st_size))


def collect(root,remote_root,cloud):
    status=root/'supplement_collection.json';started=time.time();dest=root/'remote_w80';dest.mkdir(exist_ok=True)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-p','2137','fzj@10.11.113.168']
    while time.time()-started<30*3600:
        try:
            result=subprocess.run(ssh+['cat '+shlex.quote(str(remote_root/'remote_export.json'))],capture_output=True,text=True,timeout=30)
            done=result.returncode==0 and json.loads(result.stdout).get('state')=='uploaded'
        except (subprocess.TimeoutExpired,ValueError):done=False
        if done:break
        write(status,dict(state='waiting_remote',updated_at=time.time()));time.sleep(60)
    else:raise TimeoutError('Supplement not exported within30h;localV1 unaffected')
    subprocess.run(['aliyunpan','download','--np','--saveto',str(dest),cloud+'/supplement_results.tar.gz'],check=True)
    archive=dest/'supplement_results.tar.gz'
    if not archive.exists():
        matches=list(dest.rglob('supplement_results.tar.gz'))
        assert len(matches)==1, matches
        archive=matches[0]
    with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
    assert len(list((dest/'results').glob('worker*/cases/*.json')))==7200
    while not (root/'tables/report.json').exists():
        if time.time()-started>30*3600:raise TimeoutError('WaitinglocalV1')
        write(status,dict(state='waiting_local',updated_at=time.time()));time.sleep(60)
    summarize(root,dest)
    write(status,dict(state='complete',finished_at=time.time(),local_v1=str(root/'tables/report.json'),combined_v2=str(root/'tables_three_windows/report.json')))
    record=Path('.agents/runs')/(root.name+'.yaml')
    d=json.loads(record.read_text());d['remote_supplement'].update(status='complete',collected_at=time.time(),v2=str(root/'tables_three_windows/report.json'));write(record,d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['export','collect']);p.add_argument('--root',required=True);p.add_argument('--cloud',required=True);p.add_argument('--remote-root');a=p.parse_args()
    if a.mode=='export':export(Path(a.root),a.cloud)
    else:collect(Path(a.root),Path(a.remote_root),a.cloud)
