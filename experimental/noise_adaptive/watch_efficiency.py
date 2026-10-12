"""Local CPU monitor for a requested remote run; collect selected cloud results."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

import yaml


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write(path, value):
    tmp=path.with_suffix(path.suffix+'.watch.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def handoff_state(path, state, extra=''):
    text=path.read_text();start='<!-- W4_LIVE_STATE -->';end='<!-- /W4_LIVE_STATE -->'
    block=f'{start}\n实时监控更新：{now()}，状态 `{state}`。{extra}\n{end}'
    if start in text:
        first=text.index(start);last=text.index(end,first)+len(end)
        text=text[:first]+block+text[last:]
    else:
        marker='## 目标与边界'
        text=text.replace(marker,block+'\n\n'+marker)
    path.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();project=args.project
    run=project/'.agents/runs/noise_adaptive_remote_20261011.yaml'
    handoff=project/'.agents/handoffs/noise_estimation.md'
    config=yaml.safe_load((project/'.agents/config.yaml').read_text())
    remote=config['remote']['servers'][config['remote']['default']]['connection']
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p',str(remote['port']),
         remote['username']+'@'+remote['host']]
    d=json.loads(run.read_text());follow=d['w4_efficiency_followup']
    output=follow['output_root'];destination=project/'runs/noise_adaptive_remote_20261011/w4_efficiency_v1'
    destination.mkdir(parents=True,exist_ok=True)
    previous=None;deadline=time.monotonic()+30*3600
    terminal={'failed','resource_wait_timeout','run_timeout','resource_conflict_before_stage','stopped_resource_contention','queue_process_missing'}
    while time.monotonic()<deadline:
        try:
            result=subprocess.run(ssh+['cat '+shlex.quote(output+'/queue_status.json')],
                check=True,capture_output=True,text=True,timeout=30)
            live=json.loads(result.stdout);state=live['state']
            if state not in terminal and state!='complete':
                process=subprocess.run(ssh+[f"ps -p {int(live['pid'])} -o args="],capture_output=True,text=True,timeout=30)
                if process.returncode or 'experimental.noise_adaptive.queue_efficiency' not in process.stdout:
                    state='queue_process_missing';live['state']=state
            d=json.loads(run.read_text());follow=d['w4_efficiency_followup']
            follow.update(status=state,latest_remote_state=live,last_checked_at=now(),local_monitor_pid=os.getpid())
            d['status']='w4_efficiency_'+state;d['last_verified_at']=now();write(run,d)
            write(destination/'latest_remote_state.json',live)
            if state!=previous:
                handoff_state(handoff,state)
                print(now(),state,flush=True);previous=state
            if state in terminal:
                d['next']='Inspect remote stage log and queue_status.json; no new efficiency conclusion is available.'
                write(run,d);return
            if state=='complete':
                export=subprocess.run(ssh+['cat '+shlex.quote(output+'/export.json')],capture_output=True,text=True,timeout=30)
                if export.returncode:
                    time.sleep(60);continue
                exported=json.loads(export.stdout)
                if exported['state']!='uploaded':
                    follow['transfer_status']='remote_upload_failed';write(run,d);return
                with (destination/'download.log').open('a') as log:
                    subprocess.run(['aliyunpan','download','--saveto',str(destination),exported['cloud']],
                                   stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                archive=destination/'w4_efficiency_results.tar.gz'
                if archive.stat().st_size!=exported['bytes']:
                    raise ValueError('Archive size differs from uploaded artifact')
                with tarfile.open(archive,'r:gz') as tar:
                    for member in tar.getmembers():
                        path=(destination/member.name).resolve()
                        if not path.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                            raise ValueError('Unexpected archive member')
                    tar.extractall(destination,filter='data')
                report=json.loads((destination/'report.json').read_text())
                assert report['completed_cases']==report['expected_cases']==32
                assert len(list((destination/'evaluation').glob('*/result.json')))==32
                assert json.loads((destination/'smoke.json').read_text())['state']=='passed'
                assert json.loads((destination/'benchmark.json').read_text())['state']=='complete'
                d=json.loads(run.read_text());follow=d['w4_efficiency_followup']
                follow.update(status='complete_results_returned',transfer_status='downloaded_and_readable',
                    finished_at=now(),report=str(destination/'report.json'),export=exported,
                    summary=[r for r in report['summary'] if r['true_sigma']=='all'])
                d['status']='complete';d['next']='Review W4 efficiency report and per-sigma results before adopting any candidate.'
                write(run,d)
                rows=['\n\n|方案|σ MAE|估计秒|最终W16秒|重建MSE|相对known MSE变化|',
                      '|---|---:|---:|---:|---:|---:|']
                for r in follow['summary']:
                    rows.append(f"|{r['variant']}|{r['sigma_mae']:.7f}|{r['estimator_seconds']:.3f}|{r['final_w16_seconds']:.3f}|{r['reconstruction']['mse']:.9f}|{r['mse_change_vs_known_percent']:+.4f}%|")
                extra=' 32案例已完成且结果回传；路径 `runs/noise_adaptive_remote_20261011/w4_efficiency_v1/report.json`。仍仅8个独立视频、σ域[0,.09]。'+'\n'.join(rows)+'\n'
                handoff_state(handoff,'complete_results_returned',extra)
                print(now(),'complete_results_returned',flush=True);return
        except Exception as exc:
            print(now(),repr(exc),flush=True)
            d=json.loads(run.read_text());d['w4_efficiency_followup']['monitor_last_error']=dict(at=now(),error=repr(exc));write(run,d)
        time.sleep(60)
    d=json.loads(run.read_text());d['w4_efficiency_followup']['local_monitor_status']='timeout_30h';write(run,d)


if __name__=='__main__':
    main()
