import argparse,csv,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from experimental.noise_loss_screen.common import write
from .statistics import SIGMAS,assess,merge_stats


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(root,extra_root=None):
    spec=json.loads((root/'protocol.json').read_text());videos=spec['videos']
    grouped=defaultdict(list);window_values={};case_count=0;seconds=[]
    starts=[0,48]+([80] if extra_root else [])
    files=list((root/'results').glob('worker*/cases/*.json'))
    if extra_root:
        extra=json.loads((extra_root/'protocol.json').read_text())
        assert extra['videos']==videos and extra['seed']==spec['seed'] and extra['starts']==[80]
        assert extra['weights']==spec['weights'] and extra['checkpoint']==spec['checkpoint']
        files.extend((extra_root/'results').glob('worker*/cases/*.json'))
    for file in sorted(files):
        d=json.loads(file.read_text());case_count+=1;seconds.append(d['seconds'])
        for r in d['rows']:
            k=(r['video_id'],r['rate'],r['sigma'],r['input_sigma'])
            grouped[k].append(r)
            wk=(r['video_id'],r['start'],r['rate'],r['sigma'],r['input_sigma'])
            assert wk not in window_values
            window_values[wk]=merge_stats([r['stats']])['unobserved_mse']
    assert case_count==len(videos)*len(starts)*12 and len(grouped)==28800
    merged={};long_rows=[]
    for k,rows in sorted(grouped.items()):
        assert len(rows)==len(starts) and {r['start'] for r in rows}==set(starts)
        metrics=merge_stats([r['stats'] for r in rows]);merged[k]=metrics
        long_rows.append(dict(video_id=k[0],rate=k[1],sigma=k[2],input_sigma=k[3],**metrics))
    out=root/('tables_three_windows' if extra_root else 'tables');out.mkdir(exist_ok=True)
    csv_write(out/'merged_metrics.csv',long_rows)
    all_matrices={};summary_rows=[]
    for video in videos:
        matrices={str(rate):[[merged[(video,rate,s,t)]['unobserved_mse'] for t in SIGMAS] for s in SIGMAS] for rate in [1,2,3]}
        matrices['combined']=np.mean([matrices[str(r)] for r in [1,2,3]],axis=0).tolist()
        assessments={key:assess(m) for key,m in matrices.items()}
        windows={str(start):{str(rate):assess([[window_values[(video,start,rate,s,t)] for t in SIGMAS] for s in SIGMAS]) for rate in [1,2,3]} for start in starts}
        all_matrices[video]=dict(matrices=matrices,assessment=assessments,windows=windows)
        row=dict(video_id=video,scene=video.split('/')[0],episode=video.rsplit('/',1)[0])
        for key,a in assessments.items():
            row.update({f'{key}_'+k:a[k] for k in ['pass_count','all_four','correct_mse','gain_vs_best_fixed','best_fixed_sigma']})
        row['pass_count_all12']=sum(assessments[str(r)]['pass_count'] for r in [1,2,3])
        row['all12']=row['pass_count_all12']==12
        for start in starts:row[f'window{start}_pass_count_all12']=sum(windows[str(start)][str(r)]['pass_count'] for r in [1,2,3])
        summary_rows.append(row)
    csv_write(out/'video_summary.csv',summary_rows);write(out/'video_matrices.json',all_matrices)
    def report(subset):
        result=dict(videos=len(subset),episodes=len({v.rsplit('/',1)[0] for v in subset}),by_rate={})
        for key in ['1','2','3','combined']:
            arr=np.array([all_matrices[v]['matrices'][key] for v in subset]);items=[all_matrices[v]['assessment'][key] for v in subset]
            result['by_rate'][key]=dict(pass_count_distribution={str(i):sum(a['pass_count']==i for a in items) for i in range(5)},
                all_four=sum(a['all_four'] for a in items),all_four_fraction=float(np.mean([a['all_four'] for a in items])),
                mean_matrix=arr.mean(0).tolist(),mean_matrix_assessment=assess(arr.mean(0)),
                best_input_distribution={str(s):{str(t):sum(a['row_min_input'][i]==t for a in items) for t in SIGMAS} for i,s in enumerate(SIGMAS)},
                correct_is_minimum_by_sigma={str(s):sum(a['pass_by_sigma'][i] for a in items) for i,s in enumerate(SIGMAS)},
                relative_gap_quantiles={str(s):dict(zip(['p0','p25','p50','p75','p90','p100'],np.quantile([a['relative_gap'][i] for a in items],[0,.25,.5,.75,.9,1]).tolist())) for i,s in enumerate(SIGMAS)})
        counts=[sum(all_matrices[v]['assessment'][str(r)]['pass_count'] for r in [1,2,3]) for v in subset]
        result['all12']=sum(c==12 for c in counts);result['pass_count_all12_distribution']={str(i):counts.count(i) for i in range(13)}
        return result
    correct=[r for r in long_rows if r['sigma']==r['input_sigma']]
    metric_names=list(merged[next(iter(merged))])
    def avg(rows):return {m:float(np.mean([r[m] for r in rows])) for m in metric_names}
    result=dict(complete=True,checkpoint=spec['checkpoint'],weights='ema',seed=spec['seed'],starts=starts,cases=case_count,
        window_rows=case_count*4,merged_rows=len(long_rows),all_videos=report(videos),
        by_scene={scene:report([v for v in videos if v.startswith(scene+'/')]) for scene in sorted({v.split('/')[0] for v in videos})},
        correct_sigma_metrics=avg(correct),correct_sigma_metrics_by_rate={str(r):avg([x for x in correct if x['rate']==r]) for r in [1,2,3]},
        correct_sigma_metrics_by_noise={str(s):avg([x for x in correct if x['sigma']==s]) for s in SIGMAS},
        fixed_sigma_metrics={str(s):avg([x for x in long_rows if x['input_sigma']==s]) for s in SIGMAS},
        seconds_all_inputs_median=float(np.median(seconds)),total_worker_inference_hours=sum(seconds)/3600,
        notes=['Window MSE pooled bypixelcounts within video;aggregate rates andvideos equally','PSNR computed from merged fullMSE;mean_frame_psnr also available','Tied minima count as success with numerical tolerance1e-12;argmin distribution picks first column in ties','No temporal edge is formed between disjointwindows','All600videos retained inmetrics;selectedlists separate'])
    write(out/'report.json',result)
    for key in ['1','2','3','combined']:
        selected=[r for r in summary_rows if r[key+'_all_four']]
        write(out/f'selected_{key}.json',dict(count=len(selected),videos=[r['video_id'] for r in selected]))
    write(out/'selected_all12.json',dict(count=sum(r['all12'] for r in summary_rows),videos=[r['video_id'] for r in summary_rows if r['all12']]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(14,3),sharey=True)
    for ax,key in zip(axes,['1','2','3','combined']):
        counts=result['all_videos']['by_rate'][key]['pass_count_distribution'];ax.bar(range(5),[counts[str(i)] for i in range(5)])
        ax.set_title('Combined rates' if key=='combined' else f'{key}% sampling');ax.set_xticks(range(5));ax.set_xlabel('Correct-sigma minima (of 4)')
    axes[0].set_ylabel('Videos');fig.tight_layout();fig.savefig(out/'distribution.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(videos=600,all12=result['all_videos']['all12'],per_rate={k:result['all_videos']['by_rate'][k]['all_four'] for k in ['1','2','3','combined']},correct_sigma_metrics=result['correct_sigma_metrics'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--extra-root');a=p.parse_args();summarize(Path(a.root),Path(a.extra_root) if a.extra_root else None)
