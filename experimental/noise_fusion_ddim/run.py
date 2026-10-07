"""Frozen-weight internal-sigma screening under complete original DDIM-50."""
import argparse,json,os,time
from pathlib import Path
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_hvdit.model import NoiseHVDiT
from experimental.noise_loss_screen.common import write,save,metrics
from .fusion import NoiseWeightedFusion,sample

ROOT=Path('runs/noise_fusion_ddim_20261007')
OLD=Path('runs/noise_loss_screen_20261007')
FRESH=Path('runs/noise_condition_diagnosis_20261007/fresh_inputs')
INITIAL=Path('runs/noise_iterative_remote_20261007/inputs/w16_model.pt')
CANDIDATES=[dict(film_mode=film,prior_std=tau,name=f'{film}_tau{tau:g}') for film in ['same','zero'] for tau in [.01,.03,.06,.1]]

def load():
    torch.manual_seed(20261007);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=load_config('experimental/noise_hvdit/w16_tolerance.yaml');model=NoiseHVDiT(cfg).cuda().eval()
    model.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['model']);return cfg,model

def bank(stage):
    entries=[dict(e,base=str(OLD/'inputs'),bank='original') for e in json.loads((OLD/'inputs/validation_bank.json').read_text()) if e['group']==('tune' if stage=='screen' else 'confirm')]
    if stage=='confirm':entries +=[dict(e,base=str(FRESH),bank='fresh') for e in json.loads((FRESH/'bank.json').read_text())]
    return entries

@torch.inference_mode()
def run(candidate,stage,part,parts,limit=0):
    cfg,model=load();fusion=NoiseWeightedFusion(candidate['prior_std']).cuda().eval()
    name=candidate['name'] if stage=='screen' else f'confirm_part{part}'
    out=ROOT/name;out.mkdir(exist_ok=True);status=dict(candidate=candidate,stage=stage,pid=os.getpid(),gpu=torch.cuda.get_device_name(),started_at=time.time());write(out/'status.json',dict(status,state='running'))
    file=out/'matrix.json';rows=json.loads(file.read_text())['rows'] if file.exists() else [];done={(r['bank'],r['file'],r['input_sigma']) for r in rows}
    entries=bank(stage)[part::parts]
    if limit:entries=entries[:limit]
    for e in entries:
        d=torch.load(Path(e['base'])/e['file'],weights_only=True);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()};initial=d['initial'].cuda()
        for sigma in sorted({0,.03,.05,.09,e['sigma']}):
            if (e['bank'],e['file'],sigma) in done:continue
            physical={k:v for k,v in b.items() if k in ['building','vehicle','observed_rss','sampling_mask','sampling_rate']};physical['measurement_variance']=torch.tensor([sigma**2],device='cuda')
            torch.cuda.synchronize();tick=time.perf_counter();pred=sample(model,cfg.diffusion,physical,initial,fusion=fusion,film_mode=candidate['film_mode'],steps=50);torch.cuda.synchronize()
            assert torch.isfinite(pred).all();row=dict(**e,input_sigma=sigma,metrics=metrics(pred,b),seconds=time.perf_counter()-tick);rows.append(row)
            if stage=='confirm':save(out/'predictions'/f"{e['bank']}_{Path(e['file']).stem}_sigma{sigma:g}.pt",pred.cpu())
        write(file,dict(complete=False,candidate=candidate,rows=rows));write(out/'status.json',dict(status,state='running',rows=len(rows),updated_at=time.time()))
    write(file,dict(complete=True,candidate=candidate,rows=rows));write(out/'status.json',dict(status,state='complete',rows=len(rows),finished_at=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,default=0);p.add_argument('--stage',choices=['screen','confirm'],default='screen');p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    candidate=CANDIDATES[a.candidate] if a.stage=='screen' else json.loads((ROOT/'selected.json').read_text())['candidate'];run(candidate,a.stage,a.part,a.parts,a.limit)
if __name__=='__main__':main()
