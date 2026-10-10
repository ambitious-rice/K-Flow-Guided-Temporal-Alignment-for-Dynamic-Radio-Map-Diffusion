"""Four factorial cells with historical baseline; tune-frozen comparisons."""
import argparse
import json
from pathlib import Path
from experimental.noise_loss_screen.common import write
from experimental.noise_paper_injection.summarize import comparison
from .model import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);root=Path(p.parse_args().root)
    baseline=Path(json.loads((root/'protocol.json').read_text())['reference'])
    arms={'map_iid_reference':baseline, **{v:root/v for v in VARIANTS}}
    records=[];milestones={}
    for variant,path in arms.items():
        for file in sorted((path/'training').glob('*_matrix.json')):
            d=json.loads(file.read_text())
            if not d.get('complete'):continue
            constant=d['selection']['best_fixed_sigma']
            records.append(dict(variant=variant,criterion=file.stem,selection=d['selection'],
                rates={str(rate):comparison([r for r in d['rows'] if rate is None or r['rate']==rate],constant)
                       for rate in [None,1,2,3]}))
        for file in sorted((path/'training').glob('*_repeat*.json')):
            d=json.loads(file.read_text())
            records.append(dict(variant=variant,criterion=file.stem,repeat=d['repeat'],
                confirmation=comparison(d['rows'],d['selection']['best_fixed_sigma'])))
        milestones[variant]={}
        for file in sorted((path/'training/validation').glob('step*.json')):
            d=json.loads(file.read_text())
            if 'correct_mse' in d:
                milestones[variant][file.stem]={k:d[k] for k in ['step','weights','correct_mse','alignment_gain','row_min_input','mean_response']}
    common=sorted(set.intersection(*(set(m) for m in milestones.values()))) if milestones else []
    write(root/'comparison.json',dict(results=records,common_step_tune={s:{v:milestones[v][s] for v in arms} for s in common},
        caveats=['Historical baseline, verified common initialization andIID sequence; compare common updates andcompute',
                 'Pairing reduces independent video draws4x at fixedglobal32; paired/stratification treated jointly',
                 'Repeated exploratory validation, not formal test; no causal claim beyond tested bundles']))


if __name__=='__main__':main()
