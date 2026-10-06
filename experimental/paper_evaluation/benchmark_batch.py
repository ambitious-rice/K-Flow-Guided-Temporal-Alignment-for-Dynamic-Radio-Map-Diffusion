"""Same validation inputs and draws; benchmark fold and final-condition batching."""
import json
import os
from pathlib import Path
import time
import torch
from experimental.paired_evaluation.evaluate import load_model
from experimental.noise_estimation_crossfit.core import ConditionedEnsemble,FP32,input_boundary,masked_conditions,audited_mle
from experimental.noise_estimation_vem.hvdit import FrozenHVDiT,inference_inputs
from rmdm_noise_estimation.calibration import VarianceCalibration
from .randomness import ensemble_initial
from .worker import ROOT,W16,now
from .prepare import write_json

@torch.inference_mode()
def collect(model,engine,inputs,assignment,entry,fold_batch,steps=50):
    batch=input_boundary(inputs['conditions'],inputs['observed_rss'],inputs['sampling_mask'],torch.tensor([.03**2],device='cuda'))
    visible=inputs['sampling_mask'].bool();labels=assignment[visible]
    samples=torch.empty((8,int(visible.sum())),device='cuda')
    torch.cuda.synchronize();begin=time.perf_counter()
    for first in range(0,4,fold_batch):
        folds=list(range(first,min(first+fold_batch,4)))
        hidden=[masked_conditions(batch,assignment,fold) for fold in folds]
        merged={k:torch.cat([h[k] for h in hidden]) for k in batch}
        cache=model.encode_conditions(merged)
        initial=torch.cat([ensemble_initial(inputs['observed_rss'].shape[1:],video_index=entry['full_data_video_index'],start=entry['start'],fold=fold,members=list(range(8))) for fold in folds]).cuda()
        pred=engine._sample_chunk(model,cache,initial,steps=steps,accelerator=FP32())
        for fi,fold in enumerate(folds):
            selected=assignment[0]==fold
            samples[:,labels==fold]=pred[fi*8:(fi+1)*8,selected]
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    samples=samples.cpu();return samples,seconds

@torch.inference_mode()
def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261006)
    bankpath=ROOT/'calibration_bank/bank.json';bank=json.loads(bankpath.read_text());entry=bank['entries'][4]
    data=torch.load(bankpath.parent/entry['file'],weights_only=True)
    sparse={k:v.cuda() if torch.is_tensor(v) else v for k,v in data['sparse'].items()}
    assignment=torch.load(bankpath.parent/entry['fold_file'],weights_only=True).cuda()
    inputs=dict(conditions=inference_inputs(sparse),observed_rss=sparse['observed_rss'],sampling_mask=sparse['sampling_mask'],initial_noise=data['initial'].cuda())
    model,config=load_model('hvdit',*W16);model=model.cuda().eval().requires_grad_(False)
    engine=ConditionedEnsemble(config.diffusion);predictor=FrozenHVDiT(model,config.diffusion,steps=50)
    report=dict(scope='one preselected validation case; same four folds and eight members, no test-based accuracy selection',entry=entry,started_at=now(),device=torch.cuda.get_device_name(),precision='fp32',steps=50,fold_batches=[],final_batches=[])
    cal=VarianceCalibration(**json.loads((ROOT/'calibration.json').read_text())['calibrators']['mixed']['variance_calibration'])
    reference=None;reference_sigma=None
    for fb in (1,2,4):
        collect(model,engine,inputs,assignment,entry,fb,steps=2)
        for repeat in range(2):
            torch.cuda.reset_peak_memory_stats();samples,seconds=collect(model,engine,inputs,assignment,entry,fb)
            mean=samples.double().mean(0).float();var=samples.double().var(0,unbiased=True).float()
            estimate=audited_mle(sparse['observed_rss'][sparse['sampling_mask'].bool()].cpu().numpy(),mean.numpy(),cal.apply(var.numpy()))
            if reference is None:reference=samples;reference_sigma=estimate['sigma']
            row=dict(fold_batch=fb,model_batch=fb*8,repeat=repeat,seconds=seconds,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9,max_sample_difference=float((samples-reference).abs().max()),sigma=estimate['sigma'],sigma_difference=abs(estimate['sigma']-reference_sigma))
            report['fold_batches'].append(row);write_json(ROOT/'batch_benchmark.json',report);print(json.dumps(row),flush=True)
    qs=[0.,.0009,.0025,.0081,reference_sigma**2]
    reference_final=None
    for group in (1,5):
        for repeat in range(2):
            torch.cuda.synchronize();begin=time.perf_counter();outputs=[]
            for first in range(0,5,group):
                values=qs[first:first+group];n=len(values)
                outputs.append(predictor(conditions={k:v.repeat_interleave(n,dim=0) for k,v in inputs['conditions'].items()},observed_rss=inputs['observed_rss'].repeat_interleave(n,dim=0),sampling_mask=inputs['sampling_mask'].repeat_interleave(n,dim=0),initial_noise=inputs['initial_noise'].repeat_interleave(n,dim=0),noise_var=torch.tensor(values,device='cuda')))
            torch.cuda.synchronize();seconds=time.perf_counter()-begin;pred=torch.cat(outputs).cpu()
            if reference_final is None:reference_final=pred
            row=dict(condition_batch=group,repeat=repeat,seconds=seconds,max_prediction_difference=float((pred-reference_final).abs().max()))
            report['final_batches'].append(row);write_json(ROOT/'batch_benchmark.json',report);print(json.dumps(row),flush=True)
    report['finished_at']=now();write_json(ROOT/'batch_benchmark.json',report)
if __name__=='__main__':main()
