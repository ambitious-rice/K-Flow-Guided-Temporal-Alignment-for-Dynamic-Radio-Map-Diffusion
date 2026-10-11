"""FP32 full-DDIM50 throughput and numerical parity across inference batch sizes."""
import argparse,json,time
from pathlib import Path
import torch
from experimental.noise_combined.model import CombinedNoiseDiT
from experimental.noise_hvdit.config import load_config
from experimental.noise_loss_screen.common import write
from experimental.noise_video_screen.statistics import SIGMAS,sufficient_stats
from rmdm.diffusion import DDIMSampler
from .batching import build_group
from .run import input_name


@torch.inference_mode()
def run(root,sizes):
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(20261011)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    spec=json.loads((root/'protocol.json').read_text());vi=spec['worker_video_indices'][0][0]
    data=torch.load(root/'inputs'/input_name(vi,80),weights_only=True)
    data={k:v.cuda() if torch.is_tensor(v) else v for k,v in data.items()}
    cfg=load_config(str(root/'model_config.yaml'));model=CombinedNoiseDiT(cfg,'hybrid_paired',20,7.).cuda().eval()
    payload=torch.load(root/'checkpoint.pt',map_location='cpu',mmap=True,weights_only=False);model.load_state_dict(payload['model'],strict=True);del payload
    sampler=DDIMSampler(cfg.diffusion);cases=[(r,s) for r in spec['rates'] for s in SIGMAS]
    inputs,initial,_=build_group(data,cases[:1]);sampler.sample(model,inputs,initial_noise=initial,steps=50,eta=0);torch.cuda.synchronize()
    del inputs,initial
    results=[];reference=None;reference_mse=None
    for size in sizes:
        torch.cuda.reset_peak_memory_stats();preds=[];mse=[];sampling_seconds=0;tick_all=time.perf_counter()
        for offset in range(0,len(cases),size):
            group=cases[offset:offset+size];inputs,initial,batches=build_group(data,group)
            torch.cuda.synchronize();tick=time.perf_counter()
            pred=sampler.sample(model,inputs,initial_noise=initial,steps=50,eta=0)
            torch.cuda.synchronize();sampling_seconds+=time.perf_counter()-tick;assert torch.isfinite(pred).all()
            for j,b in enumerate(batches):
                for k in range(4):
                    st=sufficient_stats(pred[j*4+k:j*4+k+1],b);mse.append(st['unseen_sse']/st['unseen_count'])
            preds.append(pred.cpu());del pred,inputs,initial,batches
        elapsed=time.perf_counter()-tick_all;predictions=torch.cat(preds);values=torch.tensor(mse,dtype=torch.float64)
        if reference is None:reference=predictions;reference_mse=values
        error=(predictions-reference).abs();relative=(values-reference_mse).abs()/reference_mse.clamp_min(1e-12)
        result=dict(cases_per_batch=size,model_batch=4*size,cases=len(cases),model_outputs=48,
            sampling_seconds=sampling_seconds,total_seconds=elapsed,cases_per_second=len(cases)/elapsed,
            seconds_per_case=elapsed/len(cases),peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
            max_abs_output_difference=float(error.max()),max_relative_unobserved_mse_difference=float(relative.max()),
            acceptable=bool(error.max()<1e-5 and relative.max()<1e-4),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))
        result['speedup_vs_batch4']=result['cases_per_second']/results[0]['cases_per_second'] if results else 1.
        results.append(result);write(root/'batch_benchmark.json',dict(state='running',video_index=vi,start=80,results=results));print(json.dumps(result),flush=True)
    best=max((x for x in results if x['acceptable']),key=lambda x:x['cases_per_second'])
    write(root/'batch_benchmark.json',dict(state='complete',video_index=vi,start=80,results=results,recommended_cases_per_batch=best['cases_per_batch']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--sizes',type=int,nargs='+',default=[1,2,4,8,12]);a=p.parse_args();run(Path(a.root),a.sizes)
