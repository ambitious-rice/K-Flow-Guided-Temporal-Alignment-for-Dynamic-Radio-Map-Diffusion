"""Real-data initialization, IID replay, sigma gradients and label isolation."""
import argparse
import gc
import json
from pathlib import Path
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_loss_screen.common import public, write
from experimental.noise_scratch_full.data import FullData
from experimental.noise_scratch_full.loss import loss_terms
from rmdm.diffusion import DiffusionProcess
from .data import CombinedData
from .model import CombinedNoiseDiT, VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['initial','smoke'],required=True)
    a=p.parse_args();root=Path(a.root);reports={}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='initial':
        reference=Path(json.loads((root/'protocol.json').read_text())['reference'])
        payload=torch.load(reference/'training/checkpoints/step000000.pt',map_location='cpu',weights_only=False)
        original=payload['model'];del payload
    for variant in VARIANTS:
        path=root/variant;spec=json.loads((path/'config.json').read_text());cfg=load_config(path/'model_config.yaml')
        torch.manual_seed(spec['seed']);model=CombinedNoiseDiT(cfg,variant)
        if a.stage=='initial':
            actual=model.state_dict()
            for k,v in original.items():
                assert torch.equal(v,actual[k]),(variant,k)
            reports[variant]=dict(common_parameters_bitwise_equal=sum(v.numel() for v in original.values()))
            baseline=FullData(path/'inputs',spec['seed']).batch(9,0,0,8)
            candidate=CombinedData(path/'inputs',spec['seed'],variant).batch(9,0,0,8)
            if variant=='hybrid_iid':
                for k in baseline:assert torch.equal(baseline[k],candidate[k]),k
                reports[variant]['iid_data_replay']='all tensors bitwise equal'
            else:
                for k in ['target','sampling_mask','diffusion_noise','timesteps','sampling_rate']:
                    assert torch.equal(candidate[k],baseline[k][torch.arange(2).repeat_interleave(4)])
                reports[variant]['paired_data']='real targets/masks/t/noise matched within4;8independent draws perglobal32'
            b={k:v[:1].cuda() for k,v in baseline.items()}
        else:
            saved=torch.load(path/f"smoke_b{spec['microbatch']}_a{spec['accumulation']}/last.pt",map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);del saved
            b={k:v.cuda() for k,v in FullData(path/'inputs',spec['seed']).batch(9,0,0,1).items()}
        model=model.cuda().eval()
        noisy=DiffusionProcess(cfg.diffusion).scheduler.add_noise(b['target'],b['diffusion_noise'],b['timesteps'])
        inp=public(b,b['measurement_variance'])
        if a.stage=='smoke':
            with torch.no_grad():
                expected=model(noisy,b['timesteps'],inp)[0]
                poison=model(noisy,b['timesteps'],dict(inp,target=b['target']+100,source_label=b['source_label']+100))[0]
                torch.testing.assert_close(expected,poison,rtol=0,atol=0)
            pred,cal=model(noisy,b['timesteps'],inp);loss,_=loss_terms(pred,cal,b);loss.backward()
            grads={}
            for stem_name in ['input_stem','condition_stem']:
                stem=getattr(model.denoiser,stem_name)
                grad=stem.observation_projection.weight.grad.view(stem.observation_projection.out_features,2,3,16)[:,:,2]
                assert torch.isfinite(grad).all() and grad.norm()>0
                grads[stem_name+'_map']=float(grad.norm())
            if model.hybrid:
                for name,heads in model.sigma_blocks.items():
                    norms=[float(h.weight.grad.norm()) for h in heads]
                    assert all(n>0 and n<float('inf') for n in norms),(name,norms)
                    grads[name+'_sigma_heads']=norms
                model.zero_grad(set_to_none=True)
                # Isolate HWM calibration path: it must receive explicit sigma,
                # independently of the noise-map path through the DiT.
                cache=model.encode_conditions(inp)
                (cache['cal']-b['target']).square().mean().backward()
                norm=sum(float(p.grad.square().sum()) for p in model.sigma_embedding.parameters() if p.grad is not None)**.5
                assert 0<norm<float('inf');grads['hwm_to_sigma_embedding']=norm
            reports[variant]=dict(target_source_isolation='bitwise exact',gradients=grads,loss_finite=bool(torch.isfinite(loss)))
            del pred,cal,loss,expected,poison
            if model.hybrid:del cache
        else:
            # Initial extra modulation is exactly zero: cached HWM and forward
            # agree across all arms on the same real input.
            with torch.no_grad():
                result,cal=model(noisy,b['timesteps'],inp)
                if variant==VARIANTS[0]:
                    first=result.cpu();first_cal=cal.cpu()
                else:
                    torch.testing.assert_close(result.cpu(),first,rtol=0,atol=0)
                    torch.testing.assert_close(cal.cpu(),first_cal,rtol=0,atol=0)
                reports[variant]['initial_forward_and_hwm']='bitwise equal'
            del result,cal,actual,baseline,candidate
        del model,b,noisy,inp;gc.collect();torch.cuda.empty_cache()
    write(root/(a.stage+'_checks.json'),reports)
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':main()
