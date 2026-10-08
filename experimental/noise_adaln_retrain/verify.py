"""Real-checkpoint checks for warm-start equivalence and blockwise gradients."""
import argparse,gc,json
from pathlib import Path
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_feature_retrain.model import FeatureNoiseModel
from experimental.noise_feature_retrain.train import INITIAL
from experimental.noise_loss_screen.common import public,write
from .model import BlockNoiseModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    torch.manual_seed(20261008);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=load_config('experimental/noise_hvdit/w16_tolerance.yaml')
    state=torch.load(INITIAL,map_location='cpu',weights_only=False)['model']
    entry=json.loads((root/'inputs/validation_bank.json').read_text())[1]
    d=torch.load(Path(entry['base'])/entry['file'],weights_only=True)
    b={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()}
    x=d['initial'].cuda();t=torch.tensor([999],device='cuda')
    model=FeatureNoiseModel(cfg,'film').initialize(state).cuda().eval()
    with torch.no_grad():reference=model(x,t,public(b,torch.tensor([.03**2],device='cuda'))).cpu()
    del model;gc.collect();torch.cuda.empty_cache()
    model=BlockNoiseModel(cfg).initialize(state).cuda().eval()
    inp=public(b,torch.tensor([.03**2],device='cuda'))
    with torch.no_grad():
        output=model(x,t,inp);torch.testing.assert_close(output.cpu(),reference,rtol=0,atol=0)
        poison=dict(inp,target=b['target']+100,source_label=torch.ones_like(b['target'])*100)
        torch.testing.assert_close(model(x,t,poison),output,rtol=0,atol=0)
    opt=torch.optim.AdamW(model.block_noise_heads.parameters(),lr=1e-3)
    loss=(model(x,t,inp).float()-b['target']).square().mean();loss.backward()
    gradients={}
    for name,heads in model.block_noise_heads.items():
        gradients[name]=[]
        for head in heads:
            norm=float(head.weight.grad.norm());assert norm>0 and torch.isfinite(head.weight.grad).all()
            gradients[name].append(norm)
    opt.step();opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        y0=model(x,t,public(b,torch.tensor([0.],device='cuda')))
        y9=model(x,t,public(b,torch.tensor([.09**2],device='cuda')))
        response=float((y9-y0).abs().max());assert response>1e-7 and torch.isfinite(y9).all()
    report=dict(initialization_equivalence='bitwise exact',target_source_input_isolation='passed',block_head_gradient_norms=gradients,post_update_sigma_response_max=response,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write(root/'functional_checks.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
