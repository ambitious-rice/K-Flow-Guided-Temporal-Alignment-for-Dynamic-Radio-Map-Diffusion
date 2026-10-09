"""Check added-channel gradients and target isolation on real smoke weights."""
import argparse,gc,json
from pathlib import Path
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_loss_screen.common import public,write
from experimental.noise_scratch_full.data import FullData
from experimental.noise_scratch_full.loss import loss_terms
from rmdm.diffusion import DiffusionProcess
from .model import PaperNoiseDiT,VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);root=Path(p.parse_args().root);reports={}
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for variant in VARIANTS:
        path=root/variant;spec=json.loads((path/'config.json').read_text());cfg=load_config(path/'model_config.yaml')
        model=PaperNoiseDiT(cfg,variant).cuda().eval();state=torch.load(path/'smoke_b8_a2/last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);del state
        b={k:v.cuda() for k,v in FullData(path/'inputs',spec['seed']).batch(9,0,0,1).items()}
        noisy=DiffusionProcess(cfg.diffusion).scheduler.add_noise(b['target'],b['diffusion_noise'],b['timesteps']);inp=public(b,b['measurement_variance'])
        with torch.no_grad():
            expected=model(noisy,b['timesteps'],inp)[0]
            poison=model(noisy,b['timesteps'],dict(inp,target=b['target']+100,source_label=b['source_label']+100))[0]
            torch.testing.assert_close(expected,poison,rtol=0,atol=0)
        pred,cal=model(noisy,b['timesteps'],inp);loss,_=loss_terms(pred,cal,b);loss.backward();gradients={}
        if variant in ('noise_map','usrnet_hyper'):
            for name in ['input_stem','condition_stem']:
                stem=getattr(model.denoiser,name);g=stem.observation_projection.weight.grad.view(stem.observation_projection.out_features,2,3,16)[:,:,2]
                norm=float(g.norm());assert norm>0 and torch.isfinite(g).all();gradients[name+'_map']=norm
        reports[variant]=dict(target_source_isolation='bitwise exact',extra_map_gradients=gradients,loss_finite=bool(torch.isfinite(loss)))
        del model,b,noisy,inp,pred,cal,loss,expected,poison;gc.collect();torch.cuda.empty_cache()
    write(root/'functional_checks.json',reports);print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
