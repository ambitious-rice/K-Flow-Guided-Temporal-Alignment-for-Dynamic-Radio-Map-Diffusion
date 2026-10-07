"""Test pure-noise one-step reconstruction and training/inference alignment."""
import argparse,json,time,os
import torch
from diffusers import DDIMScheduler
from experimental.noise_loss_screen.common import public,metrics,write
from .audit import load,fetch,ROOT,OLD

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=[0,1],required=True);a=p.parse_args()
    cfg,model,_=load();model.eval();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    entries=json.loads((OLD/'inputs/validation_bank.json').read_text())[a.part::2];rows=[];start=time.time()
    out=ROOT/f'samplers{a.part}.json'
    write(ROOT/f'samplers{a.part}_status.json',dict(state='running',pid=os.getpid(),start=start))
    for e in entries:
        b,initial=fetch(e)
        for inp in [0,.03,.05,.09]:
            cache=model.encode_conditions(public(b,torch.tensor([inp**2],device='cuda')))
            for seed_offset in [0,1,2]:
                x=initial if seed_offset==0 else torch.randn(initial.shape,generator=torch.Generator().manual_seed(20261007+10000*int(e['file'][4:8])+seed_offset)).cuda()
                pred=model.denoise(x,torch.tensor([999],device='cuda'),cache).clamp(0,1)
                rows.append(dict(**e,input_sigma=inp,mode='one_step_999',seed_offset=seed_offset,metrics=metrics(pred,b)))
            # Choose mechanisms on tuning videos only. Confirmation assesses the
            # prespecified one-step hypothesis, not the best sampler from this scan.
            if e['group']=='tune' and inp in {0,e['sigma']}:
                for count,spacing in [(5,'leading'),(5,'trailing'),(20,'trailing'),(50,'trailing')]:
                    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule=cfg.diffusion.beta_schedule,prediction_type='sample',clip_sample=True,set_alpha_to_one=True,timestep_spacing=spacing)
                    scheduler.set_timesteps(count,device='cuda');x=initial
                    for t in scheduler.timesteps:
                        pred=model.denoise(x,torch.tensor([int(t)],device='cuda'),cache)
                        x=scheduler.step(pred,t,x,eta=0,return_dict=False)[0]
                    rows.append(dict(**e,input_sigma=inp,mode=f'ddim{count}_{spacing}',seed_offset=0,metrics=metrics(x.clamp(0,1),b)))
        write(out,dict(complete=False,rows=rows,elapsed=time.time()-start))
    write(out,dict(complete=True,rows=rows,elapsed=time.time()-start))
    write(ROOT/f'samplers{a.part}_status.json',dict(state='complete',pid=os.getpid(),finish=time.time()))
if __name__=='__main__':main()
