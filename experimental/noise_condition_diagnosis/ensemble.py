"""Separate seed variance from systematic reconstruction error on tuning data."""
import argparse,json,time
import torch
from rmdm.diffusion import DDIMSampler
from experimental.noise_loss_screen.common import public,metrics,write
from .audit import ROOT,load,bank,fetch

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--part',type=int,required=True);a=p.parse_args();cfg,model,_=load();model.eval();sampler=DDIMSampler(cfg.diffusion);rows=[];start=time.time()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for e in bank()[a.part::2]:
  b,initial=fetch(e)
  for sigma in sorted({0,e['sigma']}):
   samples=[]
   for offset in [0,1,2]:
    z=initial if offset==0 else torch.randn(initial.shape,generator=torch.Generator().manual_seed(20261007+10000*int(e['file'][4:8])+offset)).cuda()
    samples.append(sampler.sample(model,public(b,torch.tensor([sigma**2],device='cuda')),initial_noise=z,steps=50,eta=0))
   avg=torch.stack(samples).mean(0);unseen=b['valid_mask']*(1-b['sampling_mask'])
   spread=float(((torch.stack(samples)-avg).square().mean(0)*unseen).sum()/unseen.sum())
   single=sum(metrics(x,b)['unobserved_mse'] for x in samples)/3
   rows.append(dict(**e,input_sigma=sigma,member_mean_mse=single,ensemble_mean_mse=metrics(avg,b)['unobserved_mse'],seed_variance=spread))
  write(ROOT/f'ensemble{a.part}.json',dict(complete=False,rows=rows,elapsed=time.time()-start))
 write(ROOT/f'ensemble{a.part}.json',dict(complete=True,rows=rows,elapsed=time.time()-start))
if __name__=='__main__':main()
