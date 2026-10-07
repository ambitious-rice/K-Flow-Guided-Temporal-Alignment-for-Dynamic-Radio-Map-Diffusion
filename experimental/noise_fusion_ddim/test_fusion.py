from types import SimpleNamespace
import torch
from experimental.noise_fusion_ddim.fusion import NoiseWeightedFusion,sample
from rmdm.diffusion import DDIMSampler

def test_physical_limits_mask_and_trainable_weight():
    m=NoiseWeightedFusion(.03);z=torch.zeros(3,2,1,2,2);y=torch.ones_like(z);mask=torch.ones_like(z);mask[:,:,:,:,1]=0
    q=torch.tensor([0,.03**2,.09**2]);out=m(z,y,mask,q)
    torch.testing.assert_close(out[:,0,0,0,0],torch.tensor([1,.5,.1]));assert (out[:,:,:,:,1]==0).all()
    out.sum().backward();assert m.log_prior_variance.grad>0

def test_disabled_matches_ddim_and_targets_are_excluded():
    class Model:
        def encode_conditions(self,b):
            assert 'target' not in b and 'source_label' not in b
            return b
        def denoise(self,x,t,b):return .2*x+.1*b['observed_rss']+.3
    cfg=SimpleNamespace(train_timesteps=1000,beta_schedule='linear',prediction_type='sample',ddim_steps=5)
    x=torch.randn(1,2,1,4,4,generator=torch.Generator().manual_seed(17));b={k:torch.ones_like(x)*.5 for k in ['building','vehicle','observed_rss','sampling_mask']};b.update(sampling_rate=torch.tensor([2.]),measurement_variance=torch.tensor([.03**2]),target=x*100,source_label=x*50)
    model=Model();a=sample(model,cfg,b,x,steps=5)
    clean={k:v for k,v in b.items() if k not in ['target','source_label']}
    expected=DDIMSampler(cfg).sample(model,clean,initial_noise=x,steps=5)
    torch.testing.assert_close(a,expected,rtol=0,atol=0)
    b['target']*=0;b['source_label']*=0;c=sample(model,cfg,b,x,steps=5)
    torch.testing.assert_close(a,c,rtol=0,atol=0)

def test_final_only_cannot_improve_unobserved_pixels():
    m=NoiseWeightedFusion(.03);z=torch.rand(1,2,1,4,4);y=torch.rand_like(z);mask=torch.zeros_like(z);mask[:,:,:,:2]=1
    out=m(z,y,mask,torch.tensor([.05**2]));torch.testing.assert_close(out[mask==0],z[mask==0],rtol=0,atol=0)
