"""Four-fold estimator with saved folds and explicit CPU integer random streams."""
import time
import torch
from experimental.noise_estimation_crossfit.core import ConditionedEnsemble, FP32, input_boundary, masked_conditions
from .randomness import ensemble_initial

class PaperEnsemble:
    def __init__(self, model, diffusion, steps=50, member_batch=8):
        self.model=model
        self.engine=ConditionedEnsemble(diffusion)
        self.steps,self.member_batch=steps,member_batch

    @torch.inference_mode()
    def collect(self, *, conditions, observed_rss, sampling_mask, assignment, video_index, start):
        device=observed_rss.device
        batch=input_boundary(conditions,observed_rss,sampling_mask,torch.tensor([.03**2],device=device))
        visible=sampling_mask.bool()
        if not torch.equal(assignment>=0,visible) or set(assignment[visible].tolist())!={0,1,2,3}:
            raise ValueError('Saved folds do not match the observed points')
        samples=torch.empty((8,int(visible.sum())),dtype=torch.float32,device=device)
        point_folds=assignment[visible]
        torch.cuda.synchronize();begin=time.perf_counter()
        for fold in range(4):
            hidden=masked_conditions(batch,assignment,fold)
            cache=self.model.encode_conditions(hidden)
            selected=assignment==fold
            for m in range(0,8,self.member_batch):
                members=list(range(m,min(m+self.member_batch,8)))
                noise=ensemble_initial(observed_rss.shape[1:],video_index=video_index,start=start,fold=fold,members=members).to(device)
                pred=self.engine._sample_chunk(self.model,cache,noise,steps=self.steps,accelerator=FP32())
                samples[m:m+len(members),point_folds==fold]=pred[:,selected[0]]
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        samples=samples.cpu()
        if not torch.isfinite(samples).all():
            raise ValueError('Nonfinite ensemble')
        return dict(samples=samples,mean=samples.double().mean(0).float(),
            raw_variance=samples.double().var(0,unbiased=True).float(),observed=observed_rss[visible].cpu(),
            flat_indices=visible.flatten().nonzero().flatten().cpu(),folds=point_folds.cpu(),seconds=elapsed)
