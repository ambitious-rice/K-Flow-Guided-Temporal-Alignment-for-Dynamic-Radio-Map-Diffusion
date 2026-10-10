import math
import numpy as np
import torch

SIGMAS=[0.,.03,.05,.09]


def sufficient_stats(pred, b):
    error=pred.float()-b['target'].float();sq=error.square()
    unseen=b['valid_mask']*(1-b['sampling_mask']);observed=b['sampling_mask']
    frame=sq.flatten(2).mean(2)
    delta=(error[:,1:]-error[:,:-1]).square()
    return dict(sse=float(sq.sum()),sae=float(error.abs().sum()),count=error.numel(),
        unseen_sse=float((sq*unseen).sum()),unseen_count=float(unseen.sum()),
        observed_sse=float((sq*observed).sum()),observed_count=float(observed.sum()),
        frame_psnr_sum=float((-10*frame.clamp_min(1e-12).log10()).sum()),frames=frame.numel(),
        temporal_sse=float(delta.sum()),temporal_count=delta.numel())


def merge_stats(items):
    total={k:sum(d[k] for d in items) for k in items[0]}
    mse=total['sse']/total['count'];unseen=total['unseen_sse']/max(1,total['unseen_count'])
    return dict(mse=mse,rmse=math.sqrt(mse),mae=total['sae']/total['count'],
        psnr=-10*math.log10(max(mse,1e-12)),mean_frame_psnr=total['frame_psnr_sum']/total['frames'],
        unobserved_mse=unseen,unobserved_rmse=math.sqrt(unseen),
        observed_mse=total['observed_sse']/max(1,total['observed_count']),
        temporal_delta_mse=total['temporal_sse']/total['temporal_count'])


def assess(matrix):
    m=np.asarray(matrix,dtype=float);assert m.shape==(4,4) and np.isfinite(m).all()
    diag=np.diag(m);minimum=m.min(1);gap=diag-minimum
    passed=gap<=1e-12 # Numerical equality only, no practical-error allowance.
    return dict(pass_count=int(passed.sum()),all_four=bool(passed.all()),pass_by_sigma=passed.tolist(),
        row_min_input=[SIGMAS[i] for i in m.argmin(1)],gap=gap.tolist(),
        relative_gap=(gap/np.maximum(minimum,1e-12)).tolist(),
        correct_mse=float(diag.mean()),best_fixed_sigma=SIGMAS[int(m.mean(0).argmin())],
        gain_vs_best_fixed=float(1-diag.mean()/max(m.mean(0).min(),1e-12)))
