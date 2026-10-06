"""CPU random streams, using only documented integer seed arithmetic."""
import torch

MASTER_SEED = 20261006


def frame_seed(video_index, frame, stream, *, seed=MASTER_SEED):
    # A unique slot for each physical frame and stream. Video indices refer to
    # the saved full-data index, so selecting another subset does not change it.
    if not 0 <= int(frame) < 100 or not 0 <= int(stream) < 100:
        raise ValueError('Frame and stream indices must be in [0,100)')
    if int(video_index) < 0:
        raise ValueError('Video index must be nonnegative')
    return int(seed)+int(video_index)*1_000_000+int(frame)*100+int(stream)


def generator(video_index, frame, stream, *, seed=MASTER_SEED):
    return torch.Generator(device='cpu').manual_seed(frame_seed(video_index,frame,stream,seed=seed))


def balanced_folds(mask, *, video_index, start, seed=MASTER_SEED, folds=4):
    if mask.device.type!='cpu' or mask.ndim!=5 or mask.shape[:1]!=(1,):
        raise ValueError('Expected one CPU [1,T,1,H,W] mask')
    assignment=torch.full_like(mask,-1,dtype=torch.int8)
    for t in range(mask.shape[1]):
        points=mask[0,t,0].flatten().nonzero().flatten()
        if len(points)<folds:
            raise ValueError('Not enough observations for balanced folds')
        order=points[torch.randperm(len(points),generator=generator(video_index,start+t,3,seed=seed))]
        assignment[0,t,0].view(-1)[order]=(torch.arange(len(points))%folds).to(torch.int8)
    return assignment


def ensemble_initial(shape, *, video_index, start, fold, members, seed=MASTER_SEED):
    """[members,T,C,H,W], generated on CPU then moved by the caller."""
    if len(shape)!=4 or not 0 <= fold < 4 or any(not 0 <= m < 8 for m in members):
        raise ValueError('Expected [T,C,H,W], four folds, and up to eight members')
    return torch.stack([torch.stack([torch.randn(shape[1:],generator=generator(
        video_index,start+t,10+8*fold+member,seed=seed),dtype=torch.float32)
        for t in range(shape[0])]) for member in members])


def materialize(dense, *, video_index, rate, sigma, seed=MASTER_SEED):
    target=dense['target'].cpu().float()
    building,vehicle=dense['building'].cpu().float(),dense['vehicle'].cpu().float()
    valid=((building<=.5)&(vehicle<=.5)).float()
    mask,noise,initial=[torch.zeros_like(target) for _ in range(3)]
    start=int(dense['start'][0])
    if target.shape[0]!=1:
        raise ValueError('Prepare one window at a time')
    for t in range(target.shape[1]):
        frame=start+t
        pixels=valid[0,t].flatten().nonzero().flatten()
        if not len(pixels):
            raise ValueError('No free-space pixels')
        order=pixels[torch.randperm(len(pixels),generator=generator(video_index,frame,0,seed=seed))]
        mask[0,t].view(-1)[order[:max(1,round(rate/100*len(pixels)))]]=1
        noise[0,t]=torch.randn(target.shape[2:],generator=generator(video_index,frame,1,seed=seed))
        initial[0,t]=torch.randn(target.shape[2:],generator=generator(video_index,frame,2,seed=seed))
    sparse=dict(target=target,building=building,vehicle=vehicle,valid_mask=valid,sampling_mask=mask,
        observed_rss=mask*(target+float(sigma)*noise),sampling_rate=torch.full(target.shape[:2],float(rate)),
        measurement_variance=torch.tensor([float(sigma)**2]),
        measurement_standard_deviation=torch.tensor([float(sigma)]),video_id=list(dense['video_id']),
        start=torch.as_tensor(dense['start']).cpu())
    return sparse,initial
