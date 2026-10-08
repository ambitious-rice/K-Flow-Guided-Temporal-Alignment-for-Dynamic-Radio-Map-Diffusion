"""Original six-term loss; sampled x0 supervised by CLEAN truth for every sigma."""
from utils import cal_pinn_components


def sampled_clean_loss(prediction,target,mask):
    return ((prediction.float()-target.float()).square()*mask.float()).sum()/mask.float().sum().clamp_min(1)


def loss_terms(prediction,cal,batch):
    target=batch['target'];mask=batch['sampling_mask']
    obstacle=((batch['building']>.5)|(batch['vehicle']>.5)).float()
    eq,boundary,source=cal_pinn_components(cal.float().flatten(0,2),obstacle.flatten(0,2),batch['source_label'].flatten(0,2),k=.2)
    terms=dict(reconstruction=(prediction.float()-target).square().mean(),sampled_clean=sampled_clean_loss(prediction,target,mask),
        calibration=(cal.float()-target).square().mean(),equation=eq.mean(),obstacle=boundary.mean(),source=source.mean())
    return sum(terms.values()),terms
