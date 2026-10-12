from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .efficient import collect_w4, pool_update
from .predict import collect
from .test_predict import fake_model, public

DIFFUSION = SimpleNamespace(train_timesteps=1000, beta_schedule='linear', prediction_type='sample')
KW = dict(video_index=7, start=0, seed=42)


@pytest.mark.parametrize('batch', [1, 2, 4])
def test_fold_batches_preserve_predictions_and_indices(batch):
    inputs = public()
    inputs['observed_rss'][:, :, :, ::2] = .2
    inputs['sampling_mask'][..., 0, 0] = 0
    inputs['observed_rss'][..., 0, 0] = 0
    old = collect(fake_model(), DIFFUSION, inputs, window=4, prefix_only=True, members=4, steps=2, **KW)
    new = collect_w4(fake_model(), DIFFUSION, inputs, fold_batch=batch, steps=2, **KW)
    assert torch.equal(new['flat_indices'], old['flat_indices'])
    assert torch.equal(new['samples'], old['samples'])


def test_holdout_and_later_frames_preserve_physical_identity():
    model = fake_model()
    all_frames = collect(model, DIFFUSION, public(), window=4, members=4, steps=2, **KW)
    later = collect_w4(model, DIFFUSION, public(), offset=4, selected_folds=(0,), members=2, steps=2, **KW)
    columns = torch.searchsorted(all_frames['flat_indices'], later['flat_indices'])
    assert torch.equal(later['samples'], all_frames['samples'][:2, columns])
    assert (later['folds'] == 0).all()
    assert later['flat_indices'].min() >= 4*24
    assert later['flat_indices'].max() < 8*24
    assert model.denoiser.frames == 16


def test_pool_uses_each_generation_q_and_rejects_duplicates():
    calibration = dict(nodes=[dict(q=0., scale=0., offset=.001, floor=1e-8),
                              dict(q=.0081, scale=0., offset=.01, floor=1e-8)])
    def block(idx, q):
        return dict(q=q, calibration=calibration, moments=dict(flat_indices=torch.tensor([idx]),
            mean=torch.tensor([0.]), raw_variance=torch.tensor([.2]), observed=torch.tensor([.1])))
    blocks = [block(0, 0.), block(1, .0081)]
    result = pool_update(blocks)
    from .statistics import mle
    expected = mle(np.array([.1, .1], dtype=np.float32), np.zeros(2), np.array([.001, .01]))
    assert result['variance'] == pytest.approx(expected['variance'])
    with pytest.raises(ValueError, match='Duplicate'):
        pool_update([blocks[0], blocks[0]])
