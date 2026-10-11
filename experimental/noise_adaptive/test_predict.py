from types import SimpleNamespace

import pytest
import torch

from experimental.paper_evaluation.randomness import ensemble_initial
from experimental.noise_adaptive.predict import assign_folds, collect, masked_inputs, short_window_context, validate_public


def public():
    shape = (1, 16, 1, 4, 6)
    return dict(building=torch.zeros(shape), vehicle=torch.zeros(shape),
                observed_rss=torch.full(shape, .5), sampling_mask=torch.ones(shape),
                sampling_rate=torch.full((1, 16), 2.), measurement_variance=torch.tensor([.0009]))


def fake_model():
    class Model:
        def __init__(self):
            self.denoiser = SimpleNamespace(frames=16, temporal_patch=2, temporal_factor=2,
                local_encoder=[SimpleNamespace(attention=SimpleNamespace(kernel_size=(3, 7, 7)))],
                local_decoder=[SimpleNamespace(attention=SimpleNamespace(kernel_size=(3, 7, 7)))])
        def encode_conditions(self, inputs):
            assert inputs['observed_rss'].shape[1] == self.denoiser.frames
            return {'value': inputs['observed_rss'], 'nested': (inputs['measurement_variance'],)}
        def denoise(self, sample, times, cache):
            assert len(times) == len(sample) == len(cache['value'])
            return sample.sigmoid() * .1 + cache['value']
    return Model()


def test_public_boundary():
    inputs = public()
    validate_public(inputs)
    with pytest.raises(ValueError):
        validate_public(dict(inputs, target=inputs['building']))
    with pytest.raises(ValueError):
        validate_public({key: value for key, value in inputs.items() if key != 'vehicle'})


@pytest.mark.parametrize('folds,spatial', [(4, False), (3, False), (3, True)])
def test_fold_coverage_and_hiding(folds, spatial):
    inputs = public()
    inputs['sampling_mask'][..., 0, 0] = 0
    inputs['observed_rss'][..., 0, 0] = 0
    assignment = assign_folds(inputs['sampling_mask'], video_index=7, start=0, seed=42,
                              folds=folds, spatial=spatial)
    assert torch.equal(assignment >= 0, inputs['sampling_mask'].bool())
    for fold in range(folds):
        hidden = masked_inputs(inputs, assignment, fold)
        assert (hidden['sampling_mask'][assignment == fold] == 0).all()
        assert (hidden['observed_rss'][assignment == fold] == 0).all()
        assert hidden['building'] is inputs['building']
        torch.testing.assert_close(hidden['sampling_rate'], inputs['sampling_rate'] *
            hidden['sampling_mask'].flatten(2).sum(2) / inputs['sampling_mask'].flatten(2).sum(2))
    assert inputs['sampling_mask'].sum() == 16 * 23


def test_context_restores_on_exception():
    model = fake_model()
    with pytest.raises(RuntimeError):
        with short_window_context(model, 4):
            assert model.denoiser.frames == 4
            assert model.denoiser.local_encoder[0].attention.kernel_size == (1, 7, 7)
            raise RuntimeError('test')
    assert model.denoiser.frames == 16
    assert model.denoiser.local_encoder[0].attention.kernel_size == (3, 7, 7)
    assert model.denoiser.local_decoder[0].attention.kernel_size == (3, 7, 7)


def test_numeric_seed_prefix_and_physical_frames():
    kwargs = dict(video_index=7, start=12, fold=2, seed=42)
    full = ensemble_initial((16, 1, 4, 6), members=list(range(8)), **kwargs)
    four = ensemble_initial((4, 1, 4, 6), members=list(range(4)), **kwargs)
    assert torch.equal(full[:4, :4], four)
    later = ensemble_initial((4, 1, 4, 6), video_index=7, start=16, fold=2,
                             seed=42, members=list(range(4)))
    assert torch.equal(full[:4, 4:8], later)


def test_collect_prefix_reuses_chunk_draws_and_restores():
    inputs = public()
    diffusion = SimpleNamespace(train_timesteps=1000, beta_schedule='linear', prediction_type='sample')
    model = fake_model()
    kwargs = dict(video_index=7, start=0, seed=42, folds=3, spatial=True,
                  window=4, steps=2, member_batch=4)
    full = collect(model, diffusion, inputs, members=8, **kwargs)
    prefix = collect(model, diffusion, inputs, members=4, prefix_only=True, **kwargs)
    selected = full['flat_indices'] < inputs['observed_rss'][:, :4].numel()
    assert torch.equal(full['flat_indices'][selected], prefix['flat_indices'])
    assert torch.equal(full['samples'][:4, selected], prefix['samples'])
    assert full['samples'].shape == (8, inputs['observed_rss'].numel())
    assert model.denoiser.frames == 16


def test_batch_timing_accounting():
    diffusion = SimpleNamespace(train_timesteps=1000, beta_schedule='linear', prediction_type='sample')
    result = collect(fake_model(), diffusion, public(), video_index=7, start=0, seed=42,
                     folds=3, spatial=True, window=4, steps=2, members=8, member_batch=4)
    batches = result['meta']['batches']
    assert len(batches) == 4 * 3 * 2
    assert {b['chunk_start'] for b in batches} == {0, 4, 8, 12}
    assert {b['member_start'] for b in batches} == {0, 4}
    assert all(b['seconds'] >= 0 for b in batches)
    shared = result['meta']['shared_setup_seconds']
    assert shared >= 0
    assert sum(b['seconds'] for b in batches) + shared == pytest.approx(result['seconds'])
    prefix_four = sum(b['seconds'] for b in batches if b['chunk_start'] == 0 and b['member_start'] == 0) + shared
    assert 0 < prefix_four <= result['seconds']
