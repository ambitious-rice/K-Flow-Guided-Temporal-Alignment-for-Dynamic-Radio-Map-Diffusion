import torch
from .data import paired_batch


def example():
    g = torch.Generator().manual_seed(4)
    x = torch.rand(8, 16, 1, 4, 4, generator=g)
    mask = (torch.rand(x.shape, generator=g)>.5).float()
    return dict(target=x, building=torch.zeros_like(x), sampling_mask=mask,
                diffusion_noise=torch.randn(x.shape, generator=g),
                timesteps=torch.arange(8), sampling_rate=torch.arange(8)[:, None].expand(-1, 16).float(),
                measurement_variance=torch.zeros(8), observed_rss=mask*x)


def test_pairing_preserves_mask_signal_diffusion_but_varies_measurement_noise():
    original = example()
    b = paired_batch(original, 20261009)
    sigma = b['measurement_variance'].sqrt()
    for start in [0, 4]:
        for k in ['target', 'sampling_mask', 'diffusion_noise', 'timesteps', 'sampling_rate']:
            assert all(torch.equal(b[k][start], b[k][j]) for j in range(start, start+4))
        assert sorted((sigma[start:start+4]/(.09/4)).floor().int().tolist())==[0, 1, 2, 3]
        recovered = (b['observed_rss']-b['sampling_mask']*b['target'])/sigma[:, None, None, None, None]
        for j in range(start+1, start+4):
            torch.testing.assert_close(recovered[start], recovered[j], atol=1e-5, rtol=1e-5)
    assert not (b['observed_rss']*(1-b['sampling_mask'])).any()
    for k,v in paired_batch(original, 20261009).items():
        assert torch.equal(v, b[k])
    assert not torch.equal(paired_batch(original, 20261010)['observed_rss'], b['observed_rss'])


def test_stratification_has_uniform_marginal_and_no_fixed_sigma_grid():
    levels = torch.cat([paired_batch(example(), s)['measurement_variance'].sqrt() for s in range(200)])
    assert 0 <= levels.min() < levels.max() < .09
    assert abs(float(levels.mean())-.045)<.002
    assert len(levels.unique())>1500
