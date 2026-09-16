import torch

from experimental.tx_prior.diagnose import epsilon_from_x0, shuffle_scene, x0_from_epsilon


def test_epsilon_x0_oracle_and_baselines():
    x0 = torch.rand(2, 1, 1, 8, 8)
    noise = torch.randn_like(x0)
    for alpha in (torch.tensor(0.99), torch.tensor(0.001)):
        xt = alpha.sqrt() * x0 + (1-alpha).sqrt() * noise
        assert torch.allclose(epsilon_from_x0(xt, x0, alpha), noise, atol=1e-6)
        assert torch.allclose(x0_from_epsilon(xt, noise, alpha), x0, atol=1e-5)
        for baseline in (torch.zeros_like(x0), x0 * 0.5):
            eps = epsilon_from_x0(xt, baseline, alpha)
            assert torch.allclose(x0_from_epsilon(xt, eps, alpha), baseline, atol=1e-5)


def test_shuffle_preserves_target_and_identity():
    batch = {k: torch.arange(2).reshape(2, 1) for k in
             ("building", "tx", "vehicle", "legacy_conditions", "target")}
    batch["frame_names"] = ["a", "b"]
    shuffled = shuffle_scene(batch)
    for k in ("building", "tx", "vehicle", "legacy_conditions"):
        assert torch.equal(shuffled[k], batch[k].flip(0))
    assert shuffled["target"] is batch["target"]
    assert shuffled["frame_names"] is batch["frame_names"]
