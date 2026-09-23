"""Scientific contracts that would invalidate the comparison if broken."""
from copy import deepcopy

import torch

from rmdm.data import SamplingPolicy
from rmdm.diffusion import DiffusionProcess
from experimental.noise_temporal_rmdm.noise import add_fixed_measurement_noise
from .config import Config
from .model import NoiseHVDiT, initialize_w16
from .step import training_step
from .evaluate import as_single_frames, frame_noise


def small_config(frames=1):
    c = Config()
    c.data.window_size = frames
    c.embedding_width = 32
    m = c.model
    m.local_dim, m.global_dim, m.head_dim = 32, 64, 16
    m.local_depth = m.global_depth = 1
    m.local_kernel = [3, 3, 3]
    m.rope_axis_dims = [4, 6, 6]
    m.mapping_width = 64
    m.decoder_token_channels = 16
    m.decoder_stage_channels = [16, 8]
    m.local_attention_backend = "reference"
    m.gradient_checkpointing = False
    return c


def dense_batch(frames=1):
    shape = (1, frames, 1, 64, 64)
    source = torch.zeros(shape)
    source[..., 30:33, 30:33] = 1
    return dict(building=torch.zeros(shape), vehicle=torch.zeros(shape), target=torch.rand(shape),
                source_label=source, video_id=["scene/episode/tx"], start=torch.tensor([0]))


def test_source_labels_never_change_predictions_but_train_hwm():
    c = small_config()
    model = NoiseHVDiT(c).eval()
    torch.nn.init.normal_(model.denoiser.output_head.final_projection.weight, std=0.01)
    dense = dense_batch()
    sparse = add_fixed_measurement_noise(SamplingPolicy(c.sampling)(dense), .03, seed=1)
    noisy, t = torch.randn_like(dense["target"]), torch.tensor([500])
    prediction, cal = model(noisy, t, sparse)
    alternate = dict(sparse, source_label=1-sparse["source_label"], tx=torch.ones_like(noisy))
    other, other_cal = model(noisy, t, alternate)
    torch.testing.assert_close(prediction, other, rtol=0, atol=0)
    torch.testing.assert_close(cal, other_cal, rtol=0, atol=0)
    loss, metrics = training_step(model, dense, SamplingPolicy(c.sampling), DiffusionProcess(c.diffusion), c, 0)
    loss.backward()
    assert metrics["source"] > 0
    assert model.hwm.network.seg_outputs[0].weight.grad.abs().sum() > 0


def test_conditioning_zero_initialization_then_variance_effect():
    c = small_config()
    model = NoiseHVDiT(c).eval()
    torch.nn.init.normal_(model.denoiser.output_head.final_projection.weight, std=0.01)
    dense = dense_batch()
    sparse = add_fixed_measurement_noise(SamplingPolicy(c.sampling)(dense), .03, seed=1)
    noisy, t = torch.randn_like(dense["target"]), torch.tensor([500])
    altered = dict(sparse, measurement_variance=torch.tensor([.0081]))
    with torch.no_grad():
        a = model(noisy, t, sparse)[0]
        b = model(noisy, t, altered)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for head in model.heads.values():
            torch.nn.init.normal_(head.weight, std=.02)
        a = model(noisy, t, sparse)[0]
        b = model(noisy, t, altered)[0]
        assert (a-b).abs().max() > 1e-6


def test_prediction_targets_and_auxiliary_losses():
    c = small_config()
    class Fixed(torch.nn.Module):
        def forward(self, x, t, batch):
            return torch.zeros_like(x), torch.full_like(x, .5)
    dense = dense_batch()
    _, x0 = training_step(Fixed(), dense, SamplingPolicy(c.sampling), DiffusionProcess(c.diffusion), c, 0)
    c.diffusion.prediction_type = "epsilon"
    _, eps = training_step(Fixed(), dense, SamplingPolicy(c.sampling), DiffusionProcess(c.diffusion), c, 0)
    torch.testing.assert_close(x0["diffusion"], dense["target"].square().mean())
    assert .85 < eps["diffusion"] < 1.15
    for key in ("calibration", "equation", "obstacle", "source"):
        torch.testing.assert_close(x0[key], eps[key], rtol=0, atol=0)


def test_w16_inflation_copies_noise_conditioning():
    w1 = NoiseHVDiT(small_config())
    w16 = NoiseHVDiT(small_config(16))
    initialize_w16(w16, {"model": w1.state_dict()})
    for name, tensor in w1.state_dict().items():
        if not name.startswith("denoiser."):
            torch.testing.assert_close(tensor, w16.state_dict()[name], rtol=0, atol=0)


def test_evaluation_physical_frames_and_initial_noise_pair():
    c = small_config(16)
    dense = dense_batch(16)
    sparse = add_fixed_measurement_noise(SamplingPolicy(c.sampling)(dense), .03, seed=1)
    flat = as_single_frames(sparse)
    torch.testing.assert_close(flat["observed_rss"].reshape_as(sparse["observed_rss"]), sparse["observed_rss"])
    clip = frame_noise(dense["target"], dense["video_id"], [0], 42)
    for t in (0, 7, 15):
        frame = frame_noise(dense["target"][:, :1], dense["video_id"], [t], 42)
        torch.testing.assert_close(clip[:, t:t+1], frame, rtol=0, atol=0)
