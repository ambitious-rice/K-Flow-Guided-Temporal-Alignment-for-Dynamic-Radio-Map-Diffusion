from dataclasses import replace

import torch
from torch.utils.data import Dataset

from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_hvdit_v4_x0_observation_balance import runner
from rmdm_hvdit_v4_x0_observation_balance.config import (
    AlignmentConfig, EvaluationConfig, ObservationBalanceConfig, OptimizerConfig,
)


class TinyDataset(Dataset):
    reads = []

    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 10

    def set_epoch(self, epoch):
        pass

    def __getitem__(self, index):
        self.reads.append(index)
        target = torch.full((1, 1, 8, 8), 0.1 + index * 0.07)
        zeros = torch.zeros_like(target)
        return {"target": target, "building": zeros, "vehicle": zeros, "tx": zeros,
                "video_id": f"video_{index}", "start": 0}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, noisy_target, timesteps, sparse_batch):
        return self.scale * noisy_target, self.scale * sparse_batch["target"]


def test_segment_resume_matches_continuous_training(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "build_t1_system", lambda config: TinyModel())
    monkeypatch.setattr(runner, "WindowDataset", TinyDataset)
    model_config = ExperimentConfig()
    model_config.data.workers = 0
    model_config.t1_train.per_gpu_batch_size = 2
    model_config.t1_train.gradient_accumulation_steps = 2
    model_config.diffusion.prediction_type = "sample"
    model_config.diffusion.train_timesteps = 10
    source = tmp_path / "source.pth"
    torch.save({"model": TinyModel().state_dict(), "global_step": 10000}, source)
    config = ObservationBalanceConfig(
        source_checkpoint=str(source), max_steps=8, segment_steps=8, log_every_steps=1,
        mixed_precision="no", output_dir="continuous",
        optimizer=OptimizerConfig(warmup_steps=0),
        alignment=AlignmentConfig(warmup_steps=0, hold_steps=0),
        evaluation=EvaluationConfig(gpus=[0]),
    )
    continuous = runner.run_training_segment(model_config, config, repository_root=tmp_path)
    expected = torch.load(continuous["checkpoint"], weights_only=False)
    config = replace(config, segment_steps=2, output_dir="segmented")
    resume = None
    for step in range(2, 9, 2):
        result = runner.run_training_segment(model_config, config, repository_root=tmp_path, resume_from=resume)
        assert result["branch_step"] == step
        resume = result["checkpoint"]
    actual = torch.load(resume, weights_only=False)
    assert torch.equal(actual["model"]["scale"], expected["model"]["scale"])
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["epoch"] == expected["epoch"]
    assert actual["microbatches_consumed_in_epoch"] == expected["microbatches_consumed_in_epoch"]
    for key, value in actual["optimizer"]["state"][0].items():
        assert torch.equal(value, expected["optimizer"]["state"][0][key])


def test_resume_skips_decoding_consumed_batches():
    from accelerate import Accelerator, DataLoaderConfiguration
    from torch.utils.data import DataLoader
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(
        use_seedable_sampler=True, data_seed=17,
    ))
    loader = accelerator.prepare_data_loader(DataLoader(TinyDataset(), batch_size=2, shuffle=True))
    loader.set_epoch(3)
    expected = [batch["video_id"] for batch in loader][3:]
    loader.set_epoch(3)
    TinyDataset.reads = []
    resumed = accelerator.skip_first_batches(loader, 3)
    resumed.set_epoch(3)
    assert [batch["video_id"] for batch in resumed] == expected
    assert len(TinyDataset.reads) == 4
