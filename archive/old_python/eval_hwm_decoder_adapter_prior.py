#!/usr/bin/env python3
"""Evaluate frozen HWM prior vs a temporal HWM decoder adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset

from hwm_decoder_adapter_student import HWMDecoderAdapterStudent
from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
from residual_kflow_student import KPathDeltaConfig, kpath_delta_target_loss
from train_temporal_pinn import build_model_config, preprocess_conditions
from utils import build_unet_from_config, cal_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--split_file", default="split.json")
    parser.add_argument("--scene_ids", default="")
    parser.add_argument("--num_clips", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--clip_length", type=int, default=32)
    parser.add_argument("--frame_stride", type=int, default=32)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--tx_heatmap_sigma_px", type=float, default=1.5)
    parser.add_argument("--k2_root", default="")
    parser.add_argument("--num_channels", type=int, default=96)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", default="16")
    parser.add_argument("--channel_mult", default="")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_checkpoint", type=bool, default=False)
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_fp16", type=bool, default=False)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=-1)
    parser.add_argument("--num_heads_upsample", type=int, default=-1)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--init_2d_checkpoint", required=True)
    parser.add_argument("--adapter_checkpoint", required=True)
    parser.add_argument("--adapter_indices", default="-2,-1")
    parser.add_argument("--adapter_reduction", type=int, default=1)
    parser.add_argument("--adapter_min_hidden", type=int, default=16)
    parser.add_argument("--pinn_k", type=float, default=0.2)
    parser.add_argument("--kpath_delta_k2_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_static_flow_quantile", type=float, default=0.70)
    parser.add_argument("--kpath_delta_dynamic_flow_quantile", type=float, default=0.85)
    parser.add_argument("--kpath_delta_margin", type=float, default=0.005)
    parser.add_argument("--kpath_delta_dynamic_lambda", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _sum_metric(acc: dict[str, torch.Tensor], key: str, value: torch.Tensor, count: int) -> None:
    acc[key] = acc.get(key, value.new_tensor(0.0)) + value.detach().float() * int(count)


def _temporal_delta(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] <= 1:
        return x.new_tensor(0.0)
    return (x[:, 1:] - x[:, :-1]).abs().mean()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    scene_ids = [item.strip() for item in args.scene_ids.split(",") if item.strip()] or None
    adapter_indices = tuple(int(item.strip()) for item in args.adapter_indices.split(",") if item.strip())

    dataset = DynamicRadioMapRMDMClip(
        root=args.data_dir,
        split=args.split,
        split_file=args.split_file,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        cache_size=args.cache_size,
        tx_heatmap_sigma_px=args.tx_heatmap_sigma_px,
        include_k2=True,
        k2_root=args.k2_root or None,
        scene_ids=scene_ids,
    )
    eval_count = min(int(args.num_clips), len(dataset))
    shard_indices = list(range(eval_count))[accelerator.process_index :: accelerator.num_processes]
    subset = Subset(dataset, shard_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    base_model = build_unet_from_config(build_model_config(args))
    base_model.load_state_dict(torch.load(args.init_2d_checkpoint, map_location="cpu"), strict=True)
    cand = HWMDecoderAdapterStudent(
        base_model,
        adapter_indices=adapter_indices,
        reduction=args.adapter_reduction,
        min_hidden=args.adapter_min_hidden,
    )
    payload = torch.load(args.adapter_checkpoint, map_location="cpu")
    cand.adapters.load_state_dict(payload["adapters"], strict=True)
    cand.to(device)
    cand.eval()

    kpd_cfg = KPathDeltaConfig(
        keypath_k2_quantile=args.kpath_delta_k2_quantile,
        static_flow_quantile=args.kpath_delta_static_flow_quantile,
        dynamic_flow_quantile=args.kpath_delta_dynamic_flow_quantile,
        margin=args.kpath_delta_margin,
        dynamic_lambda=args.kpath_delta_dynamic_lambda,
    )
    sums: dict[str, torch.Tensor] = {}

    for batch in loader:
        raw_inputs = batch["inputs"].to(device, non_blocking=True)
        conditions = preprocess_conditions(raw_inputs)
        target = batch["image"].to(device, non_blocking=True)
        k2 = batch["k2"].to(device, non_blocking=True)
        batch_size = int(target.shape[0])
        sums["_count"] = sums.get("_count", target.new_tensor(0.0)) + target.new_tensor(float(batch_size))
        with torch.no_grad():
            raw_model = accelerator.unwrap_model(cand)
            _, base_cal, _ = raw_model.forward_hwm_baseline(conditions)
            cand_cal, _ = cand(conditions)

        frames = target.shape[1]
        buildings = raw_inputs[:, :, 0].reshape(batch_size * frames, *target.shape[-2:])
        antenna = raw_inputs[:, :, 1].reshape(batch_size * frames, *target.shape[-2:])
        for prefix, cal in (("base", base_cal), ("cand", cand_cal)):
            mse = F.mse_loss(cal, target)
            mae = F.l1_loss(cal, target)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            flat_cal = cal[:, :, 0].reshape(batch_size * frames, *target.shape[-2:])
            pinn = cal_pinn(flat_cal, buildings, antenna, k=args.pinn_k).mean()
            kpd, kpd_stats = kpath_delta_target_loss(cal, target, k2, cfg=kpd_cfg)
            _sum_metric(sums, f"{prefix}_mae", mae, batch_size)
            _sum_metric(sums, f"{prefix}_mse", mse, batch_size)
            _sum_metric(sums, f"{prefix}_psnr", psnr, batch_size)
            _sum_metric(sums, f"{prefix}_tdelta", _temporal_delta(cal), batch_size)
            _sum_metric(sums, f"{prefix}_pinn", pinn, batch_size)
            _sum_metric(sums, f"{prefix}_kpd", kpd, batch_size)
            _sum_metric(sums, f"{prefix}_kpd_static", kpd_stats["static_loss"], batch_size)
            _sum_metric(sums, f"{prefix}_kpd_dynamic", kpd_stats["dynamic_loss"], batch_size)
            _sum_metric(sums, f"{prefix}_pred_dynamic", kpd_stats["pred_dynamic"], batch_size)
            _sum_metric(sums, f"{prefix}_gt_dynamic", kpd_stats["gt_dynamic"], batch_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_payload = {key: float(value.detach().cpu().item()) for key, value in sums.items()}
    shard_payload["indices"] = shard_indices
    (output_dir / f"shard_{accelerator.process_index:02d}.json").write_text(
        json.dumps(shard_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        gathered_float: dict[str, float] = {}
        merged_indices: list[int] = []
        for rank in range(accelerator.num_processes):
            shard_path = output_dir / f"shard_{rank:02d}.json"
            data = json.loads(shard_path.read_text(encoding="utf-8"))
            merged_indices.extend(int(idx) for idx in data.pop("indices"))
            for key, value in data.items():
                gathered_float[key] = gathered_float.get(key, 0.0) + float(value)
        count = float(gathered_float["_count"])
        metrics = {key: float(value / count) for key, value in gathered_float.items() if key != "_count"}
        for key in list(metrics):
            if key.startswith("base_"):
                suffix = key[len("base_") :]
                cand_key = f"cand_{suffix}"
                if cand_key in metrics:
                    metrics[f"delta_{suffix}"] = metrics[cand_key] - metrics[key]
        out = {
            "num_clips": int(count),
            "args": vars(args),
            "metrics": metrics,
        }
        out["indices"] = sorted(merged_indices)
        (output_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
