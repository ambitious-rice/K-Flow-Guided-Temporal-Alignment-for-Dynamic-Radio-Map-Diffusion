#!/usr/bin/env python3
"""Render the validation-only W16 DA calibration and RMDM noise study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _scale_method(scale: float) -> str:
    return "estimated_noise_da" if scale == 1.0 else (
        f"estimated_noise_da_scale{scale:g}".replace(".", "p")
    )


def _psnr(group: dict, method: str) -> float:
    return float(group["methods"][method]["metrics"]["full_image"]["psnr"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--stage-b-summary", required=True)
    parser.add_argument("--rmdm-results", required=True, help="Comma-separated sigma=result.json pairs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selection = _read(args.selection)
    stage_b = _read(args.stage_b_summary)
    rmdm = {
        float(item.split("=", 1)[0]): _read(item.split("=", 1)[1])
        for item in args.rmdm_results.split(",") if item
    }
    scale = float(selection["selected_scale"])
    selected_method = _scale_method(scale)

    lines = [
        "# 未知观测噪声：DA 标量校准与 RMDM 噪声敏感性",
        "",
        "所有结果均为 validation-only；冻结 checkpoint，未使用 test 集。",
        "",
        "## W16 DA 噪声底标量校准",
        "",
        f"Stage-A 在真实 σ={selection['true_sigma']:.3f}、采样率 "
        f"{selection['rates']} 上选择：`λ = {scale:g}`，并固定用于独立 Stage-B。",
        "",
        "| λ | Stage-A macro PSNR (dB) | Stage-A macro NMSE |",
        "|---:|---:|---:|",
    ]
    for candidate in sorted(selection["candidates"].values(), key=lambda value: value["scale"]):
        lines.append(
            f"| {candidate['scale']:g} | {candidate['macro_full_image_psnr']:.3f} | "
            f"{candidate['macro_full_image_nmse']:.6f} |"
        )
    lines.extend([
        "",
        "独立 Stage-B（σ=0.05）full-image PSNR：",
        "",
        "| p | no-DA | ordinary DA | estimated-noise DA | known-noise oracle |",
        "|---:|---:|---:|---:|---:|",
    ])
    for group in sorted(stage_b["groups"].values(), key=lambda value: value["rate"]):
        lines.append(
            f"| {group['rate']:g}% | {_psnr(group, 'no_da'):.3f} | "
            f"{_psnr(group, 'ordinary_da'):.3f} | {_psnr(group, selected_method):.3f} | "
            f"{_psnr(group, 'known_noise_da'):.3f} |"
        )

    lines.extend([
        "",
        "## RMDM-SF 噪声敏感性",
        "",
        "同一 no-Tx RMDM-SF checkpoint；每项报告 full-image PSNR。普通 DA 直接追逐观测，"
        "known-noise DA 仅作为已知 σ 的诊断上界。",
        "",
        "| σ | p | no-DA | ordinary DA | known-noise DA |",
        "|---:|---:|---:|---:|---:|",
    ])
    for sigma, result in sorted(rmdm.items()):
        methods = result["methods"]
        ordinary = next(name for name in methods if name.startswith("guided_s"))
        aware = next((name for name in methods if name.startswith("guided_noise_aware")), None)
        for rate in sorted(methods["baseline"]["rates"], key=float):
            baseline = methods["baseline"]["rates"][rate]["metrics"]["full_image"]["psnr"]
            ordinary_value = methods[ordinary]["rates"][rate]["metrics"]["full_image"]["psnr"]
            aware_value = (
                methods[aware]["rates"][rate]["metrics"]["full_image"]["psnr"]
                if aware else None
            )
            lines.append(
                f"| {sigma:.3f} | {rate}% | {baseline:.3f} | {ordinary_value:.3f} | "
                f"{aware_value:.3f} |" if aware_value is not None else
                f"| {sigma:.3f} | {rate}% | {baseline:.3f} | {ordinary_value:.3f} | — |"
            )
    lines.extend([
        "",
        "RMDM 在采样观测点上的 PSNR（用于检查模型是否随输入噪声过度贴合）：",
        "",
        "| σ | p | noisy input | no-DA output | ordinary DA | known-noise DA |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for sigma, result in sorted(rmdm.items()):
        methods = result["methods"]
        ordinary = next(name for name in methods if name.startswith("guided_s"))
        aware = next((name for name in methods if name.startswith("guided_noise_aware")), None)
        for rate in sorted(methods["baseline"]["rates"], key=float):
            noisy = result["noisy_input_observed_points"][rate]["metrics"]["psnr"]
            baseline = methods["baseline"]["rates"][rate]["observed_points"]["metrics"]["psnr"]
            ordinary_value = methods[ordinary]["rates"][rate]["observed_points"]["metrics"]["psnr"]
            aware_value = (
                methods[aware]["rates"][rate]["observed_points"]["metrics"]["psnr"]
                if aware else None
            )
            lines.append(
                f"| {sigma:.3f} | {rate}% | {noisy:.3f} | {baseline:.3f} | "
                f"{ordinary_value:.3f} | {aware_value:.3f} |" if aware_value is not None else
                f"| {sigma:.3f} | {rate}% | {noisy:.3f} | {baseline:.3f} | "
                f"{ordinary_value:.3f} | — |"
            )
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
