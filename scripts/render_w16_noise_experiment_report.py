#!/usr/bin/env python3
"""Render the complete W16 unknown-noise validation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _delta_cell(comparison: dict) -> str:
    delta = comparison["delta_nmse"]
    lower, upper = comparison["delta_nmse_95ci"]
    return f"{delta:+.6f} [{lower:+.6f}, {upper:+.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimation-summary", required=True)
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--corrected-summary", required=True)
    parser.add_argument("--assimilation-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    estimation = _load(args.estimation_summary)
    folds = _load(args.fold_summary)
    corrected = _load(args.corrected_summary)
    assimilation = _load(args.assimilation_summary)
    gates = estimation["gates"]
    lines = [
        "# W16 no-Tx 单窗口未知噪声实验",
        "",
        "## 噪声估计",
        "",
        f"全部采样率预注册门槛：**{'通过' if gates['passed_all_rates'] else '未通过'}**。",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| clean median sigma | {gates['clean_median_sigma']:.6f} |",
        f"| nonzero-primary window MAE | {gates['window_mae_nonzero_primary']:.6f} |",
        f"| regression slope | {gates['regression']['slope']:.6f} |",
        f"| regression intercept | {gates['regression']['intercept']:.6f} |",
        f"| regression R² | {gates['regression']['r2']:.6f} |",
        "",
        "## F=4 / F=8 p1-clean 敏感性",
        "",
        f"配对窗口数：{folds['matched_windows']}。F=4 median sigma = "
        f"{folds['baseline_median_sigma']:.6f}，F=8 = "
        f"{folds['candidate_median_sigma']:.6f}；配对绝对差均值 = "
        f"{folds['mean_absolute_paired_delta']:.6f}。",
        "",
        "## Posterior-corrected input",
        "",
        "表中为相对 noisy baseline 的 full-image NMSE 变化，负值更好；括号为配对 95% CI。",
        "",
        "| p | sigma | estimated sigma | estimated correction | known-sigma correction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for group in corrected["groups"].values():
        comparisons = group["comparisons_vs_noisy"]
        lines.append(
            f"| {group['rate']:g}% | {group['true_sigma']:.3f} | "
            f"{group['mean_estimated_sigma']:.6f} | "
            f"{_delta_cell(comparisons['corrected_estimated_sigma']['full_image'])} | "
            f"{_delta_cell(comparisons['corrected_known_sigma']['full_image'])} |"
        )

    lines.extend(
        [
            "",
            "## Estimated-noise data assimilation",
            "",
            "表中为相对 no-DA 的 full-image NMSE 变化，负值更好；括号为配对 95% CI。",
            "",
            "| p | sigma | ordinary DA | estimated-noise DA | known-noise DA |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for group in assimilation["groups"].values():
        comparisons = group["comparisons_vs_no_da"]
        lines.append(
            f"| {group['rate']:g}% | {group['true_sigma']:.3f} | "
            f"{_delta_cell(comparisons['ordinary_da']['full_image'])} | "
            f"{_delta_cell(comparisons['estimated_noise_da']['full_image'])} | "
            f"{_delta_cell(comparisons['known_noise_da']['full_image'])} |"
        )

    lines.extend(
        [
            "",
            "所有选择与校准均为 validation-only；正式 test 未用于选择 fold、ensemble、"
            "校准器、噪声档位或通过门槛。",
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
