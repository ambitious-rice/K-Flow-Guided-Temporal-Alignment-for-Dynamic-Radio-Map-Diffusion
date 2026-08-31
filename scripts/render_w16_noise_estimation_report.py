#!/usr/bin/env python3
"""Render the W16 noise-estimation calibration and blind results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibration = _load(args.calibration)
    summary = _load(args.summary)
    gates = summary["gates"]
    affine = calibration["variance_calibration"]

    lines = [
        "# W16 no-Tx 单窗口未知噪声估计",
        "",
        "## 结论",
        "",
        f"预注册主门槛：**{'通过' if gates['passed_primary'] else '未通过'}**；"
        f"全部采样率门槛：**{'通过' if gates['passed_all_rates'] else '未通过'}**。",
        "",
        "## Clean-only 方差校准",
        "",
        f"使用 `v = max({affine['scale']:.6g} * v_raw + {affine['offset']:.6g}, "
        f"{affine['floor']:.1e})`，由 {calibration['unit_count']} 个 clean W16 单元拟合。",
        "",
        "## 预注册门槛",
        "",
        "| 项目 | 数值/状态 |",
        "|---|---:|",
        f"| clean median sigma | {gates['clean_median_sigma']:.6f} |",
        f"| nonzero-primary window MAE | {gates['window_mae_nonzero_primary']:.6f} |",
        f"| regression slope | {gates['regression']['slope']:.6f} |",
        f"| regression intercept | {gates['regression']['intercept']:.6f} |",
        f"| regression R² | {gates['regression']['r2']:.6f} |",
    ]
    for name, passed in gates["checks"].items():
        lines.append(f"| gate: {name} | {'pass' if passed else 'fail'} |")

    lines.extend(
        [
            "",
            "## 主方法分组结果",
            "",
            "| p | true sigma | median estimate | MAE | median 95% bootstrap CI |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"].values():
        ci = group["paired_video_bootstrap"]["median_estimate_95ci"]
        lines.append(
            f"| {group['rate']:g}% | {group['true_sigma']:.3f} | "
            f"{group['median_estimate']:.6f} | {group['mae']:.6f} | "
            f"[{ci[0]:.6f}, {ci[1]:.6f}] |"
        )

    lines.extend(
        [
            "",
            "## 估计器消融：非零主档位 pooled MAE",
            "",
            "| 方法 | MAE |",
            "|---|---:|",
        ]
    )
    method_errors: dict[str, list[tuple[float, int]]] = {}
    for group in summary["groups"].values():
        if group["true_sigma"] not in {0.005, 0.01, 0.02, 0.03, 0.05}:
            continue
        for method, metrics in group["methods"].items():
            method_errors.setdefault(method, []).append((metrics["mae"], group["windows"]))
    for method, values in method_errors.items():
        pooled = sum(error * count for error, count in values) / sum(count for _, count in values)
        lines.append(f"| {method} | {pooled:.6f} |")

    audit = summary["numerical_audit"]
    lines.extend(
        [
            "",
            "## 数值审计",
            "",
            f"MLE 与 EM 的最大方差差为 `{audit['maximum_em_mle_variance_gap']:.6g}`，"
            f"中位差为 `{audit['median_em_mle_variance_gap']:.6g}`。",
            "",
            "所有结果均为 validation-only；未使用正式 test 选择 fold、ensemble、校准器或门槛。",
            "",
        ]
    )
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
