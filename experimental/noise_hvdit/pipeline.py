"""Train W1 targets in sequence, compare on validation, then expand to W16."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from .config import load_config
from .train import atomic_save


ROOT = Path("runs/noise_hvdit")


def execute(module, *args):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "-m", f"experimental.noise_hvdit.{module}", *map(str, args)]
    write_json_atomic(ROOT/"pipeline_status.json", dict(state="running", command=command))
    print("Executing:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def train(config_path, initialize=""):
    config = load_config(config_path)
    output = Path(config.output)
    status = output/"status.json"
    if status.exists() and json.loads(status.read_text())["state"] == "complete":
        return
    args = ["--config", config_path]
    last = output/"checkpoints"/"last.pth"
    if last.exists():
        args.extend(["--resume", last])
    elif initialize:
        args.extend(["--initialize", initialize])
    execute("train", *args)


def select(config_path):
    config = load_config(config_path)
    output = Path(config.output)
    selected = output/"selection.json"
    if selected.exists():
        return json.loads(selected.read_text())
    candidates = [json.loads(p.read_text()) for p in (output/"validation").glob("step_*.json")]
    # Fast validation shortlists three (checkpoint, weight-type) candidates;
    # the full paired DDIM50 protocol makes the final selection.
    candidates.sort(key=lambda c: c["summary"]["all"]["unobserved_mse"])
    reports = []
    for candidate in candidates[:3]:
        step, weights = candidate["step"], candidate["weights"]
        checkpoint = output/"checkpoints"/f"step_{step:06d}.pth"
        report = output/"validation_full"/f"step_{step:06d}_{weights}.json"
        if not report.exists():
            execute("evaluate", "--config", config_path, "--checkpoint", checkpoint,
                    "--weights", weights, "--output", report)
        reports.append(json.loads(report.read_text()))
    best_clean = min(r["summary"]["clean"]["unobserved_mse"] for r in reports)
    eligible = [r for r in reports if r["summary"]["clean"]["unobserved_mse"] <= 1.05*best_clean]
    winner = min(eligible, key=lambda r: r["summary"]["all"]["unobserved_mse"])
    write_json_atomic(selected, winner)
    return winner


def compare(x0, epsilon):
    reports = {"x0": x0, "epsilon": epsilon}
    winner = min(reports, key=lambda key: reports[key]["summary"]["all"]["unobserved_mse"])
    loser = "epsilon" if winner == "x0" else "x0"
    win, lose = reports[winner], reports[loser]
    gain = 1-win["summary"]["all"]["unobserved_mse"]/lose["summary"]["all"]["unobserved_mse"]
    def by_video(report):
        values = {}
        for row in report["rows"]:
            values.setdefault(row["video_id"], []).append(row["unobserved_mse"])
        return {video: np.mean(metrics) for video, metrics in values.items()}
    w, l = by_video(win), by_video(lose)
    videos = sorted(w)
    differences = np.array([w[v]-l[v] for v in videos])
    rng = np.random.default_rng(20260923)
    scenes = sorted({v.split('/')[0] for v in videos})
    # Paired bootstrap clustered by video and stratified by scene.
    indices = np.concatenate([
        rng.choice([i for i,v in enumerate(videos) if v.split('/')[0] == scene],
                   size=(5000, sum(v.split('/')[0] == scene for v in videos)), replace=True)
        for scene in scenes
    ], axis=1)
    interval = np.quantile(differences[indices].mean(1), [0.025, 0.975]).tolist()
    no_regression = all(win["summary"][group]["unobserved_mse"] <= 1.02*lose["summary"][group]["unobserved_mse"]
                        for group in ("clean", "high_noise"))
    clear = gain >= 0.05 and interval[1] < 0 and no_regression
    return dict(winner=winner, relative_gain=gain, paired_video_95ci=interval,
                clean_high_noise_within_2pct=no_regression, clear_advantage=clear,
                w16_variants=[winner] if clear else ["x0", "epsilon"],
                rule="at least 5% mean unseen-MSE gain, paired video CI below zero, no >2% clean/high-noise regression")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,2,3,4")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    ROOT.mkdir(parents=True, exist_ok=True)
    selected = {}
    for name in ("x0", "epsilon"):
        path = f"experimental/noise_hvdit/w1_{name}.yaml"
        train(path)
        selected[name] = select(path)
    decision = compare(selected["x0"], selected["epsilon"])
    write_json_atomic(ROOT/"w1_comparison.json", decision)
    print("W1 comparison:", decision, flush=True)
    final_configs = [f"experimental/noise_hvdit/w1_{name}.yaml" for name in selected]
    for name in decision["w16_variants"]:
        report = selected[name]
        initialization = ROOT/f"w16_{name}_initialization.pth"
        if not initialization.exists():
            payload = torch.load(report["checkpoint"], map_location="cpu", weights_only=False)
            atomic_save(dict(model=payload[report["weights"]], source_selection=report), initialization)
            del payload
        path = f"experimental/noise_hvdit/w16_{name}.yaml"
        train(path, str(initialization))
        select(path)
        final_configs.append(path)
    # Test is report-only, after all training and validation decisions are fixed.
    for path in final_configs:
        report = select(path)
        result = Path(load_config(path).output)/"final_test.json"
        if not result.exists():
            execute("evaluate", "--config", path, "--checkpoint", report["checkpoint"],
                    "--weights", report["weights"], "--split", "test", "--output", result)
    write_json_atomic(ROOT/"pipeline_status.json", dict(state="complete", decision=decision))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json_atomic(ROOT/"pipeline_status.json", dict(state="failed", error=str(error)))
        raise
