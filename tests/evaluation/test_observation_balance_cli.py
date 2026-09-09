"""Exercise local planning, resume, and selection without a torch environment."""

import argparse
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validation_cli", ROOT / "scripts/run_observation_balance_validation.py"
)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)
SUITE = ROOT / "configs/evaluation/observation_balance_validation_v1.yaml"


class ValidationCliTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = argparse.Namespace(
            suite=str(SUITE), output_root=str(self.root), candidate_id="candidate",
            baseline_id="baseline", variant="w1", config="model.yaml",
            checkpoint="model.pth", manifest=str(self.root / "manifest.json"),
            stages="fast_w1", gpus="0,1,2,3,4,5,6,7", python="python3", base_port=29700,
            rerun=False,
        )
        CLI.write_json_atomic(self.args.manifest, {
            "stage_a": {"videos": [{"scene_id": "a", "video_id": "a/video_01"}]}
        })
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(redirect_stdout(io.StringIO()))

    def test_prepare_is_repeatable_and_run_resumes_completed_cells(self):
        CLI.prepare(self.args)
        root = self.root / "candidate"
        before = {p: p.read_bytes() for p in root.rglob("*.json")}
        CLI.prepare(self.args)
        self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*.json")})
        plan = CLI.read_json(root / "plan.json")
        cell = CLI.read_json(plan["cell_files"][0])
        CLI.write_json_atomic(cell["result"], {"schema": CLI.RESULT_SCHEMA})
        with patch.object(CLI.subprocess, "run") as launch:
            CLI.run(self.args)
            self.assertEqual(launch.call_count, 3)
            for call in launch.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--num_processes") + 1], "8")
                self.assertIn(command[-1], plan["cell_files"][1:])
            self.args.rerun = True
            launch.reset_mock()
            CLI.run(self.args)
            self.assertEqual(launch.call_count, 4)

    def write_summary(self, candidate_id, variant):
        rows = []
        for cell in CLI.expand_cells(CLI.load_suite(SUITE)):
            if cell["variant"] != variant:
                continue
            for method in cell["methods"]:
                value = 1.0
                if candidate_id == "candidate":
                    value = 0.8 if cell["noise_std"] == 0 else 1.01
                    if method == "known_noise_da":
                        value *= 0.95
                rows.append({**cell, "method": method, "unobserved_free_space_nmse": value})
        summary = {"candidate_id": candidate_id, "rows": rows}
        CLI.write_json_atomic(self.root / candidate_id / "summary.json", summary)
        return summary

    def test_complete_w1_and_w16_comparisons_reuse_no_da_results(self):
        for variant in ("w1", "w16"):
            with self.subTest(variant=variant):
                self.args.variant = variant
                self.write_summary("baseline", variant)
                self.write_summary("candidate", variant)
                CLI.compare(self.args)
                result = CLI.read_json(self.root / "candidate/comparison_to_baseline.json")
                self.assertTrue(result["passed"])
                self.assertAlmostEqual(result["clean_relative_improvement"], 0.2)
                self.assertAlmostEqual(result["noisy_relative_regression"], 0.01)
                self.assertAlmostEqual(result["known_noise_da_relative_improvement"], 0.05)
                self.assertEqual(len(result["known_noise_da_by_rate"]), 2 if variant == "w1" else 3)

    def test_inclusive_selection_thresholds(self):
        self.write_summary("baseline", "w1")
        for offset, expected in ((0.0, True), (1e-6, False)):
            with self.subTest(offset=offset):
                summary = self.write_summary("candidate", "w1")
                for row in summary["rows"]:
                    if row["noise_std"] == 0:
                        value = 0.9 + offset
                    elif row["method"] == "known_noise_da":
                        value = (1.02 + offset) * (0.98 + offset)
                    else:
                        value = 1.02 + offset
                    row["unobserved_free_space_nmse"] = value
                CLI.write_json_atomic(self.root / "candidate/summary.json", summary)
                CLI.compare(self.args)
                result = CLI.read_json(self.root / "candidate/comparison_to_baseline.json")
                self.assertEqual(result["passed"], expected)
                for gate in ("clean", "robustness", "da_mean"):
                    self.assertEqual(result["gates"][gate], expected, gate)

    def test_compare_rejects_missing_rates_in_each_required_selection(self):
        self.write_summary("baseline", "w1")
        for candidate_id, stage, sigma, method in (
            ("candidate", "full_w1", 0, "no_da"),
            ("baseline", "full_w1", 0.05, "no_da"),
            ("candidate", "w1_da_gate", 0.05, "known_noise_da"),
        ):
            with self.subTest(candidate_id=candidate_id, stage=stage, sigma=sigma):
                self.write_summary("candidate", "w1")
                self.write_summary("baseline", "w1")
                summary = self.write_summary(candidate_id, "w1")
                summary["rows"] = [row for row in summary["rows"] if not (
                    row["stage"] == stage and row["noise_std"] == sigma
                    and row["method"] == method and row["rate"] == 1
                )]
                CLI.write_json_atomic(self.root / candidate_id / "summary.json", summary)
                with self.assertRaisesRegex(ValueError, "incomplete rates"):
                    CLI.compare(self.args)


if __name__ == "__main__":
    unittest.main()
