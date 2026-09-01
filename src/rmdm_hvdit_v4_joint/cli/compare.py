"""Write the protocol-checked HV-DiT v4 joint versus factorized baseline report."""

from __future__ import annotations

import argparse
import json

from rmdm_hvdit_v4_joint.evaluation import compare_with_factorized_baseline
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _, _ = load_arguments(args)
    report = compare_with_factorized_baseline(
        factorized_run_dir=config.evaluation.factorized_baseline_run_dir,
        hvdit_selection_path=args.selection,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
