"""Migrate an audited T1 checkpoint across execution-only code changes."""

from __future__ import annotations

import argparse
import json

from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_hvdit_v4_joint.transfer.migrate_execution import (
    migrate_pipeline_state,
    migrate_t1_execution_checkpoint,
)

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--pipeline-state", default="")
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    report = migrate_t1_execution_checkpoint(
        args.source,
        args.output,
        config,
        config_path=config_path,
        repository_root=root,
        expected_global_step=args.expected_global_step,
    )
    write_json_atomic(args.report, report)
    if args.pipeline_state:
        migrate_pipeline_state(args.pipeline_state, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
