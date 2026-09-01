"""Inflate the selected T1 checkpoint into a CPU-audited W16 artifact."""

from __future__ import annotations

import argparse
import json

from rmdm_hvdit_v4_joint.transfer import inflate_t1_checkpoint

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--t1-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _, _ = load_arguments(args)
    audit = inflate_t1_checkpoint(config, t1_checkpoint=args.t1_checkpoint, output_path=args.output)
    print(json.dumps({"audit_sha256": audit["audit_sha256"], "records": audit["record_count"]}), flush=True)


if __name__ == "__main__":
    main()
