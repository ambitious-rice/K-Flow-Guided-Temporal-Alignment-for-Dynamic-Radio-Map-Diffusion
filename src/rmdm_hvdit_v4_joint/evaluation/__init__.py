"""HV-DiT v4 joint validation and gate logic."""

from .evaluator import combine_results, evaluate_stage_a
from .gate import evaluate_t1_gate, validate_t1_reference
from .comparison import compare_with_factorized_baseline

__all__ = [
    "combine_results",
    "compare_with_factorized_baseline",
    "evaluate_stage_a",
    "evaluate_t1_gate",
    "validate_t1_reference",
]
