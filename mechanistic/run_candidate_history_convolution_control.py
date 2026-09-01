from __future__ import annotations

"""Minimal convolution-safe control for candidate-history relay mediation.

The parent Stage-B restorer preserves a selected relay token's lesioned local
output while restoring only the ordinary-attention K/V and recurrent GLA writes
seen downstream.  Qwen3.6-27B has a four-token causal GLA convolution, so the
final readout can still receive lesioned q/k/v input from the three immediately
preceding assistant-prefix tokens.  This control repeats the decisive cells
while leaving the exact three-token receptive-field suffix, plus a conservative
four-token suffix, free to recompute from restored earlier state.
"""

from pathlib import Path

from . import run_candidate_history_relay_mediation as base


CONV_KERNEL_DIM = 4
EXACT_FREE_SUFFIX = CONV_KERNEL_DIM - 1
CONSERVATIVE_FREE_SUFFIX = CONV_KERNEL_DIM
RESTORE_ALL_EXCEPT_LAST3_MASK = 32
RESTORE_ALL_EXCEPT_LAST4_MASK = 33

CONTROL_SCENARIOS = (
    ("none", 0, "none"),
    ("complete_matching_block", 0, "none"),
    ("complete_balanced_wrong_block", 0, "none"),
    ("complete_matching_block", 31, "both"),
    ("complete_matching_block", 15, "both"),
    ("complete_matching_block", RESTORE_ALL_EXCEPT_LAST3_MASK, "both"),
    ("complete_matching_block", RESTORE_ALL_EXCEPT_LAST4_MASK, "both"),
    ("complete_balanced_wrong_block", RESTORE_ALL_EXCEPT_LAST4_MASK, "both"),
)
CONTROL_SCENARIO_IDS = tuple(
    f"{source}__relay_{mask:02d}__{mechanism}"
    for source, mask, mechanism in CONTROL_SCENARIOS
)

_base_selected_relay_positions = base._selected_relay_positions
_base_relay_mask_label = base._relay_mask_label


def _convolution_safe_positions(
    groups_by_row: list[dict[str, list[int]]], relay_mask: int
) -> dict[int, list[int]]:
    if relay_mask not in {
        RESTORE_ALL_EXCEPT_LAST3_MASK,
        RESTORE_ALL_EXCEPT_LAST4_MASK,
    }:
        return _base_selected_relay_positions(groups_by_row, relay_mask)
    free_count = (
        EXACT_FREE_SUFFIX
        if relay_mask == RESTORE_ALL_EXCEPT_LAST3_MASK
        else CONSERVATIVE_FREE_SUFFIX
    )
    selected: dict[int, list[int]] = {}
    for row, groups in enumerate(groups_by_row):
        prefix = groups["final_assistant_prefix"]
        if len(prefix) <= free_count:
            raise RuntimeError("Assistant prefix is too short for convolution control")
        free = set(prefix[-free_count:])
        all_positions = {
            position
            for name in base.RELAY_GROUPS
            for position in groups[name]
        }
        kept = sorted(all_positions - free)
        if not kept or set(kept) & free:
            raise RuntimeError("Invalid convolution-safe relay partition")
        if free != set(range(prefix[-free_count], prefix[-1] + 1)):
            raise RuntimeError("Free convolution suffix is not contiguous")
        selected[row] = kept
    return selected


def _control_mask_label(mask: int) -> str:
    if mask == RESTORE_ALL_EXCEPT_LAST3_MASK:
        return "all_relays_except_last_3_prefix_tokens"
    if mask == RESTORE_ALL_EXCEPT_LAST4_MASK:
        return "all_relays_except_last_4_prefix_tokens"
    return _base_relay_mask_label(mask)


def run(args: object) -> None:
    base.SCENARIOS = CONTROL_SCENARIOS
    base.SCENARIO_IDS = CONTROL_SCENARIO_IDS
    # The parent's real no-lesion sentinel loop uses JOINT_RELAY_MASK. Point it
    # at the conservative convolution-safe operation being validated here.
    base.JOINT_RELAY_MASK = RESTORE_ALL_EXCEPT_LAST4_MASK
    base._selected_relay_positions = _convolution_safe_positions
    base._relay_mask_label = _control_mask_label
    base.EXPERIMENT_NAME = "candidate-history convolution-safe joint relay control"
    base.COMPLETE_MODEL_WORK = (
        "Per task: natural, matching and balanced-wrong source lesions, the "
        "existing all-five and all-except-prefix restorations, exact "
        "kernel-minus-one and conservative kernel-size free-prefix controls, "
        "and the balanced-wrong conservative control. Real no-source "
        "conservative restoration runs in ordinary-only, GLA-only, and both "
        "modes on the frozen sentinel cohorts."
    )
    base.RESTORATION_SEMANTICS = (
        "The exact control restores every causal-tail relay except the final "
        "three assistant-prefix tokens in the four-token GLA convolutional "
        "receptive field. The conservative control also leaves the immediately "
        "preceding prefix token free. Those suffix tokens recompute from the "
        "restored upstream path instead of having lesioned local outputs pinned."
    )
    base.UNINTERCEPTED_CHANNEL = (
        "The short GLA convolution remains unintercepted, but every prefix token "
        "whose q/k/v can directly enter the final readout is left free in the "
        "exact control; the conservative control adds one-token boundary slack."
    )
    base.run(args)


def build_parser():
    return base.build_parser()


if __name__ == "__main__":
    run(build_parser().parse_args())
