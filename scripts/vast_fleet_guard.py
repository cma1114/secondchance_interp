#!/usr/bin/env python3
"""Audit Vast fleet state and enforce transition-aware repository policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "outputs/operations/vast_fleet_policy.json"
VASTAI = ROOT / ".venv/bin/vastai"
KEY_ENV = "SPAR_SPRING_EXTENSION_2026_VAST_KEY"


def load_instances() -> list[dict]:
    key = os.environ.get(KEY_ENV)
    if not key:
        raise RuntimeError(f"{KEY_ENV} is not set; refusing an unauthenticated audit")
    result = subprocess.run(
        [str(VASTAI), "show", "instances", "--raw", "--api-key", key],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Vast audit failed: {payload.get('msg', payload)}")
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Vast response type: {type(payload).__name__}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "prestart", "precreate", "finalize"))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--intended-instance", type=int)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text())
    instances = load_instances()
    retained = {int(row["instance_id"]): row for row in policy["retained_instances"]}
    # Explicitly user-authorized external reservations belong to other active
    # projects on the same Vast account.  They are shown in every audit but are
    # excluded from this repository's reservation/running caps and are never
    # eligible for Second Chance lifecycle actions.
    external = {
        int(row["instance_id"]): row
        for row in policy.get("external_instances", [])
    }
    replacement = policy.get("replacement_plan")
    managed = dict(retained)
    if replacement is not None and replacement.get("new_instance_id") is not None:
        managed[int(replacement["new_instance_id"])] = {
            "role": "replacement",
            "reason": replacement.get("reason", "temporary replacement"),
        }

    steady_target = int(policy["steady_state_target_total_instances"])
    total_cap = int(policy["max_temporary_instances_during_replacement"])
    running_cap = int(policy["default_max_running_instances"])
    exception = policy.get("temporary_exception")
    if exception is not None:
        total_cap = int(exception["max_total_instances"])
        running_cap = int(exception["max_running_instances"])

    current_ids = {int(row["id"]) for row in instances}
    project_instances = [
        row for row in instances if int(row["id"]) not in external
    ]
    running = [
        row for row in project_instances
        if row.get("actual_status") in {"running", "loading"}
    ]
    external_running = [
        row for row in instances if row.get("actual_status") in {"running", "loading"}
        and int(row["id"]) in external
    ]
    unmanaged = sorted(current_ids - set(managed) - set(external))
    missing = sorted(set(retained) - current_ids)
    violations: list[str] = []
    if len(project_instances) > total_cap:
        violations.append(
            f"project instances {len(project_instances)} exceeds cap {total_cap}"
        )
    if len(running) > running_cap:
        violations.append(f"running/loading instances {len(running)} exceeds cap {running_cap}")
    if unmanaged:
        violations.append(f"unmanaged instance IDs: {unmanaged}")
    if missing:
        violations.append(f"policy lists absent instance IDs: {missing}")
    if len(project_instances) > steady_target and replacement is None and exception is None:
        violations.append(
            f"fleet has {len(project_instances)} project instances above steady-state target "
            f"{steady_target} without a replacement plan"
        )

    if args.mode == "prestart":
        if args.intended_instance is None:
            violations.append("prestart requires --intended-instance ID")
        elif args.intended_instance not in current_ids:
            violations.append(f"intended instance {args.intended_instance} is not rented")
        other_running = sorted(
            int(row["id"])
            for row in running
            if int(row["id"]) != args.intended_instance
        )
        if other_running and exception is None:
            violations.append(f"other instances already running/loading: {other_running}")

    if args.mode == "precreate":
        if len(project_instances) >= total_cap:
            violations.append(
                f"no temporary replacement slot: {len(project_instances)}/{total_cap} project instances"
            )
        if running and exception is None:
            violations.append(
                "cancel queued/running retained requests before fresh creation; active IDs: "
                + str(sorted(int(row["id"]) for row in running))
            )
        if len(project_instances) >= steady_target:
            if replacement is None:
                violations.append("precreate at steady-state capacity requires replacement_plan")
            else:
                required = (
                    "reason",
                    "retirement_candidate_id",
                    "required_compatibility",
                    "post_validation_role",
                    "destruction_authorization_status",
                    "unique_remote_data_checked",
                )
                absent = [key for key in required if key not in replacement]
                if absent:
                    violations.append(f"replacement_plan missing fields: {absent}")
                candidate = replacement.get("retirement_candidate_id")
                if candidate is not None and int(candidate) not in retained:
                    violations.append(
                        f"retirement candidate {candidate} is not a retained instance"
                    )
    if args.mode == "finalize" and running:
        violations.append(
            "finalize requires every instance stopped; running/loading IDs: "
            + str(sorted(int(row["id"]) for row in running))
        )
    if args.mode == "finalize" and len(project_instances) > steady_target:
        violations.append(
            f"finalize requires at most {steady_target} project reservations; found {len(project_instances)}"
        )
    if args.mode == "finalize" and replacement is not None:
        violations.append("finalize requires replacement_plan to be resolved and cleared")

    rows = [
        {
            "id": int(row["id"]),
            "status": row.get("actual_status"),
            "label": row.get("label"),
            "role": (
                managed.get(int(row["id"]), {}).get("role")
                or ("EXTERNAL" if int(row["id"]) in external else "UNMANAGED")
            ),
        }
        for row in instances
    ]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "instances": rows,
                "external_running_ids": sorted(int(row["id"]) for row in external_running),
                "violations": violations,
            },
            indent=2,
        )
    )
    return 1 if violations else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
