#!/usr/bin/env python3
"""Build and validate the deliberately small GitHub payload for this repo.

The workspace is a research working directory, not an artifact store suitable
for wholesale Git tracking. This tool discovers a source-first payload, adds
only small canonical artifacts linked by the root README, scans it for secrets,
records omitted large artifacts, and can stage the exact audited set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = REPO_ROOT / "scripts" / "github_payload_policy.json"
PAYLOAD_MANIFEST_PATH = Path("version_control/github_payload_manifest.json")
EXCLUDED_MANIFEST_PATH = Path("version_control/excluded_artifacts.json")

MARKDOWN_LINK_RE = re.compile(r"\]\(([^)]+)\)")
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
)
TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
GENERIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b"
    r"\s*[:=]\s*(['\"])([^'\"]+)\2"
)
SAFE_SECRET_LITERALS = {
    "EMPTY",
    "REDACTED",
    "YOUR_API_KEY",
    "YOUR_TOKEN",
}


class PayloadError(RuntimeError):
    """Raised when the proposed Git payload violates policy."""


@dataclass(frozen=True)
class ReadmeSelection:
    included: tuple[Path, ...]
    skipped: tuple[dict[str, object], ...]
    directory_links: tuple[str, ...]


def relative_path(root: Path, path: Path) -> Path:
    # Use lexical absolute paths here. Resolving symlinks would make a harmless
    # repository entry such as .venv/bin/python appear to escape the root before
    # the caller has a chance to exclude the environment. Payload symlinks are
    # rejected explicitly by validate_payload.
    resolved_root = root.absolute()
    resolved_path = path.absolute()
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PayloadError(f"Path escapes repository: {path}") from exc


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("version") != 1:
        raise PayloadError(f"Unsupported policy version: {policy.get('version')}")
    return policy


def is_forbidden(path: Path, policy: dict[str, object]) -> bool:
    rendered = path.as_posix()
    if rendered in set(policy["forbidden_exact_paths"]):
        return True
    if any(part in set(policy["forbidden_path_parts"]) for part in path.parts):
        return True
    return path.suffix.lower() in set(policy["forbidden_suffixes"])


def clean_markdown_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if not target or target.startswith("#"):
        return None
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        return None
    target = target.split("#", 1)[0]
    target = unquote(target)
    return target or None


def readme_link_selection(
    root: Path, policy: dict[str, object]
) -> ReadmeSelection:
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    allowed_suffixes = set(policy["readme_link_allowed_suffixes"])
    maximum = int(policy["maximum_file_bytes"])
    excluded_prefixes = tuple(policy["readme_link_excluded_prefixes"])

    included: set[Path] = set()
    skipped: list[dict[str, object]] = []
    directory_links: list[str] = []

    raw_targets = sorted(set(MARKDOWN_LINK_RE.findall(text)))
    for raw_target in raw_targets:
        target = clean_markdown_target(raw_target)
        if target is None:
            continue
        candidate = root / target
        try:
            rel = relative_path(root, candidate)
        except PayloadError:
            skipped.append({"path": target, "reason": "outside_repository"})
            continue
        rendered = rel.as_posix()
        if rendered.startswith(excluded_prefixes):
            skipped.append({"path": rendered, "reason": "excluded_prefix"})
            continue
        if candidate.is_dir():
            directory_links.append(rendered)
            members = sorted(p for p in candidate.rglob("*") if p.is_file())
        elif candidate.is_file():
            members = [candidate]
        else:
            skipped.append({"path": rendered, "reason": "missing"})
            continue

        for member in members:
            member_rel = relative_path(root, member)
            size = member.stat().st_size
            if member_rel.suffix.lower() not in allowed_suffixes:
                skipped.append(
                    {
                        "path": member_rel.as_posix(),
                        "reason": "readme_suffix_not_allowed",
                        "bytes": size,
                    }
                )
            elif size > maximum:
                skipped.append(
                    {
                        "path": member_rel.as_posix(),
                        "reason": "readme_file_over_size_limit",
                        "bytes": size,
                    }
                )
            elif is_forbidden(member_rel, policy):
                skipped.append(
                    {
                        "path": member_rel.as_posix(),
                        "reason": "forbidden_by_policy",
                        "bytes": size,
                    }
                )
            else:
                included.add(member_rel)

    return ReadmeSelection(
        included=tuple(sorted(included)),
        skipped=tuple(sorted(skipped, key=lambda item: str(item["path"]))),
        directory_links=tuple(sorted(directory_links)),
    )


def discover_payload(
    root: Path, policy: dict[str, object]
) -> tuple[set[Path], ReadmeSelection]:
    candidates: set[Path] = set()
    root_suffixes = set(policy["root_allowed_suffixes"])
    root_names = set(policy["root_allowed_names"])

    for path in root.iterdir():
        if not path.is_file():
            continue
        rel = relative_path(root, path)
        if path.name in root_names or path.suffix.lower() in root_suffixes:
            candidates.add(rel)

    for tree_name, suffixes in policy["source_trees"].items():
        tree = root / tree_name
        if not tree.exists():
            continue
        allowed = set(suffixes)
        for path in tree.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed:
                candidates.add(relative_path(root, path))

    compiled_prefix = str(policy["compiled_results_directory_prefix"])
    for tree in root.glob(f"{compiled_prefix}*"):
        if not tree.is_dir():
            continue
        for path in tree.rglob("*.json"):
            if path.is_file():
                candidates.add(relative_path(root, path))

    explicit_paths = {Path(rendered) for rendered in policy["explicit_includes"]}
    for rendered in policy["explicit_includes"]:
        path = root / rendered
        if not path.is_file():
            raise PayloadError(f"Explicitly included file is missing: {rendered}")
        candidates.add(Path(rendered))

    readme_selection = readme_link_selection(root, policy)
    readme_selection = ReadmeSelection(
        included=readme_selection.included,
        skipped=tuple(
            item
            for item in readme_selection.skipped
            if Path(str(item["path"])) not in explicit_paths
        ),
        directory_links=readme_selection.directory_links,
    )
    candidates.update(readme_selection.included)
    return candidates, readme_selection


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_text_for_secrets(path: Path) -> list[str]:
    if path.suffix.lower() in {".png"}:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    findings: list[str] = []
    if PRIVATE_KEY_RE.search(text):
        findings.append("private_key_block")
    for pattern in TOKEN_PATTERNS:
        if pattern.search(text):
            findings.append(f"token_pattern:{pattern.pattern}")
    for match in GENERIC_SECRET_ASSIGNMENT_RE.finditer(text):
        value = match.group(3).strip()
        if value in SAFE_SECRET_LITERALS:
            continue
        if value.startswith(("${", "<", "YOUR_", "REDACTED")):
            continue
        findings.append(f"literal_assignment:{match.group(1).lower()}")
    return sorted(set(findings))


def validate_payload(
    root: Path, candidates: set[Path], policy: dict[str, object]
) -> None:
    failures: list[str] = []
    maximum = int(policy["maximum_file_bytes"])
    for rel in sorted(candidates):
        path = root / rel
        if not path.is_file():
            failures.append(f"missing payload file: {rel.as_posix()}")
            continue
        if path.is_symlink():
            failures.append(f"symlink not allowed: {rel.as_posix()}")
        if is_forbidden(rel, policy):
            failures.append(f"forbidden payload file: {rel.as_posix()}")
        if path.stat().st_size > maximum:
            failures.append(
                f"payload file exceeds {maximum} bytes: {rel.as_posix()} "
                f"({path.stat().st_size})"
            )
        for finding in scan_text_for_secrets(path):
            failures.append(f"possible secret in {rel.as_posix()}: {finding}")
    for rendered in policy["local_only_artifacts"]:
        rel = Path(rendered)
        if not (root / rel).is_file():
            failures.append(f"declared local-only artifact is missing: {rendered}")
        if rel in candidates:
            failures.append(f"local-only artifact entered payload: {rendered}")
    if failures:
        raise PayloadError("\n".join(failures))


def write_excluded_manifest(
    root: Path,
    candidates: set[Path],
    policy: dict[str, object],
    readme_selection: ReadmeSelection,
) -> None:
    destination = root / EXCLUDED_MANIFEST_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    threshold = int(policy["large_artifact_threshold_bytes"])
    skip_parts = {".git", ".venv", "venv", "__pycache__"}
    exact_sensitive = set(policy["forbidden_exact_paths"])
    known_payload = set(candidates) | {PAYLOAD_MANIFEST_PATH, EXCLUDED_MANIFEST_PATH}
    top_level = Counter()
    suffixes = Counter()
    excluded_count = 0
    excluded_bytes = 0
    large_files: list[dict[str, object]] = []

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = relative_path(root, path)
        if any(part in skip_parts for part in rel.parts):
            continue
        if rel in known_payload:
            continue
        size = path.stat().st_size
        excluded_count += 1
        excluded_bytes += size
        top_level[rel.parts[0]] += size
        suffixes[rel.suffix.lower() or "[none]"] += size
        if size >= threshold and rel.as_posix() not in exact_sensitive:
            large_files.append(
                {
                    "path": rel.as_posix(),
                    "bytes": size,
                    "sha256": sha256_file(path),
                }
            )

    readme_omissions = []
    for item in readme_selection.skipped:
        record = dict(item)
        omitted_path = root / str(item["path"])
        if omitted_path.is_file() and omitted_path.as_posix() not in exact_sensitive:
            record["sha256"] = sha256_file(omitted_path)
        readme_omissions.append(record)
    recorded_paths = {str(item["path"]) for item in readme_omissions}
    for rendered in policy["local_only_artifacts"]:
        if rendered in recorded_paths:
            continue
        omitted_path = root / rendered
        record = {
            "path": rendered,
            "reason": "declared_local_only",
            "bytes": omitted_path.stat().st_size,
            "sha256": sha256_file(omitted_path),
        }
        readme_omissions.append(record)
    readme_omissions.sort(key=lambda item: str(item["path"]))

    document = {
        "schema_version": 1,
        "purpose": (
            "Inventory of large or generated local artifacts intentionally "
            "excluded from the GitHub repository. It contains no artifact data."
        ),
        "payload_policy": "scripts/github_payload_policy.json",
        "excluded_file_count": excluded_count,
        "excluded_bytes": excluded_bytes,
        "excluded_gib": round(excluded_bytes / (1024**3), 6),
        "excluded_bytes_by_top_level_path": dict(sorted(top_level.items())),
        "excluded_bytes_by_suffix": dict(sorted(suffixes.items())),
        "large_files_at_or_above_threshold": large_files,
        "readme_linked_local_only_artifacts": readme_omissions,
        "notes": [
            "Credential files and virtual environments are never hashed or listed as artifacts.",
            "Small excluded intermediates are summarized by location and suffix rather than listed individually.",
            "This manifest is provenance, not a promise that every excluded file will be archived externally."
        ],
    }
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def write_payload_manifest(
    root: Path,
    candidates: set[Path],
    readme_selection: ReadmeSelection,
) -> None:
    destination = root / PAYLOAD_MANIFEST_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_members = sorted(candidates - {PAYLOAD_MANIFEST_PATH})
    records = []
    total_bytes = 0
    for rel in manifest_members:
        path = root / rel
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": rel.as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    document = {
        "schema_version": 1,
        "policy": "scripts/github_payload_policy.json",
        "manifest_self_excluded_from_hash_list": PAYLOAD_MANIFEST_PATH.as_posix(),
        "file_count_excluding_this_manifest": len(records),
        "bytes_excluding_this_manifest": total_bytes,
        "mib_excluding_this_manifest": round(total_bytes / (1024**2), 6),
        "readme_linked_directory_targets": list(readme_selection.directory_links),
        "readme_linked_files_skipped": list(readme_selection.skipped),
        "files": records,
    }
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def regenerate_manifests(
    root: Path, policy: dict[str, object]
) -> tuple[set[Path], ReadmeSelection]:
    candidates, selection = discover_payload(root, policy)
    write_excluded_manifest(root, candidates, policy, selection)
    candidates, selection = discover_payload(root, policy)
    write_payload_manifest(root, candidates, selection)
    candidates, selection = discover_payload(root, policy)
    validate_payload(root, candidates, policy)
    return candidates, selection


def validate_payload_manifest(root: Path, candidates: set[Path]) -> None:
    path = root / PAYLOAD_MANIFEST_PATH
    if not path.is_file():
        raise PayloadError(f"Missing payload manifest: {PAYLOAD_MANIFEST_PATH}")
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_records = {
        rel.as_posix(): {
            "bytes": (root / rel).stat().st_size,
            "sha256": sha256_file(root / rel),
        }
        for rel in candidates
        if rel != PAYLOAD_MANIFEST_PATH
    }
    recorded = {item["path"]: item for item in document["files"]}
    if set(recorded) != set(expected_records):
        missing = sorted(set(expected_records) - set(recorded))
        extra = sorted(set(recorded) - set(expected_records))
        raise PayloadError(
            f"Payload manifest membership mismatch; missing={missing}, extra={extra}"
        )
    for rendered, expected in expected_records.items():
        item = recorded[rendered]
        if item["bytes"] != expected["bytes"] or item["sha256"] != expected["sha256"]:
            raise PayloadError(f"Stale payload manifest record: {rendered}")


def git_paths(root: Path, args: list[str]) -> set[Path]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return {Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item}


def stage_payload(root: Path, candidates: set[Path]) -> None:
    if not (root / ".git").exists():
        raise PayloadError("Git repository is not initialized")
    payload = b"\0".join(rel.as_posix().encode("utf-8") for rel in sorted(candidates)) + b"\0"
    subprocess.run(
        [
            "git",
            "add",
            "--force",
            "--pathspec-from-file=-",
            "--pathspec-file-nul",
        ],
        cwd=root,
        input=payload,
        check=True,
    )


def validate_initial_staging(root: Path, candidates: set[Path]) -> None:
    staged = git_paths(
        root,
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
    )
    if staged != candidates:
        missing = sorted(path.as_posix() for path in candidates - staged)
        extra = sorted(path.as_posix() for path in staged - candidates)
        raise PayloadError(
            f"Staged payload differs from audited payload; missing={missing}, extra={extra}"
        )


def print_report(root: Path, candidates: set[Path], selection: ReadmeSelection) -> None:
    rows = sorted(
        ((root / rel).stat().st_size, rel.as_posix()) for rel in candidates
    )
    total = sum(size for size, _ in rows)
    print(f"AUDIT PASS: {len(rows)} files, {total / (1024**2):.2f} MiB")
    print(
        f"README-linked: {len(selection.included)} included, "
        f"{len(selection.skipped)} skipped, "
        f"{len(selection.directory_links)} directory links expanded"
    )
    print("Largest payload files:")
    for size, rendered in reversed(rows[-20:]):
        print(f"  {size / (1024**2):8.2f} MiB  {rendered}")
    if selection.skipped:
        print("README-linked files intentionally omitted:")
        for item in selection.skipped:
            size = item.get("bytes")
            suffix = f" ({int(size) / (1024**2):.2f} MiB)" if size is not None else ""
            print(f"  {item['reason']}: {item['path']}{suffix}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Payload policy JSON",
    )
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="Regenerate payload and excluded-artifact manifests before auditing",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage exactly the audited payload (requires an initialized Git repository)",
    )
    parser.add_argument(
        "--check-initial-staging",
        action="store_true",
        help="Require the initial staged path set to equal the audited payload exactly",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy_path = args.policy
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    try:
        policy = load_policy(policy_path)
        if args.write_manifests:
            candidates, selection = regenerate_manifests(REPO_ROOT, policy)
        else:
            candidates, selection = discover_payload(REPO_ROOT, policy)
            validate_payload(REPO_ROOT, candidates, policy)
            validate_payload_manifest(REPO_ROOT, candidates)
        if args.stage:
            stage_payload(REPO_ROOT, candidates)
        if args.check_initial_staging:
            validate_initial_staging(REPO_ROOT, candidates)
        print_report(REPO_ROOT, candidates, selection)
    except (OSError, PayloadError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"AUDIT FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
