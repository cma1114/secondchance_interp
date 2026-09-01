from pathlib import Path

from scripts.audit_github_payload import (
    clean_markdown_target,
    discover_payload,
    is_forbidden,
    load_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_markdown_target_cleaning() -> None:
    assert clean_markdown_target("figures/example.png#panel") == "figures/example.png"
    assert clean_markdown_target("<outputs/a report/REPORT.md>") == (
        "outputs/a report/REPORT.md"
    )
    assert clean_markdown_target("https://example.com/result") is None
    assert clean_markdown_target("#section") is None


def test_sensitive_and_raw_paths_are_forbidden() -> None:
    policy = load_policy()
    assert is_forbidden(Path(".env"), policy)
    assert is_forbidden(Path(".vast-state/state.json"), policy)
    assert is_forbidden(Path("outputs/run/decision_residuals.npy"), policy)
    assert not is_forbidden(Path("mechanistic/analyze_result.py"), policy)


def test_repository_payload_is_source_first() -> None:
    policy = load_policy()
    candidates, selection = discover_payload(REPO_ROOT, policy)
    assert Path("README.md") in candidates
    assert Path("scripts/audit_github_payload.py") in candidates
    assert Path(".env") not in candidates
    assert not any(path.suffix == ".npy" for path in candidates)
    assert not any("__pycache__" in path.parts for path in candidates)
    assert not any(Path(path) in candidates for path in policy["local_only_artifacts"])
    assert selection.included
