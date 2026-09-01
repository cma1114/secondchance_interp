from __future__ import annotations

from scripts.check_canonical_headlines import collect_errors


def test_canonical_headlines_match_machine_readable_results():
    assert collect_errors() == []
