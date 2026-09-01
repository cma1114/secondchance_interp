from mechanistic.run_question_stem_access_factorial import (
    _merge_query_specs,
    _queries_with_sources,
)


def test_query_specs_keep_only_causal_sources() -> None:
    result = _queries_with_sources([5, 8, 12], [2, 6, 10])
    assert result == {5: [2], 8: [2, 6], 12: [2, 6, 10]}


def test_joint_specs_are_exact_union() -> None:
    first = {8: [1, 2], 12: [1, 2]}
    second = {10: [6], 12: [6, 9]}
    assert _merge_query_specs(first, second) == {
        8: [1, 2],
        10: [6],
        12: [1, 2, 6, 9],
    }
