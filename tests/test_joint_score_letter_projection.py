from __future__ import annotations

import numpy as np
import torch

from mechanistic.run_option_newline_all_four_projection import (
    JOINT_MODES,
    _initialize,
    _project_answer_identity,
)


def test_answer_identity_projection_removes_entire_basis() -> None:
    generator = torch.Generator().manual_seed(7)
    current = torch.randn(4, 32, generator=generator)
    raw_basis = torch.randn(3, 32, generator=generator)
    _q, _r = torch.linalg.qr(raw_basis.T)
    basis = _q[:, :3].T.contiguous()

    projected, before, after = _project_answer_identity(current, basis)

    assert torch.allclose(before, current @ basis.T)
    assert torch.max(torch.abs(after)).item() < 2e-6
    assert torch.allclose(
        current - projected,
        (current @ basis.T) @ basis,
        atol=2e-6,
        rtol=0,
    )


def test_joint_checkpoint_has_all_factorial_modes(tmp_path) -> None:
    qids = ["q0", "q1"]
    split = np.asarray(["discovery", "confirmation"])
    arrays = _initialize(tmp_path / "results.npz", qids, split, JOINT_MODES)

    assert arrays["logits"].shape == (2, 5, 2, 4)
    assert arrays["pre_score"].shape == (2, 5, 2, 64, 4)
    assert arrays["decision_pre_ad_norm"].shape == (2, 5, 2, 64)
    assert arrays["first_decision_logits"].shape == (2, 4)
