from types import SimpleNamespace

import torch

from mechanistic.runner_intervention import CpuAnswerLens, ReadoutAdd, ReadoutCapture


def _parts(hidden: int = 12):
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    norm = torch.nn.LayerNorm(hidden, elementwise_affine=True)
    head = torch.nn.Linear(hidden, 8, bias=False)
    return SimpleNamespace(layers=layers, final_norm=norm, output_head=head)


def test_calibrated_delta_hits_requested_contrast():
    torch.manual_seed(1)
    parts = _parts()
    lens = CpuAnswerLens(parts, [0, 1, 2, 3])
    residual = torch.randn(5, 12)
    target_letters = torch.tensor([0, 1, 2, 3, 1])
    target_change = torch.tensor([0.5, -0.3, 0.1, 0.8, -0.6])
    delta, achieved = lens.calibrated_delta(residual, target_letters, target_change, 5)
    assert delta.shape == residual.shape
    assert torch.allclose(achieved, target_change, atol=2e-4, rtol=2e-4)


def test_answer_orthogonal_control_matches_norm_and_first_order_scores():
    torch.manual_seed(2)
    parts = _parts()
    lens = CpuAnswerLens(parts, [0, 1, 2, 3])
    residual = torch.randn(4, 12)
    matched = torch.randn(4, 12) * 0.1
    control = lens.answer_orthogonal_control(residual, matched, seed=9)
    assert torch.allclose(control.norm(dim=-1), matched.norm(dim=-1), atol=1e-5)

    point = residual.detach().requires_grad_(True)
    logits = lens.logits(point)
    for letter in range(4):
        gradient = torch.autograd.grad(logits[:, letter].sum(), point, retain_graph=True)[0]
        assert torch.allclose((gradient * control).sum(dim=-1), torch.zeros(4), atol=2e-5)


def test_readout_hooks_capture_and_add_final_position():
    parts = _parts(hidden=4)
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    with ReadoutCapture(parts, 2, [2, 1]) as capture:
        _ = parts.layers[2](parts.layers[1](parts.layers[0](values)))
    assert torch.equal(capture.value, torch.stack([values[0, 2], values[1, 1]]))

    delta = torch.ones(2, 4)
    with ReadoutAdd(parts, 2, [2, 1], delta):
        changed = parts.layers[2](parts.layers[1](parts.layers[0](values)))
    assert torch.equal(changed[0, 2], values[0, 2] + 1)
    assert torch.equal(changed[1, 1], values[1, 1] + 1)
