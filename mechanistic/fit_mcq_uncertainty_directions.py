from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


LAYERS = 64
QUANTILE = 0.25
RIDGE_ALPHA = 1000.0
PCA_COMPONENTS = 100
SEED = 42


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _metrics(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-300, None))).sum(axis=-1)
    ordered = np.sort(logits.astype(np.float64), axis=-1)
    gap = ordered[:, -1] - ordered[:, -2]
    return entropy.astype(np.float32), gap.astype(np.float32)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator < 1e-20:
        return float("nan")
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def _split_ids(path: Path, qids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    if "discovery_question_ids" in payload:
        discovery = set(str(value) for value in payload["discovery_question_ids"])
        confirmation = set(str(value) for value in payload["confirmation_question_ids"])
    elif "question_ids" in payload:
        discovery = set(str(value) for value in payload["question_ids"])
        confirmation = set(qids) - discovery
    else:
        raise ValueError(f"Unrecognized split plan: {path}")
    if discovery & confirmation or discovery | confirmation != set(qids):
        raise RuntimeError("Frozen split does not partition the state questions")
    return (
        np.asarray([qid in discovery for qid in qids], dtype=bool),
        np.asarray([qid in confirmation for qid in qids], dtype=bool),
    )


def _load_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(spec["output"])
    with np.load(output / "results.npz", allow_pickle=False) as loaded:
        qids = loaded["question_ids"].astype(str).tolist()
        first_logits = np.asarray(loaded["first_logits"], dtype=np.float32)
    first = np.load(output / "first_decision_residuals.npy", mmap_mode="r")
    second = np.load(output / "second_decision_residuals.npy", mmap_mode="r")
    if (
        first.shape[:2] != (len(qids), LAYERS)
        or second.shape[:3] != (2, len(qids), LAYERS)
        or second.shape[-1] != first.shape[-1]
        or first_logits.shape != (len(qids), 4)
    ):
        raise ValueError(f"{spec['slug']}: incompatible residual cache shape")
    discovery, confirmation = _split_ids(Path(spec["split_plan"]), qids)
    entropy, gap = _metrics(first_logits)
    return {
        "spec": spec,
        "qids": qids,
        "first": first,
        "second": second,
        "entropy": entropy,
        "gap": gap,
        "discovery": discovery,
        "confirmation": confirmation,
    }


def _mean_difference(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    order = np.argsort(target, kind="stable")
    count = max(1, int(np.floor(len(order) * QUANTILE)))
    direction = x[order[-count:]].mean(axis=0) - x[order[:count]].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError("Degenerate entropy mean-difference direction")
    return (direction / norm).astype(np.float32)


def _fit_ridge_pair(x: np.ndarray, entropy: np.ndarray, gap: np.ndarray):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float32)
    scaler = StandardScaler().fit(x)
    scaler.scale_ = np.maximum(np.asarray(scaler.scale_, dtype=np.float64), 1e-6)
    scaled = scaler.transform(x)
    components = min(PCA_COMPONENTS, x.shape[0], x.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=SEED).fit(scaled)
    reduced = pca.transform(scaled)
    models = {}
    directions = {}
    for name, target in (("entropy", entropy), ("logit_gap", gap)):
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(reduced, target)
        raw = pca.inverse_transform(ridge.coef_[None])[0] / scaler.scale_
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm < 1e-8:
            raise RuntimeError(f"Degenerate ridge {name} direction")
        directions[name] = (raw / norm).astype(np.float32)
        models[name] = ridge
    return scaler, pca, models, directions


def _predict(x: np.ndarray, scaler: Any, pca: Any, ridge: Any) -> np.ndarray:
    return ridge.predict(pca.transform(scaler.transform(np.asarray(x, dtype=np.float32))))


def fit(specs_path: Path) -> None:
    payload = json.loads(specs_path.read_text())
    specs = payload["datasets"]
    if len(specs) != 2:
        raise ValueError("The frozen cross-dataset design requires exactly two datasets")
    datasets = [_load_dataset(spec) for spec in specs]
    widths = {int(data["first"].shape[-1]) for data in datasets}
    if len(widths) != 1:
        raise ValueError("Datasets use different model widths")
    width = widths.pop()
    primary = np.empty((2, LAYERS, width), dtype=np.float32)
    ridge_entropy = np.empty_like(primary)
    ridge_gap = np.empty_like(primary)
    means = np.empty((2, LAYERS), dtype=np.float32)
    stds = np.empty((2, LAYERS), dtype=np.float32)
    summary: dict[str, Any] = {
        "model_name": payload["model_name"],
        "primary_direction": "discovery high-entropy quartile mean minus low-entropy quartile mean",
        "quantile": QUANTILE,
        "ridge_alpha": RIDGE_ALPHA,
        "pca_components": PCA_COMPONENTS,
        "datasets": {},
    }

    for source_index, source in enumerate(datasets):
        target = datasets[1 - source_index]
        source_name = source["spec"]["slug"]
        target_name = target["spec"]["slug"]
        disc = source["discovery"]
        conf = source["confirmation"]
        layer_rows: list[dict[str, Any]] = []
        cross_rows: list[dict[str, Any]] = []
        for layer in range(LAYERS):
            x_disc = np.asarray(source["first"][disc, layer], dtype=np.float32)
            x_conf = np.asarray(source["first"][conf, layer], dtype=np.float32)
            direction = _mean_difference(x_disc, source["entropy"][disc])
            primary[source_index, layer] = direction
            disc_projection = x_disc @ direction
            means[source_index, layer] = float(disc_projection.mean())
            stds[source_index, layer] = max(float(disc_projection.std()), 1e-6)

            scaler, pca, models, ridge_dirs = _fit_ridge_pair(
                x_disc, source["entropy"][disc], source["gap"][disc]
            )
            ridge_entropy[source_index, layer] = ridge_dirs["entropy"]
            ridge_gap[source_index, layer] = ridge_dirs["logit_gap"]
            entropy_prediction = _predict(x_conf, scaler, pca, models["entropy"])
            gap_prediction = _predict(x_conf, scaler, pca, models["logit_gap"])
            conf_projection = x_conf @ direction
            layer_rows.append(
                {
                    "layer": layer + 1,
                    "mean_diff_entropy_projection_r": _corr(
                        source["entropy"][conf], conf_projection
                    ),
                    "ridge_entropy_r": _corr(source["entropy"][conf], entropy_prediction),
                    "ridge_entropy_r2": _r2(source["entropy"][conf], entropy_prediction),
                    "ridge_logit_gap_r": _corr(source["gap"][conf], gap_prediction),
                    "ridge_logit_gap_r2": _r2(source["gap"][conf], gap_prediction),
                    "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                }
            )

            target_mask = target["confirmation"]
            x_cross = np.asarray(target["first"][target_mask, layer], dtype=np.float32)
            cross_projection = x_cross @ direction
            cross_entropy_prediction = _predict(x_cross, scaler, pca, models["entropy"])
            cross_gap_prediction = _predict(x_cross, scaler, pca, models["logit_gap"])
            cross_rows.append(
                {
                    "layer": layer + 1,
                    "mean_diff_entropy_projection_r": _corr(
                        target["entropy"][target_mask], cross_projection
                    ),
                    "ridge_entropy_r": _corr(
                        target["entropy"][target_mask], cross_entropy_prediction
                    ),
                    "ridge_logit_gap_r": _corr(
                        target["gap"][target_mask], cross_gap_prediction
                    ),
                }
            )
            if layer == 0 or (layer + 1) % 8 == 0:
                print(f"{payload['model_name']} {source_name} directions: {layer + 1}/64", flush=True)
        summary["datasets"][source_name] = {
            "n_questions": len(source["qids"]),
            "n_discovery": int(disc.sum()),
            "n_confirmation": int(conf.sum()),
            "within_dataset": layer_rows,
            f"cross_to_{target_name}": cross_rows,
        }

    output = Path(payload["direction_output"])
    output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output / "directions.npz",
        dataset_slugs=np.asarray([data["spec"]["slug"] for data in datasets]),
        entropy_mean_diff=primary,
        ridge_entropy=ridge_entropy,
        ridge_logit_gap=ridge_gap,
        first_projection_discovery_mean=means,
        first_projection_discovery_std=stds,
    )

    for target_index, target in enumerate(datasets):
        other_index = 1 - target_index
        first_projection = np.empty((len(target["qids"]), LAYERS), dtype=np.float32)
        within = np.empty((2, len(target["qids"]), LAYERS), dtype=np.float32)
        cross = np.empty_like(within)
        for layer in range(LAYERS):
            first_projection[:, layer] = (
                np.asarray(target["first"][:, layer], dtype=np.float32)
                @ primary[target_index, layer]
            )
            for condition in range(2):
                states = np.asarray(target["second"][condition, :, layer], dtype=np.float32)
                within[condition, :, layer] = states @ primary[target_index, layer]
                cross[condition, :, layer] = states @ primary[other_index, layer]
        within_z = (within - means[target_index][None, None]) / stds[target_index][None, None]
        cross_z = (cross - means[other_index][None, None]) / stds[other_index][None, None]
        _atomic_npz(
            output / f"{target['spec']['slug']}_projections.npz",
            question_ids=np.asarray(target["qids"]),
            conditions=np.asarray(["game", "neutral"]),
            discovery_mask=target["discovery"],
            confirmation_mask=target["confirmation"],
            first_entropy=target["entropy"],
            first_logit_gap=target["gap"],
            first_projection=first_projection,
            second_projection_within=within,
            second_projection_cross=cross,
            second_projection_within_z=within_z.astype(np.float32),
            second_projection_cross_z=cross_z.astype(np.float32),
        )
    summary["all_outputs_finite"] = bool(
        np.isfinite(primary).all()
        and np.isfinite(ridge_entropy).all()
        and np.isfinite(ridge_gap).all()
        and np.isfinite(means).all()
        and np.isfinite(stds).all()
    )
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "all_outputs_finite": summary["all_outputs_finite"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    fit(parser.parse_args().specs)


if __name__ == "__main__":
    main()
