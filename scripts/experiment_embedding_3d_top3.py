"""Compare local 3D layouts by recommendation-set hit rate.

This command keeps the learned 512D-to-32D recommendation coordinates fixed.
It trains only local 32D-to-3D layouts and never connects to or writes the DB.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.manifold import TSNE

from app.embedding_pipeline.core import (
    VectorArtifact,
    file_sha256,
    knn_indices,
    l2_normalize,
    load_vector_artifact,
    neighbor_recall,
    save_vector_artifact,
    write_json,
    write_vector_csv,
)

DEFAULT_ARTIFACT_DIR = Path("embedding-artifacts")
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "top3-3d-experiment"
TSNE_PERPLEXITIES = (3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0)
TSNE_EARLY_EXAGGERATIONS = (4.0, 12.0)
UMAP_NEIGHBORS = (3, 4, 5, 6, 7, 8, 10, 15, 30)
UMAP_MIN_DISTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True, slots=True)
class Trial:
    method: str
    mode: Literal["ball", "surface"]
    parameters: dict[str, float | int]
    against_32d: dict[str, Any]
    against_512d: dict[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the learned 32D embedding fixed and compare 3D layouts by "
            "whether their nearest points contain the ANN recommendations."
        )
    )
    parser.add_argument(
        "--embedding-512",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "embeddings-512.npz",
    )
    parser.add_argument(
        "--embedding-32",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "embeddings-32.npz",
    )
    parser.add_argument(
        "--cluster-surface",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "embedding-3d-cluster-surface.npz",
        help="Optional existing cluster-surface baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recommendation-k", type=int, default=5)
    parser.add_argument("--visual-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser


def _require_matching_rows(left: VectorArtifact, right: VectorArtifact) -> None:
    if not np.array_equal(left.cocktail_ids, right.cocktail_ids):
        raise ValueError("artifact cocktail IDs or order do not match")
    if left.cocktail_names != right.cocktail_names:
        raise ValueError("artifact cocktail names or order do not match")


def _ball(coordinates: np.ndarray) -> np.ndarray:
    centered = np.asarray(coordinates, dtype=np.float32)
    centered = centered - centered.mean(axis=0, keepdims=True)
    scale = float(np.quantile(np.linalg.norm(centered, axis=1), 0.99))
    if scale <= 1e-12:
        raise ValueError("3D coordinates collapsed")
    return np.asarray(centered / scale, dtype=np.float32)


def _surface(coordinates: np.ndarray) -> np.ndarray:
    centered = np.asarray(coordinates, dtype=np.float32)
    centered = centered - centered.mean(axis=0, keepdims=True)
    return l2_normalize(centered)


def _set_hit_metrics(
    reference_neighbors: np.ndarray,
    candidate_neighbors: np.ndarray,
) -> dict[str, Any]:
    overlaps = np.asarray(
        [
            len(set(reference).intersection(candidate))
            for reference, candidate in zip(
                reference_neighbors,
                candidate_neighbors,
                strict=True,
            )
        ],
        dtype=np.int32,
    )
    candidate_k = candidate_neighbors.shape[1]
    return {
        "any_hit_rate": float(np.mean(overlaps >= 1)),
        "mean_match_count": float(np.mean(overlaps)),
        "mean_candidate_match_rate": float(np.mean(overlaps / candidate_k)),
        "all_candidates_match_rate": float(np.mean(overlaps == candidate_k)),
        "distribution": {
            str(count): int(np.sum(overlaps == count))
            for count in range(candidate_k + 1)
        },
    }


def _strict_recall_metrics(
    reference_vectors: np.ndarray,
    coordinates: np.ndarray,
    *,
    k: int,
) -> dict[str, float]:
    reference = knn_indices(reference_vectors, k, metric="cosine")
    candidate = knn_indices(coordinates, k, metric="euclidean")
    overlaps = np.asarray(
        [
            len(set(expected).intersection(actual))
            for expected, actual in zip(reference, candidate, strict=True)
        ],
        dtype=np.int32,
    )
    return {
        f"recall@{k}": float(np.mean(overlaps / k)),
        f"any_hit@{k}": float(np.mean(overlaps >= 1)),
    }


def _evaluate(
    coordinates: np.ndarray,
    *,
    method: str,
    mode: Literal["ball", "surface"],
    parameters: dict[str, float | int],
    recommendation_32: np.ndarray,
    recommendation_512: np.ndarray,
    vectors_32: np.ndarray,
    vectors_512: np.ndarray,
    visual_k: int,
) -> Trial:
    candidate = knn_indices(coordinates, visual_k, metric="euclidean")
    against_32d = {
        **_set_hit_metrics(recommendation_32, candidate),
        **_strict_recall_metrics(vectors_32, coordinates, k=visual_k),
    }
    against_512d = {
        **_set_hit_metrics(recommendation_512, candidate),
        **_strict_recall_metrics(vectors_512, coordinates, k=visual_k),
    }
    return Trial(
        method=method,
        mode=mode,
        parameters=parameters,
        against_32d=against_32d,
        against_512d=against_512d,
    )


def _rank(trial: Trial) -> tuple[float, float, float, float]:
    strict_recall = next(
        float(value)
        for key, value in trial.against_32d.items()
        if key.startswith("recall@")
    )
    return (
        float(trial.against_32d["any_hit_rate"]),
        float(trial.against_32d["mean_match_count"]),
        float(trial.against_32d["all_candidates_match_rate"]),
        strict_recall,
    )


def _fit_tsne(
    vectors_32: np.ndarray,
    *,
    perplexity: float,
    early_exaggeration: float,
    seed: int,
) -> np.ndarray:
    return np.asarray(
        TSNE(
            n_components=3,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate="auto",
            max_iter=2000,
            metric="cosine",
            init="random",
            random_state=seed,
            method="barnes_hut",
            angle=0.3,
            n_jobs=1,
        ).fit_transform(vectors_32),
        dtype=np.float32,
    )


def _fit_umap(
    vectors_32: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "umap-learn is required; install requirements-embedding.txt"
        ) from error
    return np.asarray(
        umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            output_metric="euclidean",
            random_state=seed,
            transform_seed=seed,
            n_epochs=1000,
            init="spectral",
            low_memory=True,
            n_jobs=1,
            verbose=False,
        ).fit_transform(vectors_32),
        dtype=np.float32,
    )


def _run_trials(
    embedding_32: VectorArtifact,
    embedding_512: VectorArtifact,
    *,
    recommendation_k: int,
    visual_k: int,
    seed: int,
) -> tuple[
    list[tuple[np.ndarray, Trial]],
    list[tuple[np.ndarray, Trial]],
]:
    vectors_32 = l2_normalize(embedding_32.vectors)
    vectors_512 = l2_normalize(embedding_512.vectors)
    recommendation_32 = knn_indices(
        vectors_32,
        recommendation_k,
        metric="cosine",
    )
    recommendation_512 = knn_indices(
        vectors_512,
        recommendation_k,
        metric="cosine",
    )
    ball_trials: list[tuple[np.ndarray, Trial]] = []
    surface_trials: list[tuple[np.ndarray, Trial]] = []

    for perplexity in TSNE_PERPLEXITIES:
        for early_exaggeration in TSNE_EARLY_EXAGGERATIONS:
            raw = _fit_tsne(
                vectors_32,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                seed=seed,
            )
            parameters: dict[str, float | int] = {
                "perplexity": perplexity,
                "early_exaggeration": early_exaggeration,
                "seed": seed,
            }
            ball_coordinates = _ball(raw)
            surface_coordinates = _surface(raw)
            ball_trial = _evaluate(
                ball_coordinates,
                method="tsne",
                mode="ball",
                parameters=parameters,
                recommendation_32=recommendation_32,
                recommendation_512=recommendation_512,
                vectors_32=vectors_32,
                vectors_512=vectors_512,
                visual_k=visual_k,
            )
            surface_trial = _evaluate(
                surface_coordinates,
                method="tsne_radial_projection",
                mode="surface",
                parameters=parameters,
                recommendation_32=recommendation_32,
                recommendation_512=recommendation_512,
                vectors_32=vectors_32,
                vectors_512=vectors_512,
                visual_k=visual_k,
            )
            ball_trials.append((ball_coordinates, ball_trial))
            surface_trials.append((surface_coordinates, surface_trial))
            print(
                "t-SNE",
                perplexity,
                early_exaggeration,
                f"ball_hit={ball_trial.against_32d['any_hit_rate']:.4f}",
                f"surface_hit={surface_trial.against_32d['any_hit_rate']:.4f}",
                flush=True,
            )

    for n_neighbors in UMAP_NEIGHBORS:
        for min_dist in UMAP_MIN_DISTS:
            raw = _fit_umap(
                vectors_32,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                seed=seed,
            )
            coordinates = _ball(raw)
            trial = _evaluate(
                coordinates,
                method="umap",
                mode="ball",
                parameters={
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "seed": seed,
                },
                recommendation_32=recommendation_32,
                recommendation_512=recommendation_512,
                vectors_32=vectors_32,
                vectors_512=vectors_512,
                visual_k=visual_k,
            )
            ball_trials.append((coordinates, trial))
            print(
                "UMAP",
                n_neighbors,
                min_dist,
                f"ball_hit={trial.against_32d['any_hit_rate']:.4f}",
                flush=True,
            )

    ball_trials.sort(key=lambda item: _rank(item[1]), reverse=True)
    surface_trials.sort(key=lambda item: _rank(item[1]), reverse=True)
    return ball_trials, surface_trials


def _save_layout(
    path: Path,
    *,
    source: VectorArtifact,
    coordinates: np.ndarray,
    trial: Trial,
    recommendation_k: int,
    visual_k: int,
) -> None:
    metadata = {
        "experiment": "fixed_32d_top3_recommendation_set_hit",
        "recommendation_k": recommendation_k,
        "visual_k": visual_k,
        "trial": asdict(trial),
    }
    save_vector_artifact(
        path,
        cocktail_ids=source.cocktail_ids,
        cocktail_names=source.cocktail_names,
        vectors=coordinates,
        metadata=metadata,
    )
    artifact = load_vector_artifact(path)
    write_vector_csv(path.with_suffix(".csv"), artifact, vector_column="embedding_3d")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    embedding_512 = load_vector_artifact(arguments.embedding_512)
    embedding_32 = load_vector_artifact(arguments.embedding_32)
    _require_matching_rows(embedding_512, embedding_32)
    row_count = len(embedding_32)
    if not 1 <= arguments.visual_k < arguments.recommendation_k < row_count:
        raise ValueError("expected 1 <= visual-k < recommendation-k < cocktail count")

    ball_trials, surface_trials = _run_trials(
        embedding_32,
        embedding_512,
        recommendation_k=arguments.recommendation_k,
        visual_k=arguments.visual_k,
        seed=arguments.seed,
    )
    best_ball_coordinates, best_ball = ball_trials[0]
    best_surface_coordinates, best_surface = surface_trials[0]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _save_layout(
        arguments.output_dir / "best-ball.npz",
        source=embedding_32,
        coordinates=best_ball_coordinates,
        trial=best_ball,
        recommendation_k=arguments.recommendation_k,
        visual_k=arguments.visual_k,
    )
    _save_layout(
        arguments.output_dir / "best-surface.npz",
        source=embedding_32,
        coordinates=best_surface_coordinates,
        trial=best_surface,
        recommendation_k=arguments.recommendation_k,
        visual_k=arguments.visual_k,
    )

    report: dict[str, Any] = {
        "experiment": "fixed_32d_top3_recommendation_set_hit",
        "database_writes": False,
        "catalog_size": row_count,
        "source_artifacts": {
            str(arguments.embedding_512): file_sha256(arguments.embedding_512),
            str(arguments.embedding_32): file_sha256(arguments.embedding_32),
        },
        "recommendation_k": arguments.recommendation_k,
        "visual_k": arguments.visual_k,
        "learned_32d_recall_against_512d": neighbor_recall(
            embedding_512.vectors,
            embedding_32.vectors,
            arguments.recommendation_k,
        ),
        "selection_rule": (
            "maximize 32D recommendation-set any-hit, then mean matches, "
            "all visual neighbors matched, strict recall"
        ),
        "best_ball": asdict(best_ball),
        "best_surface": asdict(best_surface),
        "ball_trials": [asdict(trial) for _, trial in ball_trials],
        "surface_trials": [asdict(trial) for _, trial in surface_trials],
    }

    if arguments.cluster_surface.exists():
        cluster_surface = load_vector_artifact(arguments.cluster_surface)
        _require_matching_rows(embedding_32, cluster_surface)
        recommendation_32 = knn_indices(
            embedding_32.vectors,
            arguments.recommendation_k,
            metric="cosine",
        )
        recommendation_512 = knn_indices(
            embedding_512.vectors,
            arguments.recommendation_k,
            metric="cosine",
        )
        report["cluster_surface_baseline"] = asdict(
            _evaluate(
                cluster_surface.vectors,
                method="cluster_surface",
                mode="surface",
                parameters={},
                recommendation_32=recommendation_32,
                recommendation_512=recommendation_512,
                vectors_32=embedding_32.vectors,
                vectors_512=embedding_512.vectors,
                visual_k=arguments.visual_k,
            )
        )

    metrics_path = arguments.output_dir / "metrics.json"
    write_json(metrics_path, report)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "best_ball": asdict(best_ball),
                "best_surface": asdict(best_surface),
                "cluster_surface_baseline": report.get("cluster_surface_baseline"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
