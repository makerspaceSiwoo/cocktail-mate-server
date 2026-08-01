"""Build local 512D taste embeddings, learned 32D vectors, and 3D experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.embedding_pipeline.core import (
    VectorArtifact,
    file_sha256,
    l2_normalize,
    load_vector_artifact,
    knn_indices,
    neighbor_metrics,
    read_taste_dataset,
    save_vector_artifact,
    write_json,
    write_vector_csv,
)
from app.embedding_pipeline.local_embedding import (
    MODEL_DIMENSION,
    MODEL_ID,
    MODEL_MAX_SEQUENCE_LENGTH,
    MODEL_REVISION,
    download_model,
    encode_taste_dataset,
)
from app.embedding_pipeline.projection import (
    ProjectionConfig,
    save_projection_model,
    train_projection,
)
from app.embedding_pipeline.reduction_3d import (
    BallConfig,
    ClusterSurfaceConfig,
    SurfaceConfig,
    reduce_to_cluster_surface,
    run_3d_experiment,
)

DEFAULT_TASTE_CSV = Path("taste-data/cocktail-taste-descriptions.csv")
DEFAULT_MODEL_DIR = Path("embedding-models/distiluse-base-multilingual-cased-v2")
DEFAULT_ARTIFACT_DIR = Path("embedding-artifacts")

logger = logging.getLogger("scripts.build_cocktail_embeddings")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only cocktail embedding pipeline. It never imports or calls "
            "Gemini; network access is used only by download-model for Hugging Face."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="Validate data and ML setup.")
    preflight.add_argument("--input", type=Path, default=DEFAULT_TASTE_CSV)
    preflight.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    download = commands.add_parser(
        "download-model", help="Download the pinned 512D model to local storage."
    )
    download.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    embed = commands.add_parser("embed", help="Encode embedding_text into 512D.")
    _add_embedding_arguments(embed)

    train = commands.add_parser(
        "train-32", help="Train and save the neighbor-distilled 512D-to-32D model."
    )
    _add_projection_arguments(train)

    experiment = commands.add_parser(
        "experiment-3d", help="Compare free-ball UMAP with direct spherical reduction."
    )
    _add_3d_arguments(experiment)

    cluster_surface = commands.add_parser(
        "experiment-cluster-surface",
        help=(
            "Build separated spherical caps while preserving only strong local "
            "neighbors."
        ),
    )
    _add_cluster_surface_arguments(cluster_surface)

    run_all = commands.add_parser(
        "run-all", help="Download, embed, train 32D, and compare both 3D modes."
    )
    run_all.add_argument("--input", type=Path, default=DEFAULT_TASTE_CSV)
    run_all.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    run_all.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    run_all.add_argument("--batch-size", type=int, default=32)
    run_all.add_argument("--device", default="cpu")
    run_all.add_argument("--projection-epochs", type=int, default=2000)
    run_all.add_argument("--projection-restarts", type=int, default=4)
    run_all.add_argument("--surface-epochs", type=int, default=2000)
    run_all.add_argument("--surface-restarts", type=int, default=4)
    run_all.add_argument("--seed", type=int, default=20260801)

    apply_db = commands.add_parser(
        "apply-db", help="Transactionally write 32D and selected 3D vectors to DB."
    )
    apply_db.add_argument(
        "--embedding-32",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "embeddings-32.npz",
    )
    apply_db.add_argument(
        "--embedding-3d",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "embedding-3d-selected.npz",
    )
    apply_db.add_argument(
        "--commit",
        action="store_true",
        help="Commit the transaction. Without this flag validation is read-only.",
    )
    return parser


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_TASTE_CSV)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "embeddings-512.npz",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")


def _add_projection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "embeddings-512.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--teacher-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cpu")


def _add_3d_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--surface-teacher-k", type=int, default=10)
    parser.add_argument("--surface-epochs", type=int, default=2000)
    parser.add_argument("--surface-patience", type=int, default=250)
    parser.add_argument("--surface-restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cpu")


def _add_cluster_surface_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--min-clusters", type=int, default=7)
    parser.add_argument("--max-clusters", type=int, default=7)
    parser.add_argument("--silhouette-tolerance", type=float, default=0.01)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--local-neighbors", type=int, default=3)
    parser.add_argument("--local-min-dist", type=float, default=0.15)
    parser.add_argument("--local-radius-quantile", type=float, default=0.95)
    parser.add_argument("--max-cap-radius", type=float, default=0.95)
    parser.add_argument("--cap-radius-fraction", type=float, default=0.95)
    parser.add_argument("--center-epochs", type=int, default=3000)
    parser.add_argument("--center-restarts", type=int, default=4)
    parser.add_argument("--center-minimum-angle-degrees", type=float, default=40.0)
    parser.add_argument("--center-balance-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260801)


def _preflight(input_path: Path, model_dir: Path) -> dict[str, Any]:
    dataset = read_taste_dataset(input_path)
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "numpy",
            "scipy",
            "sklearn",
            "torch",
            "sentence_transformers",
            "huggingface_hub",
            "safetensors",
            "umap",
        )
    }
    return {
        "rows": len(dataset),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "model_dir": str(model_dir),
        "model_downloaded": (model_dir / "modules.json").is_file(),
        "dependencies": dependencies,
        "ready": all(dependencies.values()) and (model_dir / "modules.json").is_file(),
        "gemini_used": False,
    }


def _embed(
    input_path: Path,
    model_dir: Path,
    output_path: Path,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    dataset = read_taste_dataset(input_path)
    vectors = encode_taste_dataset(
        dataset,
        model_dir,
        batch_size=batch_size,
        device=device,
    )
    metadata = {
        "stage": "local_text_embedding",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dimension": MODEL_DIMENSION,
        "model_max_sequence_length": MODEL_MAX_SEQUENCE_LENGTH,
        "pooling": "L2-normalized mean of per-sentence embeddings",
        "normalized": True,
        "source_csv": str(input_path),
        "source_sha256": file_sha256(input_path),
        "rows": len(dataset),
        "gemini_used": False,
    }
    save_vector_artifact(
        output_path,
        cocktail_ids=dataset.cocktail_ids,
        cocktail_names=dataset.cocktail_names,
        vectors=vectors,
        metadata=metadata,
    )
    return metadata | {"output": str(output_path)}


def _train_32(
    input_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    patience: int,
    restarts: int,
    teacher_k: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    source = load_vector_artifact(input_path)
    if source.vectors.shape[1] != MODEL_DIMENSION:
        raise ValueError(f"expected a 512D artifact, got {source.vectors.shape[1]}D")
    config = ProjectionConfig(
        teacher_k=teacher_k,
        max_epochs=epochs,
        patience=patience,
        restarts=restarts,
        seed=seed,
        device=device,
    )
    result = train_projection(source.vectors, config)

    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required; install requirements-embedding.txt"
        ) from error
    pca = PCA(n_components=config.output_dim, random_state=seed)
    pca_vectors = l2_normalize(pca.fit_transform(source.vectors))
    report = result.report | {
        "pca_32_baseline": {
            "metrics": neighbor_metrics(source.vectors, pca_vectors),
            "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
        }
    }
    artifact_path = output_dir / "embeddings-32.npz"
    artifact_metadata = {
        "stage": "learned_neighbor_projection",
        "dimension": config.output_dim,
        "normalized": True,
        "source_artifact": str(input_path),
        "source_sha256": file_sha256(input_path),
        "training": report,
    }
    save_vector_artifact(
        artifact_path,
        cocktail_ids=source.cocktail_ids,
        cocktail_names=source.cocktail_names,
        vectors=result.vectors,
        metadata=artifact_metadata,
    )
    artifact = load_vector_artifact(artifact_path)
    write_vector_csv(
        output_dir / "embeddings-32.csv", artifact, vector_column="embedding"
    )
    save_projection_model(
        output_dir / "projection-512-to-32.safetensors",
        output_dir / "projection-512-to-32.json",
        vectors=result.vectors,
        cocktail_ids=source.cocktail_ids,
        config=config,
        report=report,
        source_metadata=source.metadata,
    )
    write_json(output_dir / "projection-metrics.json", report)
    return report | {
        "output": str(artifact_path),
        "weights": str(output_dir / "projection-512-to-32.safetensors"),
    }


def _experiment_3d(
    embedding_512_path: Path,
    embedding_32_path: Path,
    output_dir: Path,
    *,
    n_neighbors: int,
    min_dist: float,
    surface_teacher_k: int,
    surface_epochs: int,
    surface_patience: int,
    surface_restarts: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    reference = load_vector_artifact(embedding_512_path)
    recommendation = load_vector_artifact(embedding_32_path)
    _require_matching_rows(reference, recommendation)
    experiment = run_3d_experiment(
        recommendation.vectors,
        reference.vectors,
        ball_config=BallConfig(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=seed,
        ),
        surface_config=SurfaceConfig(
            teacher_k=surface_teacher_k,
            max_epochs=surface_epochs,
            patience=surface_patience,
            restarts=surface_restarts,
            seed=seed,
            device=device,
        ),
    )

    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required; install requirements-embedding.txt"
        ) from error
    pca = PCA(n_components=3, random_state=seed).fit(reference.vectors)
    report = experiment.report | {
        "pca_512_top3_explained_variance_ratio": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "spot_checks": _spot_checks(
            reference,
            recommendation,
            experiment.ball_coordinates,
            experiment.surface_coordinates,
            names=("플레이밍 닥터페퍼",),
        ),
    }
    shared_metadata = {
        "stage": "neighbor_preserving_3d_experiment",
        "source_32_sha256": file_sha256(embedding_32_path),
        "source_512_sha256": file_sha256(embedding_512_path),
    }
    ball = _save_coordinate_artifact(
        output_dir / "embedding-3d-ball.npz",
        recommendation,
        experiment.ball_coordinates,
        metadata=shared_metadata | {"mode": "ball", "metrics": report["ball"]},
    )
    surface = _save_coordinate_artifact(
        output_dir / "embedding-3d-surface.npz",
        recommendation,
        experiment.surface_coordinates,
        metadata=shared_metadata | {"mode": "surface", "metrics": report["surface"]},
    )
    write_vector_csv(
        output_dir / "embedding-3d-ball.csv", ball, vector_column="embedding_3d"
    )
    write_vector_csv(
        output_dir / "embedding-3d-surface.csv",
        surface,
        vector_column="embedding_3d",
    )
    selected_coordinates = (
        experiment.ball_coordinates
        if experiment.selected_mode == "ball"
        else experiment.surface_coordinates
    )
    selected_artifact = _save_coordinate_artifact(
        output_dir / "embedding-3d-selected.npz",
        recommendation,
        selected_coordinates,
        metadata=shared_metadata
        | {"mode": experiment.selected_mode, "selection": report},
    )
    write_vector_csv(
        output_dir / "embedding-3d-selected.csv",
        selected_artifact,
        vector_column="embedding_3d",
    )
    write_json(output_dir / "embedding-3d-metrics.json", report)
    return report | {
        "metrics_output": str(output_dir / "embedding-3d-metrics.json"),
        "selected_output": str(output_dir / "embedding-3d-selected.npz"),
    }


def _experiment_cluster_surface(
    embedding_512_path: Path,
    embedding_32_path: Path,
    output_dir: Path,
    *,
    min_clusters: int,
    max_clusters: int,
    silhouette_tolerance: float,
    min_cluster_size: int,
    local_neighbors: int,
    local_min_dist: float,
    local_radius_quantile: float,
    max_cap_radius: float,
    cap_radius_fraction: float,
    center_epochs: int,
    center_restarts: int,
    center_minimum_angle_degrees: float,
    center_balance_weight: float,
    seed: int,
) -> dict[str, Any]:
    reference = load_vector_artifact(embedding_512_path)
    recommendation = load_vector_artifact(embedding_32_path)
    _require_matching_rows(reference, recommendation)
    experiment = reduce_to_cluster_surface(
        recommendation.vectors,
        ClusterSurfaceConfig(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            silhouette_tolerance=silhouette_tolerance,
            min_cluster_size=min_cluster_size,
            local_neighbors=local_neighbors,
            local_min_dist=local_min_dist,
            local_radius_quantile=local_radius_quantile,
            max_cap_radius=max_cap_radius,
            cap_radius_fraction=cap_radius_fraction,
            center_epochs=center_epochs,
            center_restarts=center_restarts,
            center_minimum_angle_degrees=center_minimum_angle_degrees,
            center_balance_weight=center_balance_weight,
            seed=seed,
        ),
    )
    report = experiment.report | {
        "reference_512d_visual_neighbor_recall": neighbor_metrics(
            reference.vectors,
            experiment.coordinates,
            candidate_metric="euclidean",
            ks=(1, 3, 5, 10, 15),
        ),
        "cluster_summaries": _cluster_summaries(
            recommendation,
            experiment.cluster_labels,
        ),
    }
    shared_metadata = {
        "stage": "cluster_first_spherical_surface_experiment",
        "mode": "cluster_surface",
        "source_32_sha256": file_sha256(embedding_32_path),
        "source_512_sha256": file_sha256(embedding_512_path),
        "metrics": report,
    }
    artifact_path = output_dir / "embedding-3d-cluster-surface.npz"
    artifact = _save_coordinate_artifact(
        artifact_path,
        recommendation,
        experiment.coordinates,
        metadata=shared_metadata,
    )
    write_vector_csv(
        output_dir / "embedding-3d-cluster-surface.csv",
        artifact,
        vector_column="embedding_3d",
    )
    metrics_path = output_dir / "embedding-3d-cluster-surface-metrics.json"
    assignments_path = output_dir / "embedding-3d-cluster-assignments.json"
    write_json(metrics_path, report)
    write_json(
        assignments_path,
        {
            "assignments": [
                {
                    "cocktail_id": int(cocktail_id),
                    "cocktail_name_ko": name,
                    "cluster_id": int(cluster_id),
                }
                for cocktail_id, name, cluster_id in zip(
                    recommendation.cocktail_ids,
                    recommendation.cocktail_names,
                    experiment.cluster_labels,
                    strict=True,
                )
            ]
        },
    )
    return report | {
        "coordinate_output": str(artifact_path),
        "metrics_output": str(metrics_path),
        "assignments_output": str(assignments_path),
        "database_updated": False,
    }


def _cluster_summaries(
    source: VectorArtifact,
    labels: np.ndarray,
    *,
    representative_count: int = 8,
) -> list[dict[str, Any]]:
    normalized = l2_normalize(source.vectors)
    summaries: list[dict[str, Any]] = []
    for cluster_id in range(int(labels.max()) + 1):
        member_indices = np.flatnonzero(labels == cluster_id)
        centroid = l2_normalize(normalized[member_indices].mean(axis=0, keepdims=True))[
            0
        ]
        ranked = member_indices[
            np.argsort(-(normalized[member_indices] @ centroid), kind="stable")
        ][:representative_count]
        summaries.append(
            {
                "cluster_id": cluster_id,
                "size": len(member_indices),
                "representatives": [
                    {
                        "cocktail_id": int(source.cocktail_ids[index]),
                        "name": source.cocktail_names[index],
                    }
                    for index in ranked
                ],
            }
        )
    return summaries


def _save_coordinate_artifact(
    path: Path,
    source: VectorArtifact,
    coordinates: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> VectorArtifact:
    save_vector_artifact(
        path,
        cocktail_ids=source.cocktail_ids,
        cocktail_names=source.cocktail_names,
        vectors=coordinates,
        metadata=metadata,
    )
    return load_vector_artifact(path)


def _spot_checks(
    reference: VectorArtifact,
    recommendation: VectorArtifact,
    ball_coordinates: np.ndarray,
    surface_coordinates: np.ndarray,
    *,
    names: tuple[str, ...],
    k: int = 5,
) -> dict[str, Any]:
    name_to_index = {name: index for index, name in enumerate(reference.cocktail_names)}
    neighbor_sets = {
        "reference_512d": knn_indices(reference.vectors, k, metric="cosine"),
        "recommendation_32d": knn_indices(recommendation.vectors, k, metric="cosine"),
        "ball": knn_indices(ball_coordinates, k, metric="euclidean"),
        "surface": knn_indices(surface_coordinates, k, metric="euclidean"),
    }
    checks: dict[str, Any] = {}
    for name in names:
        if name not in name_to_index:
            checks[name] = {"error": "cocktail not found"}
            continue
        index = name_to_index[name]
        reference_neighbors = set(neighbor_sets["reference_512d"][index])
        check: dict[str, Any] = {
            "cocktail_id": int(reference.cocktail_ids[index]),
        }
        for mode, neighbors in neighbor_sets.items():
            indices = neighbors[index]
            check[mode] = {
                "neighbors": [
                    {
                        "cocktail_id": int(reference.cocktail_ids[neighbor]),
                        "name": reference.cocktail_names[neighbor],
                    }
                    for neighbor in indices
                ],
                "overlap_with_512d": len(reference_neighbors.intersection(indices)) / k,
            }
        checks[name] = check
    return checks


def _require_matching_rows(left: VectorArtifact, right: VectorArtifact) -> None:
    if not np.array_equal(left.cocktail_ids, right.cocktail_ids):
        raise ValueError("artifact cocktail IDs or order do not match")
    if left.cocktail_names != right.cocktail_names:
        raise ValueError("artifact cocktail names or order do not match")


def _apply_db(
    embedding_32_path: Path,
    embedding_3d_path: Path,
    *,
    commit: bool,
) -> dict[str, Any]:
    embedding_32 = load_vector_artifact(embedding_32_path)
    embedding_3d = load_vector_artifact(embedding_3d_path)
    _require_matching_rows(embedding_32, embedding_3d)
    if embedding_32.vectors.shape[1] != 32:
        raise ValueError("recommendation artifact must contain 32D vectors")
    if embedding_3d.vectors.shape[1] != 3:
        raise ValueError("visual artifact must contain 3D coordinates")

    from cocktail_mate_db.models import Cocktail  # noqa: PLC0415
    from sqlalchemy import select  # noqa: PLC0415

    from app.core.database import SessionLocal  # noqa: PLC0415

    ids = [int(value) for value in embedding_32.cocktail_ids]
    with SessionLocal() as session:
        existing = set(
            session.scalars(select(Cocktail.id).where(Cocktail.id.in_(ids))).all()
        )
        missing = sorted(set(ids) - existing)
        if missing:
            raise ValueError(f"DB is missing artifact cocktail IDs: {missing}")
        if commit:
            updated_at = datetime.now(UTC)
            mappings = [
                {
                    "id": cocktail_id,
                    "embedding": vector_32.tolist(),
                    "embedding_3d": vector_3d.tolist(),
                    "embedding_updated_at": updated_at,
                }
                for cocktail_id, vector_32, vector_3d in zip(
                    ids,
                    embedding_32.vectors,
                    embedding_3d.vectors,
                    strict=True,
                )
            ]
            session.bulk_update_mappings(Cocktail, mappings)
            session.commit()
        else:
            session.rollback()
    return {
        "validated": len(ids),
        "updated": len(ids) if commit else 0,
        "committed": commit,
        "embedding_32": str(embedding_32_path),
        "embedding_3d": str(embedding_3d_path),
    }


def _run_all(arguments: argparse.Namespace) -> dict[str, Any]:
    download_model(arguments.model_dir)
    embedding_512_path = arguments.output_dir / "embeddings-512.npz"
    embed_report = _embed(
        arguments.input,
        arguments.model_dir,
        embedding_512_path,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    projection_report = _train_32(
        embedding_512_path,
        arguments.output_dir,
        epochs=arguments.projection_epochs,
        patience=300,
        restarts=arguments.projection_restarts,
        teacher_k=15,
        seed=arguments.seed,
        device=arguments.device,
    )
    experiment_report = _experiment_3d(
        embedding_512_path,
        arguments.output_dir / "embeddings-32.npz",
        arguments.output_dir,
        n_neighbors=5,
        min_dist=0.15,
        surface_teacher_k=10,
        surface_epochs=arguments.surface_epochs,
        surface_patience=250,
        surface_restarts=arguments.surface_restarts,
        seed=arguments.seed,
        device=arguments.device,
    )
    return {
        "embedding": embed_report,
        "projection_32": projection_report,
        "experiment_3d": experiment_report,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "preflight":
            result = _preflight(arguments.input, arguments.model_dir)
        elif arguments.command == "download-model":
            path = download_model(arguments.model_dir)
            result = {
                "model_dir": str(path),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        elif arguments.command == "embed":
            result = _embed(
                arguments.input,
                arguments.model_dir,
                arguments.output,
                batch_size=arguments.batch_size,
                device=arguments.device,
            )
        elif arguments.command == "train-32":
            result = _train_32(
                arguments.input,
                arguments.output_dir,
                epochs=arguments.epochs,
                patience=arguments.patience,
                restarts=arguments.restarts,
                teacher_k=arguments.teacher_k,
                seed=arguments.seed,
                device=arguments.device,
            )
        elif arguments.command == "experiment-3d":
            result = _experiment_3d(
                arguments.embedding_512,
                arguments.embedding_32,
                arguments.output_dir,
                n_neighbors=arguments.n_neighbors,
                min_dist=arguments.min_dist,
                surface_teacher_k=arguments.surface_teacher_k,
                surface_epochs=arguments.surface_epochs,
                surface_patience=arguments.surface_patience,
                surface_restarts=arguments.surface_restarts,
                seed=arguments.seed,
                device=arguments.device,
            )
        elif arguments.command == "experiment-cluster-surface":
            result = _experiment_cluster_surface(
                arguments.embedding_512,
                arguments.embedding_32,
                arguments.output_dir,
                min_clusters=arguments.min_clusters,
                max_clusters=arguments.max_clusters,
                silhouette_tolerance=arguments.silhouette_tolerance,
                min_cluster_size=arguments.min_cluster_size,
                local_neighbors=arguments.local_neighbors,
                local_min_dist=arguments.local_min_dist,
                local_radius_quantile=arguments.local_radius_quantile,
                max_cap_radius=arguments.max_cap_radius,
                cap_radius_fraction=arguments.cap_radius_fraction,
                center_epochs=arguments.center_epochs,
                center_restarts=arguments.center_restarts,
                center_minimum_angle_degrees=(arguments.center_minimum_angle_degrees),
                center_balance_weight=arguments.center_balance_weight,
                seed=arguments.seed,
            )
        elif arguments.command == "apply-db":
            result = _apply_db(
                arguments.embedding_32,
                arguments.embedding_3d,
                commit=arguments.commit,
            )
        else:
            result = _run_all(arguments)
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
