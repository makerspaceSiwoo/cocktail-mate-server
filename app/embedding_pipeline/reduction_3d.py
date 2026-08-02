"""Neighbor-preserving free-ball and spherical 3D reductions."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.embedding_pipeline.core import (
    knn_indices,
    l2_normalize,
    neighbor_metrics,
    normalized_stress,
)


@dataclass(frozen=True, slots=True)
class BallConfig:
    n_neighbors: int = 5
    min_dist: float = 0.15
    seed: int = 20260801


@dataclass(frozen=True, slots=True)
class SurfaceConfig:
    teacher_k: int = 10
    teacher_temperature: float = 0.01
    student_temperature: float = 0.05
    learning_rate: float = 0.03
    max_epochs: int = 2000
    patience: int = 250
    restarts: int = 4
    seed: int = 20260801
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ClusterSurfaceConfig:
    """Settings for a cluster-first layout on the unit sphere surface."""

    min_clusters: int = 7
    max_clusters: int = 7
    silhouette_tolerance: float = 0.01
    min_cluster_size: int = 8
    local_neighbors: int = 3
    local_min_dist: float = 0.15
    local_radius_quantile: float = 0.95
    max_cap_radius: float = 0.95
    cap_radius_fraction: float = 0.95
    relation_neighbor_k: int = 5
    center_teacher_k: int = 3
    centroid_relation_weight: float = 0.25
    cross_neighbor_relation_weight: float = 0.75
    center_teacher_temperature: float = 0.6
    center_student_temperature: float = 0.12
    center_learning_rate: float = 0.03
    center_epochs: int = 3000
    center_restarts: int = 4
    center_minimum_angle_degrees: float = 40.0
    center_separation_weight: float = 10.0
    center_balance_weight: float = 0.5
    seed: int = 20260801


@dataclass(frozen=True, slots=True)
class ThreeDimensionalExperiment:
    ball_coordinates: np.ndarray
    surface_coordinates: np.ndarray
    report: dict[str, Any]
    selected_mode: str


@dataclass(frozen=True, slots=True)
class ClusterSurfaceExperiment:
    coordinates: np.ndarray
    cluster_labels: np.ndarray
    report: dict[str, Any]


def reduce_to_ball(
    source_vectors: np.ndarray,
    config: BallConfig = BallConfig(),
) -> np.ndarray:
    """Use cosine UMAP and only a global affine fit into the unit ball."""
    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "umap-learn is required; install requirements-embedding.txt"
        ) from error
    source = l2_normalize(source_vectors)
    if not 2 <= config.n_neighbors < len(source):
        raise ValueError("n_neighbors must be between 2 and n-1")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric="cosine",
        n_jobs=1,
        random_state=config.seed,
        transform_seed=config.seed,
    )
    raw = np.asarray(reducer.fit_transform(source), dtype=np.float32)
    centered = raw - raw.mean(axis=0, keepdims=True)
    max_radius = float(np.linalg.norm(centered, axis=1).max())
    if max_radius <= 1e-12:
        raise ValueError("UMAP collapsed all coordinates")
    # One translation and one scale preserve every Euclidean neighbor exactly.
    return np.asarray(centered / max_radius, dtype=np.float32)


def reduce_to_surface(
    source_vectors: np.ndarray,
    config: SurfaceConfig = SurfaceConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize points on S² directly; no Euclidean reduction is normalized later."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install requirements-embedding.txt"
        ) from error

    source = l2_normalize(source_vectors)
    if not 1 <= config.teacher_k < len(source):
        raise ValueError("teacher_k must be between 1 and n-1")
    if config.restarts <= 0 or config.max_epochs <= 0:
        raise ValueError("restarts and max_epochs must be positive")

    device = torch.device(config.device)
    source_tensor = torch.as_tensor(source, dtype=torch.float32, device=device)
    teacher_similarity = source_tensor @ source_tensor.T
    teacher_similarity.fill_diagonal_(-torch.inf)
    values, indices = torch.topk(teacher_similarity, k=config.teacher_k, dim=1)
    weights = torch.softmax(values / config.teacher_temperature, dim=1)
    teacher = torch.zeros_like(teacher_similarity)
    teacher.scatter_(1, indices, weights)
    diagonal = torch.eye(len(source), dtype=torch.bool, device=device)

    best_coordinates: np.ndarray | None = None
    best_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    restart_reports: list[dict[str, Any]] = []
    for restart in range(config.restarts):
        restart_seed = config.seed + restart
        random.seed(restart_seed)
        np.random.seed(restart_seed)
        torch.manual_seed(restart_seed)
        parameters = torch.nn.Parameter(
            torch.randn((len(source), 3), dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam([parameters], lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.05
        )
        best_restart_coordinates: np.ndarray | None = None
        best_restart_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
        best_restart_epoch = 0
        stale_epochs = 0
        final_loss = float("nan")

        for epoch in range(1, config.max_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            coordinates = torch.nn.functional.normalize(parameters, dim=1)
            logits = coordinates @ coordinates.T / config.student_temperature
            logits = logits.masked_fill(diagonal, -torch.inf)
            log_probabilities = torch.log_softmax(logits, dim=1)
            log_probabilities = log_probabilities.masked_fill(diagonal, 0.0)
            loss = -(teacher * log_probabilities).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.detach().cpu())

            if epoch != 1 and epoch % 25 != 0:
                continue
            with torch.no_grad():
                candidate = (
                    torch.nn.functional.normalize(parameters, dim=1).cpu().numpy()
                )
            metrics = neighbor_metrics(
                source,
                candidate,
                candidate_metric="euclidean",
            )
            score_key = (
                metrics["recall@5"],
                metrics["recall@10"],
                metrics["recall@15"],
            )
            if score_key > best_restart_key:
                best_restart_key = score_key
                best_restart_epoch = epoch
                best_restart_coordinates = candidate.copy()
                stale_epochs = 0
            else:
                stale_epochs += 25
                if stale_epochs >= config.patience:
                    break

        assert best_restart_coordinates is not None
        metrics = neighbor_metrics(
            source,
            best_restart_coordinates,
            candidate_metric="euclidean",
        )
        restart_reports.append(
            {
                "restart": restart,
                "seed": restart_seed,
                "best_epoch": best_restart_epoch,
                "metrics": metrics,
                "final_loss": final_loss,
            }
        )
        if best_restart_key > best_key:
            best_key = best_restart_key
            best_coordinates = best_restart_coordinates

    assert best_coordinates is not None
    return (
        l2_normalize(best_coordinates),
        {
            "algorithm": "direct_spherical_neighbor_distribution_embedding",
            "config": asdict(config),
            "restarts": restart_reports,
        },
    )


def reduce_to_cluster_surface(
    source_vectors: np.ndarray,
    config: ClusterSurfaceConfig = ClusterSurfaceConfig(),
) -> ClusterSurfaceExperiment:
    """Place local neighborhoods around relation-aware cluster centers on S².

    Cluster centers learn a spherical layout from both 32D centroid similarity and
    cross-cluster top-k recommendation edges. Each cluster keeps an independent
    local cosine-UMAP layout, mapped into an adaptive spherical cap around its
    learned center. Nearby cluster caps may overlap instead of imposing an
    artificial equal-angle partition.
    """
    try:
        import umap
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn and umap-learn are required; "
            "install requirements-embedding.txt"
        ) from error

    source = l2_normalize(source_vectors)
    _validate_cluster_surface_config(config, len(source))
    source_neighbors = knn_indices(
        source,
        min(10, len(source) - 1),
        metric="cosine",
    )

    candidates: list[dict[str, Any]] = []
    candidate_labels: dict[int, np.ndarray] = {}
    for cluster_count in range(config.min_clusters, config.max_clusters + 1):
        raw_labels = KMeans(
            n_clusters=cluster_count,
            n_init=30,
            random_state=config.seed,
        ).fit_predict(source)
        labels = _canonicalize_cluster_labels(raw_labels)
        sizes = np.bincount(labels, minlength=cluster_count)
        valid = bool(sizes.min() >= config.min_cluster_size)
        silhouette = float(silhouette_score(source, labels, metric="cosine"))
        candidates.append(
            {
                "cluster_count": cluster_count,
                "silhouette_cosine": silhouette,
                "minimum_cluster_size": int(sizes.min()),
                "maximum_cluster_size": int(sizes.max()),
                "valid": valid,
                "source_neighbor_containment": _neighbor_containment_from_indices(
                    source_neighbors,
                    labels,
                    ks=(1, 3, 5, 10),
                ),
            }
        )
        if valid:
            candidate_labels[cluster_count] = labels

    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if not valid_candidates:
        raise ValueError(
            "no cluster count satisfies min_cluster_size; lower the constraint"
        )
    best_silhouette = max(
        candidate["silhouette_cosine"] for candidate in valid_candidates
    )
    threshold = best_silhouette - (abs(best_silhouette) * config.silhouette_tolerance)
    selected = min(
        (
            candidate
            for candidate in valid_candidates
            if candidate["silhouette_cosine"] >= threshold
        ),
        key=lambda candidate: candidate["cluster_count"],
    )
    cluster_count = int(selected["cluster_count"])
    labels = candidate_labels[cluster_count]

    cap_centers, center_relationships = _learn_cluster_center_layout(
        source,
        labels,
        config,
    )
    center_angles = np.arccos(np.clip(cap_centers @ cap_centers.T, -1.0, 1.0))
    np.fill_diagonal(center_angles, np.inf)
    nearest_center_angles = center_angles.min(axis=1)
    cap_radii = np.minimum(
        config.max_cap_radius,
        nearest_center_angles * config.cap_radius_fraction,
    )
    coordinates = np.empty((len(source), 3), dtype=np.float32)
    for cluster_id in range(cluster_count):
        member_indices = np.flatnonzero(labels == cluster_id)
        local_source = source[member_indices]
        local_neighbor_count = min(config.local_neighbors, len(member_indices) - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=local_neighbor_count,
            min_dist=config.local_min_dist,
            metric="cosine",
            n_jobs=1,
            random_state=config.seed + cluster_id,
            transform_seed=config.seed + cluster_id,
        )
        local_layout = np.asarray(
            reducer.fit_transform(local_source),
            dtype=np.float32,
        )
        coordinates[member_indices] = _map_to_spherical_cap(
            local_layout,
            cap_centers[cluster_id],
            float(cap_radii[cluster_id]),
            radius_quantile=config.local_radius_quantile,
        )

    coordinates = l2_normalize(coordinates)
    report: dict[str, Any] = {
        "algorithm": "cluster_first_local_umap_spherical_caps",
        "config": asdict(config),
        "selection_rule": (
            "choose the smallest valid cluster count retaining at least "
            f"{(1.0 - config.silhouette_tolerance) * 100:.1f}% of the best "
            "cosine silhouette"
        ),
        "cluster_candidates": candidates,
        "selected_cluster_count": cluster_count,
        "selected_silhouette_cosine": float(selected["silhouette_cosine"]),
        "source_neighbor_containment": selected["source_neighbor_containment"],
        "visual_neighbor_recall": neighbor_metrics(
            source,
            coordinates,
            candidate_metric="euclidean",
            ks=(1, 3, 5, 10, 15),
        ),
        "within_cluster_neighbor_recall": _within_cluster_neighbor_recall(
            source,
            coordinates,
            labels,
            ks=(1, 3, 5, 10),
        ),
        "within_cluster_separation_degrees": _within_cluster_separation_degrees(
            coordinates,
            labels,
        ),
        "global_stress_secondary": normalized_stress(source, coordinates),
        "minimum_center_angle_radians": float(nearest_center_angles.min()),
        "cap_radius_radians": {
            "minimum": float(cap_radii.min()),
            "mean": float(cap_radii.mean()),
            "maximum": float(cap_radii.max()),
            "by_cluster": [float(value) for value in cap_radii],
        },
        "center_layout": "learned_cluster_relationship_sphere",
        "cluster_center_coordinates": cap_centers.tolist(),
        "cluster_relationships": center_relationships,
        "radius": {
            "minimum": float(np.linalg.norm(coordinates, axis=1).min()),
            "mean": float(np.linalg.norm(coordinates, axis=1).mean()),
            "maximum": float(np.linalg.norm(coordinates, axis=1).max()),
        },
    }
    return ClusterSurfaceExperiment(coordinates, labels, report)


def _validate_cluster_surface_config(
    config: ClusterSurfaceConfig,
    row_count: int,
) -> None:
    if not 2 <= config.min_clusters <= config.max_clusters < row_count:
        raise ValueError("cluster range must satisfy 2 <= min <= max < row count")
    if not 0.0 <= config.silhouette_tolerance < 1.0:
        raise ValueError("silhouette_tolerance must be in [0, 1)")
    if config.min_cluster_size < 3:
        raise ValueError("min_cluster_size must be at least 3")
    if config.local_neighbors < 2:
        raise ValueError("local_neighbors must be at least 2")
    if config.local_neighbors >= config.min_cluster_size:
        raise ValueError("local_neighbors must be smaller than min_cluster_size")
    if not 0.0 <= config.local_min_dist <= 1.0:
        raise ValueError("local_min_dist must be in [0, 1]")
    if not 0.5 <= config.local_radius_quantile <= 1.0:
        raise ValueError("local_radius_quantile must be in [0.5, 1]")
    if not 0.0 < config.max_cap_radius < np.pi / 2.0:
        raise ValueError("max_cap_radius must be in (0, pi/2)")
    if not 0.0 < config.cap_radius_fraction <= 1.0:
        raise ValueError("cap_radius_fraction must be in (0, 1]")
    if not 1 <= config.relation_neighbor_k < row_count:
        raise ValueError("relation_neighbor_k must be between 1 and row count - 1")
    if not 1 <= config.center_teacher_k < config.min_clusters:
        raise ValueError("center_teacher_k must be smaller than the cluster count")
    relation_weight_sum = (
        config.centroid_relation_weight + config.cross_neighbor_relation_weight
    )
    if config.centroid_relation_weight < 0 or config.cross_neighbor_relation_weight < 0:
        raise ValueError("cluster relation weights cannot be negative")
    if not np.isclose(relation_weight_sum, 1.0):
        raise ValueError("cluster relation weights must sum to 1")
    if config.center_epochs <= 0 or config.center_restarts <= 0:
        raise ValueError("center epochs and restarts must be positive")
    if config.center_learning_rate <= 0:
        raise ValueError("center_learning_rate must be positive")
    if config.center_teacher_temperature <= 0 or config.center_student_temperature <= 0:
        raise ValueError("center temperatures must be positive")
    if not 0.0 < config.center_minimum_angle_degrees < 180.0:
        raise ValueError("center_minimum_angle_degrees must be in (0, 180)")
    if config.center_separation_weight < 0:
        raise ValueError("center_separation_weight cannot be negative")
    if config.center_balance_weight < 0:
        raise ValueError("center_balance_weight cannot be negative")


def _canonicalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    values, counts = np.unique(labels, return_counts=True)
    ordered = sorted(
        zip(values.tolist(), counts.tolist(), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(ordered)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)


def _neighbor_containment_from_indices(
    neighbor_indices: np.ndarray,
    labels: np.ndarray,
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    return {
        f"top_{k}_same_cluster_rate": float(
            np.mean(labels[neighbor_indices[:, :k]] == labels[:, np.newaxis])
        )
        for k in ks
        if k <= neighbor_indices.shape[1]
    }


def _learn_cluster_center_layout(
    source_vectors: np.ndarray,
    labels: np.ndarray,
    config: ClusterSurfaceConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from scipy.stats import spearmanr
    except ImportError as error:
        raise RuntimeError(
            "PyTorch and SciPy are required; install requirements-embedding.txt"
        ) from error

    source = l2_normalize(source_vectors)
    cluster_count = int(labels.max()) + 1
    cluster_sizes = np.bincount(labels, minlength=cluster_count)
    centroids = l2_normalize(
        np.stack(
            [
                source[labels == cluster_id].mean(axis=0)
                for cluster_id in range(cluster_count)
            ]
        )
    )
    centroid_similarity = centroids @ centroids.T

    source_neighbors = knn_indices(
        source,
        min(config.relation_neighbor_k, len(source) - 1),
        metric="cosine",
    )
    directed_flow = np.zeros((cluster_count, cluster_count), dtype=np.int64)
    source_rows = np.repeat(np.arange(len(source)), source_neighbors.shape[1])
    np.add.at(
        directed_flow,
        (labels[source_rows], labels[source_neighbors.ravel()]),
        1,
    )
    symmetric_flow = directed_flow + directed_flow.T
    np.fill_diagonal(symmetric_flow, 0)
    flow_association = symmetric_flow / np.sqrt(
        cluster_sizes[:, np.newaxis] * cluster_sizes[np.newaxis, :]
    )
    np.fill_diagonal(flow_association, 0.0)

    combined_relation = config.centroid_relation_weight * _zscore_off_diagonal(
        centroid_similarity
    ) + config.cross_neighbor_relation_weight * _zscore_off_diagonal(flow_association)
    np.fill_diagonal(combined_relation, 0.0)

    relation_tensor = torch.as_tensor(combined_relation, dtype=torch.float32)
    diagonal = torch.eye(cluster_count, dtype=torch.bool)
    teacher_logits = relation_tensor.masked_fill(diagonal, -torch.inf)
    values, indices = torch.topk(
        teacher_logits,
        k=config.center_teacher_k,
        dim=1,
    )
    teacher_weights = torch.softmax(
        values / config.center_teacher_temperature,
        dim=1,
    )
    teacher = torch.zeros_like(relation_tensor)
    teacher.scatter_(1, indices, teacher_weights)
    upper_indices = torch.triu_indices(cluster_count, cluster_count, offset=1)
    maximum_close_similarity = float(
        np.cos(np.radians(config.center_minimum_angle_degrees))
    )

    best_coordinates: np.ndarray | None = None
    best_key: tuple[float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    )
    restart_reports: list[dict[str, Any]] = []
    for restart in range(config.center_restarts):
        restart_seed = config.seed + restart
        torch.manual_seed(restart_seed)
        parameters = torch.nn.Parameter(
            torch.randn((cluster_count, 3), dtype=torch.float32)
        )
        optimizer = torch.optim.Adam(
            [parameters],
            lr=config.center_learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.center_epochs,
            eta_min=config.center_learning_rate * 0.05,
        )
        final_loss = float("nan")
        for _ in range(config.center_epochs):
            optimizer.zero_grad(set_to_none=True)
            coordinates = torch.nn.functional.normalize(parameters, dim=1)
            coordinate_similarity = coordinates @ coordinates.T
            student_logits = (
                coordinate_similarity / config.center_student_temperature
            ).masked_fill(diagonal, -torch.inf)
            log_probabilities = torch.log_softmax(student_logits, dim=1)
            log_probabilities = log_probabilities.masked_fill(diagonal, 0.0)
            relation_loss = -(teacher * log_probabilities).sum(dim=1).mean()
            pair_similarity = coordinate_similarity[
                upper_indices[0],
                upper_indices[1],
            ]
            separation_loss = (
                torch.relu(pair_similarity - maximum_close_similarity).square().mean()
            )
            balance_loss = coordinates.mean(dim=0).square().sum()
            loss = (
                relation_loss
                + config.center_separation_weight * separation_loss
                + config.center_balance_weight * balance_loss
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.detach())

        candidate = (
            torch.nn.functional.normalize(parameters, dim=1).detach().cpu().numpy()
        )
        metrics = _relation_neighbor_metrics(
            combined_relation,
            candidate,
            ks=(1, 2, 3),
        )
        combined_spearman = _relationship_spearman(
            -combined_relation,
            candidate,
            spearmanr,
        )
        mean_resultant_length = float(np.linalg.norm(candidate.mean(axis=0)))
        candidate_key = (
            metrics["recall@2"],
            metrics["recall@3"],
            combined_spearman,
            -mean_resultant_length,
            metrics["recall@1"],
        )
        restart_reports.append(
            {
                "restart": restart,
                "seed": restart_seed,
                "final_loss": final_loss,
                "center_neighbor_recall": metrics,
                "combined_relation_distance_spearman": combined_spearman,
                "mean_resultant_length": mean_resultant_length,
            }
        )
        if candidate_key > best_key:
            best_key = candidate_key
            best_coordinates = candidate

    assert best_coordinates is not None
    center_angles = np.arccos(np.clip(best_coordinates @ best_coordinates.T, -1.0, 1.0))
    upper_center_angles = center_angles[np.triu_indices(cluster_count, k=1)]
    pair_reports = [
        {
            "left_cluster_id": left,
            "right_cluster_id": right,
            "centroid_cosine_similarity": float(centroid_similarity[left, right]),
            "cross_top_k_edges": int(symmetric_flow[left, right]),
            "normalized_cross_neighbor_association": float(
                flow_association[left, right]
            ),
            "combined_relation_score": float(combined_relation[left, right]),
            "surface_angle_degrees": float(np.degrees(center_angles[left, right])),
        }
        for left in range(cluster_count)
        for right in range(left + 1, cluster_count)
    ]
    pair_reports.sort(
        key=lambda pair: pair["combined_relation_score"],
        reverse=True,
    )
    return l2_normalize(best_coordinates), {
        "relation_definition": (
            "z-scored 32D centroid cosine similarity and symmetric cross-cluster "
            f"top-{source_neighbors.shape[1]} edge association"
        ),
        "relation_weights": {
            "centroid_similarity": config.centroid_relation_weight,
            "cross_neighbor_association": config.cross_neighbor_relation_weight,
        },
        "center_neighbor_recall": _relation_neighbor_metrics(
            combined_relation,
            best_coordinates,
            ks=(1, 2, 3),
        ),
        "centroid_distance_spearman": _relationship_spearman(
            1.0 - centroid_similarity,
            best_coordinates,
            spearmanr,
        ),
        "cross_neighbor_distance_spearman": _relationship_spearman(
            -flow_association,
            best_coordinates,
            spearmanr,
        ),
        "combined_relation_distance_spearman": _relationship_spearman(
            -combined_relation,
            best_coordinates,
            spearmanr,
        ),
        "center_distribution": {
            "mean_resultant_length": float(
                np.linalg.norm(best_coordinates.mean(axis=0))
            ),
            "minimum_angle_degrees": float(np.degrees(upper_center_angles.min())),
            "mean_angle_degrees": float(np.degrees(upper_center_angles.mean())),
            "maximum_angle_degrees": float(np.degrees(upper_center_angles.max())),
        },
        "pairs": pair_reports,
        "restarts": restart_reports,
    }


def _zscore_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    mask = ~np.eye(len(values), dtype=bool)
    standard_deviation = float(values[mask].std())
    if standard_deviation <= 1e-12:
        raise ValueError("cluster relationship matrix has no variation")
    normalized = np.zeros_like(values)
    normalized[mask] = (values[mask] - float(values[mask].mean())) / standard_deviation
    return normalized.astype(np.float32)


def _relation_neighbor_metrics(
    relation: np.ndarray,
    center_coordinates: np.ndarray,
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    source_scores = np.asarray(relation, dtype=np.float64).copy()
    np.fill_diagonal(source_scores, -np.inf)
    center_similarity = (
        l2_normalize(center_coordinates) @ l2_normalize(center_coordinates).T
    )
    np.fill_diagonal(center_similarity, -np.inf)
    metrics: dict[str, float] = {}
    for k in ks:
        expected = np.argsort(-source_scores, axis=1, kind="stable")[:, :k]
        actual = np.argsort(-center_similarity, axis=1, kind="stable")[:, :k]
        metrics[f"recall@{k}"] = float(
            np.mean(
                [
                    len(set(left).intersection(right)) / k
                    for left, right in zip(expected, actual, strict=True)
                ]
            )
        )
    return metrics


def _relationship_spearman(
    source_distance: np.ndarray,
    center_coordinates: np.ndarray,
    spearmanr: Any,
) -> float:
    upper = np.triu_indices(len(center_coordinates), k=1)
    center_similarity = (
        l2_normalize(center_coordinates) @ l2_normalize(center_coordinates).T
    )
    center_angles = np.arccos(np.clip(center_similarity, -1.0, 1.0))
    statistic = spearmanr(
        np.asarray(source_distance)[upper],
        center_angles[upper],
    ).statistic
    return float(statistic)


def _within_cluster_neighbor_recall(
    source_vectors: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    source = l2_normalize(source_vectors)
    source_similarity = source @ source.T
    points = np.asarray(coordinates, dtype=np.float32)
    squared = np.sum(points * points, axis=1, keepdims=True)
    visual_distance = np.maximum(squared + squared.T - 2.0 * points @ points.T, 0.0)
    recalls: dict[str, float] = {}
    for k in ks:
        row_recalls: list[float] = []
        for row in range(len(source)):
            candidates = np.flatnonzero(labels == labels[row])
            candidates = candidates[candidates != row]
            effective_k = min(k, len(candidates))
            if effective_k == 0:
                continue
            expected = candidates[
                np.argsort(-source_similarity[row, candidates], kind="stable")[
                    :effective_k
                ]
            ]
            actual = candidates[
                np.argsort(visual_distance[row, candidates], kind="stable")[
                    :effective_k
                ]
            ]
            row_recalls.append(
                len(set(expected.tolist()).intersection(actual.tolist())) / effective_k
            )
        recalls[f"recall@{k}"] = float(np.mean(row_recalls))
    return recalls


def _within_cluster_separation_degrees(
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    points = l2_normalize(coordinates)
    nearest_angles: list[float] = []
    by_cluster: list[dict[str, float | int]] = []
    for cluster_id in sorted(np.unique(labels).tolist()):
        members = points[labels == cluster_id]
        similarities = np.clip(members @ members.T, -1.0, 1.0)
        angular_distances = np.degrees(np.arccos(similarities))
        np.fill_diagonal(angular_distances, np.inf)
        cluster_nearest = angular_distances.min(axis=1)
        nearest_angles.extend(cluster_nearest.tolist())
        by_cluster.append(
            {
                "cluster_id": int(cluster_id),
                "point_count": int(len(members)),
                "nearest_p10": float(np.percentile(cluster_nearest, 10)),
                "nearest_median": float(np.median(cluster_nearest)),
                "nearest_mean": float(cluster_nearest.mean()),
            }
        )
    values = np.asarray(nearest_angles, dtype=np.float64)
    return {
        "nearest_p10": float(np.percentile(values, 10)),
        "nearest_median": float(np.median(values)),
        "nearest_mean": float(values.mean()),
        "by_cluster": by_cluster,
    }


def _fibonacci_sphere(point_count: int) -> np.ndarray:
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    indices = np.arange(point_count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / point_count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    angle = golden_angle * indices
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z)).astype(
        np.float32
    )


def _map_to_spherical_cap(
    local_layout: np.ndarray,
    center: np.ndarray,
    cap_radius: float,
    *,
    radius_quantile: float = 1.0,
) -> np.ndarray:
    layout = np.asarray(local_layout, dtype=np.float64)
    if layout.ndim != 2 or layout.shape[1] != 2:
        raise ValueError("local_layout must have shape (n, 2)")
    if not 0.0 < cap_radius < np.pi:
        raise ValueError("cap_radius must be in (0, pi)")
    if not 0.5 <= radius_quantile <= 1.0:
        raise ValueError("radius_quantile must be in [0.5, 1]")
    centered = layout - layout.mean(axis=0, keepdims=True)
    planar_radius = np.linalg.norm(centered, axis=1)
    scale_radius = float(np.quantile(planar_radius, radius_quantile))
    if scale_radius <= 1e-12:
        raise ValueError("local layout collapsed to a single point")

    unit_center = np.asarray(center, dtype=np.float64)
    unit_center /= np.linalg.norm(unit_center)
    reference_axis = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(unit_center[2]) > 0.9
        else np.asarray([0.0, 0.0, 1.0])
    )
    tangent_x = np.cross(reference_axis, unit_center)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(unit_center, tangent_x)

    # A single disconnected UMAP outlier must not compress every other point.
    # Quantile scaling remains linear for most members and only clips the outer
    # tail to the cap boundary.
    normalized_radius = np.clip(planar_radius / scale_radius, 0.0, 1.0)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    tangent_direction = (
        np.cos(angles)[:, np.newaxis] * tangent_x
        + np.sin(angles)[:, np.newaxis] * tangent_y
    )
    spherical_radius = normalized_radius * cap_radius
    mapped = (
        np.cos(spherical_radius)[:, np.newaxis] * unit_center
        + np.sin(spherical_radius)[:, np.newaxis] * tangent_direction
    )
    return l2_normalize(mapped)


def run_3d_experiment(
    recommendation_vectors: np.ndarray,
    reference_512_vectors: np.ndarray,
    *,
    ball_config: BallConfig = BallConfig(),
    surface_config: SurfaceConfig = SurfaceConfig(),
) -> ThreeDimensionalExperiment:
    recommendation = l2_normalize(recommendation_vectors)
    reference_512 = l2_normalize(reference_512_vectors)
    if len(recommendation) != len(reference_512):
        raise ValueError("32D and 512D artifacts must have the same row count")

    ball = reduce_to_ball(recommendation, ball_config)
    surface, surface_training = reduce_to_surface(recommendation, surface_config)
    ball_report = _coordinate_metrics(recommendation, reference_512, ball)
    surface_report = _coordinate_metrics(recommendation, reference_512, surface)

    # The production recommender searches the learned 32D space, so recall@5 there
    # is the primary selection criterion. 512D recall and stress remain visible.
    ball_key = (
        ball_report["against_32d"]["recall@5"],
        ball_report["against_32d"]["recall@10"],
        -ball_report["against_32d"]["stress"],
    )
    surface_key = (
        surface_report["against_32d"]["recall@5"],
        surface_report["against_32d"]["recall@10"],
        -surface_report["against_32d"]["stress"],
    )
    selected_mode = "ball" if ball_key >= surface_key else "surface"
    report: dict[str, Any] = {
        "selection_rule": (
            "maximize 32D recommendation-space recall@5; then recall@10; "
            "then minimize stress"
        ),
        "selected_mode": selected_mode,
        "ball": {"config": asdict(ball_config), **ball_report},
        "surface": {"training": surface_training, **surface_report},
    }
    return ThreeDimensionalExperiment(ball, surface, report, selected_mode)


def _coordinate_metrics(
    recommendation_vectors: np.ndarray,
    reference_512_vectors: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, Any]:
    return {
        "against_32d": {
            **neighbor_metrics(
                recommendation_vectors,
                coordinates,
                candidate_metric="euclidean",
            ),
            "stress": normalized_stress(recommendation_vectors, coordinates),
        },
        "against_512d": {
            **neighbor_metrics(
                reference_512_vectors,
                coordinates,
                candidate_metric="euclidean",
            ),
            "stress": normalized_stress(reference_512_vectors, coordinates),
        },
        "radius": {
            "minimum": float(np.linalg.norm(coordinates, axis=1).min()),
            "mean": float(np.linalg.norm(coordinates, axis=1).mean()),
            "maximum": float(np.linalg.norm(coordinates, axis=1).max()),
        },
    }
