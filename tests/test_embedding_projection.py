from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from app.embedding_pipeline.core import neighbor_metrics  # noqa: E402
from app.embedding_pipeline.projection import (  # noqa: E402
    ProjectionConfig,
    train_projection,
)
from app.embedding_pipeline.reduction_3d import (  # noqa: E402
    ClusterSurfaceConfig,
    SurfaceConfig,
    _fibonacci_sphere,
    _map_to_spherical_cap,
    reduce_to_cluster_surface,
    reduce_to_surface,
)


def _clustered_vectors() -> np.ndarray:
    generator = np.random.default_rng(11)
    centers = np.eye(6, dtype=np.float32)
    return np.vstack(
        [center + generator.normal(scale=0.03, size=(5, 6)) for center in centers]
    ).astype(np.float32)


def test_transductive_projection_outputs_unit_32d_vectors() -> None:
    source = _clustered_vectors()

    result = train_projection(
        source,
        ProjectionConfig(
            input_dim=6,
            output_dim=4,
            teacher_k=4,
            max_epochs=150,
            patience=75,
            restarts=1,
            seed=3,
        ),
    )

    assert result.vectors.shape == (30, 4)
    assert np.linalg.norm(result.vectors, axis=1) == pytest.approx(np.ones(30))
    assert neighbor_metrics(source, result.vectors, ks=(4,))["recall@4"] > 0.8


def test_surface_reduction_optimizes_directly_on_unit_sphere() -> None:
    source = _clustered_vectors()

    coordinates, report = reduce_to_surface(
        source,
        SurfaceConfig(
            teacher_k=4,
            max_epochs=150,
            patience=75,
            restarts=1,
            seed=3,
        ),
    )

    assert coordinates.shape == (30, 3)
    assert np.linalg.norm(coordinates, axis=1) == pytest.approx(np.ones(30))
    assert report["algorithm"] == "direct_spherical_neighbor_distribution_embedding"


def test_spherical_cap_mapping_stays_on_surface_and_inside_cap() -> None:
    center = _fibonacci_sphere(5)[0]
    layout = np.asarray(
        [[-1.0, -0.5], [0.0, 0.0], [0.5, 1.0], [1.0, -0.5]],
        dtype=np.float32,
    )

    coordinates = _map_to_spherical_cap(layout, center, cap_radius=0.4)
    angles = np.arccos(np.clip(coordinates @ center, -1.0, 1.0))

    assert np.linalg.norm(coordinates, axis=1) == pytest.approx(np.ones(4))
    assert float(angles.max()) <= 0.400001


def test_spherical_cap_quantile_scaling_prevents_outlier_compression() -> None:
    center = _fibonacci_sphere(5)[0]
    layout = np.asarray(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [20.0, 0.0]],
        dtype=np.float32,
    )

    maximum_scaled = _map_to_spherical_cap(layout, center, cap_radius=0.8)
    robust_scaled = _map_to_spherical_cap(
        layout,
        center,
        cap_radius=0.8,
        radius_quantile=0.75,
    )
    maximum_angles = np.arccos(np.clip(maximum_scaled @ center, -1.0, 1.0))
    robust_angles = np.arccos(np.clip(robust_scaled @ center, -1.0, 1.0))

    assert np.median(robust_angles) > np.median(maximum_angles) * 1.5
    assert float(robust_angles.max()) <= 0.800001


def test_cluster_surface_learns_relation_aware_centers_for_local_clusters() -> None:
    source = _clustered_vectors()

    experiment = reduce_to_cluster_surface(
        source,
        ClusterSurfaceConfig(
            min_clusters=6,
            max_clusters=6,
            silhouette_tolerance=0.0,
            min_cluster_size=4,
            local_neighbors=3,
            center_epochs=200,
            center_restarts=1,
            seed=3,
        ),
    )

    assert experiment.coordinates.shape == (30, 3)
    assert np.linalg.norm(experiment.coordinates, axis=1) == pytest.approx(np.ones(30))
    assert experiment.report["selected_cluster_count"] == 6
    assert experiment.report["center_layout"] == "learned_cluster_relationship_sphere"
    assert (
        experiment.report["cluster_relationships"]["center_neighbor_recall"]["recall@2"]
        > 0.8
    )
    assert (
        experiment.report["source_neighbor_containment"]["top_3_same_cluster_rate"]
        > 0.95
    )
