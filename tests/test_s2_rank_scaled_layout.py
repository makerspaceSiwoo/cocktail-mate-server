"""Unit tests for the rank-rescaled S² layout script."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.s2_rank_scaled_layout import (
    CONTRACT_RESCALERS,
    RESCALERS,
    TOP_K,
    LayoutConfig,
    _make_objective,
    bottom_decile_false_close_count,
    check_rescaling_contract,
    enforce_top_k_band,
    normalise_rows,
    optimise_layout,
    rank_stress_weights,
    recall_metrics,
    similarity_rank_matrix,
    top_k_cap_radians,
    unit_norm_max_error,
)


def _synthetic_similarity(
    n: int = 24, seed: int = 7, spread: float = 0.25
) -> tuple[np.ndarray, list[str]]:
    """Unit vectors in 8D compressed towards a common direction, like Graph48."""

    rng = np.random.default_rng(seed)
    base = np.zeros(8)
    base[0] = 1.0
    raw = base + spread * rng.standard_normal((n, 8))
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(cosines, 1.0)
    return cosines, [str(i + 1) for i in range(n)]


# --------------------------------------------------------------------------
# cap radius
# --------------------------------------------------------------------------


def test_top_k_cap_radians_matches_the_602_node_brief_value():
    assert top_k_cap_radians(602, 5) == pytest.approx(0.1827, abs=5e-5)
    assert 1.0 - math.cos(top_k_cap_radians(602, 5)) == pytest.approx(2 * 5 / 601)


def test_top_k_cap_radians_rejects_degenerate_input():
    with pytest.raises(ValueError):
        top_k_cap_radians(1)
    with pytest.raises(ValueError):
        top_k_cap_radians(10, 0)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def test_similarity_rank_matrix_is_one_based_and_excludes_self():
    cosines, ids = _synthetic_similarity(n=12)
    ranks, order = similarity_rank_matrix(cosines, ids)
    assert np.all(np.diag(ranks) == 0)
    for i in range(12):
        assert sorted(ranks[i][np.arange(12) != i]) == list(range(1, 12))
        assert list(order[i]) == sorted(
            [j for j in range(12) if j != i], key=lambda j: ranks[i, j]
        )


def test_similarity_rank_matrix_breaks_ties_on_node_id():
    cosines = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.9],
            [0.5, 0.9, 1.0],
        ]
    )
    ranks, _ = similarity_rank_matrix(cosines, ["10", "2", "3"])
    # Tied at 0.5 from source "10": numeric ids sort ascending, so "2" wins.
    assert ranks[0, 1] == 1
    assert ranks[0, 2] == 2


# --------------------------------------------------------------------------
# rescaling contract: monotonicity and the top-5 band
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONTRACT_RESCALERS)
def test_rescaling_is_monotone_in_cosine(name):
    cosines, ids = _synthetic_similarity(n=24)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    target = RESCALERS[name](ranks, cosines)
    angles = np.arccos(np.clip(target, -1.0, 1.0))
    for i in range(len(ids)):
        others = np.array([j for j in range(len(ids)) if j != i])
        by_rank = others[np.argsort(ranks[i, others])]
        ordered_angles = angles[i, by_rank]
        ordered_cosines = cosines[i, by_rank]
        # Higher 48D cosine must never map to a strictly larger target angle.
        assert np.all(np.diff(ordered_angles) >= -1e-12)
        assert np.all(np.diff(ordered_cosines) <= 1e-12)


@pytest.mark.parametrize("name", CONTRACT_RESCALERS)
def test_rescaling_puts_rank_1_to_5_inside_the_cap_and_rank_6_outside(name):
    cosines, ids = _synthetic_similarity(n=24)
    n = len(ids)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    target = RESCALERS[name](ranks, cosines)
    angles = np.arccos(np.clip(target, -1.0, 1.0))
    cap = top_k_cap_radians(n, TOP_K)
    top = (ranks >= 1) & (ranks <= TOP_K)
    rest = ranks > TOP_K
    assert np.max(angles[top]) <= cap + 1e-12
    assert np.min(angles[rest]) > cap
    # The full [0, pi] range is used, not a 0.23 rad huddle.
    assert np.max(angles[rest]) == pytest.approx(math.pi, abs=1e-9)


@pytest.mark.parametrize("name", CONTRACT_RESCALERS)
def test_check_rescaling_contract_accepts_every_shipped_candidate(name):
    cosines, ids = _synthetic_similarity(n=24)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    contract = check_rescaling_contract(RESCALERS[name](ranks, cosines), ranks, cosines)
    assert contract["monotone"]
    assert contract["top_k_inside_cap"]
    assert contract["rest_outside_cap"]


def test_check_rescaling_contract_rejects_raw_acos():
    """Raw acos is exactly the failure mode this task exists to fix.

    With 120 nodes the 5-slot cap is 0.413 rad while the true rank-5 angles run
    past 1.1 rad, so the top-5 cannot possibly be exclusive at that target.
    """

    cosines, ids = _synthetic_similarity(n=120, spread=1.0)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    contract = check_rescaling_contract(
        RESCALERS["raw_acos"](ranks, cosines), ranks, cosines
    )
    assert contract["monotone"]
    assert not contract["top_k_inside_cap"]
    assert contract["max_target_angle_rank_le_k"] > contract["top_k_cap_radians"]


def test_enforce_top_k_band_preserves_order_and_lands_rank_k_on_the_cap():
    cosines, ids = _synthetic_similarity(n=24)
    n = len(ids)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    # Raw areas monotone in rank but bunched far from the required bands.
    raw = 0.30 + 0.002 * ranks.astype(float)
    areas = enforce_top_k_band(raw, ranks)
    cap = 2.0 * TOP_K / (n - 1)
    for i in range(n):
        assert areas[i, ranks[i] == TOP_K][0] == pytest.approx(cap)
        assert np.all(areas[i, (ranks[i] >= 1) & (ranks[i] <= TOP_K)] <= cap + 1e-12)
        assert np.all(areas[i, ranks[i] > TOP_K] > cap)
    assert np.allclose(np.diag(areas), 0.0)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_recall_metrics_separates_the_three_definitions():
    """Hand-built fixture where the three metrics are deliberately different."""

    true_top5 = np.array(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    )
    coordinate_top5 = np.array(
        [
            [1, 2, 3, 4, 5],  # 5/5 -> full recovery
            [1, 9, 10, 11, 12],  # 1/5 -> hit only
            [20, 21, 22, 23, 24],  # 0/5 -> miss
            [1, 2, 30, 31, 32],  # 2/5 -> hit only
        ]
    )
    result = recall_metrics(true_top5, coordinate_top5)
    assert result["mean_recall_at_5"] == pytest.approx((1.0 + 0.2 + 0.0 + 0.4) / 4)
    assert result["mean_recall_at_5"] == pytest.approx(0.4)
    assert result["full_recovery_rate"] == pytest.approx(0.25)
    assert result["hit_rate_at_5"] == pytest.approx(0.75)
    assert (
        result["mean_recall_at_5"]
        != result["full_recovery_rate"]
        != result["hit_rate_at_5"]
    )


def test_recall_metrics_is_order_insensitive_within_a_row():
    true_top5 = np.array([[5, 4, 3, 2, 1]])
    coordinate_top5 = np.array([[1, 2, 3, 4, 5]])
    result = recall_metrics(true_top5, coordinate_top5)
    assert result["mean_recall_at_5"] == pytest.approx(1.0)
    assert result["full_recovery_rate"] == pytest.approx(1.0)


def test_recall_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        recall_metrics(np.zeros((2, 5), dtype=int), np.zeros((2, 4), dtype=int))


def test_bottom_decile_false_close_count_counts_only_intruding_far_nodes():
    n = 21
    angles = np.full((n, n), 1.5)
    np.fill_diagonal(angles, 0.0)
    # cosine_order[i] runs nearest -> farthest and excludes self, so the bottom
    # decile is its tail: floor(0.1 * 20) = 2 nodes per source.
    cosine_order = np.asarray(
        [[j for j in range(n) if j != i] for i in range(n)], dtype=np.int64
    )
    true_top5 = np.tile(np.array([1, 2, 3, 4, 5]), (n, 1))
    true_top5[1:6] = np.array([6, 7, 8, 9, 10])
    # Source 0: bottom decile is {19, 20}. Node 20 lands at 0.1 rad, well inside
    # the farthest true top-5 coordinate distance (1.5 rad); node 19 does not.
    angles[0, 20] = 0.1
    assert bottom_decile_false_close_count(angles, cosine_order, true_top5) == 1
    # Pulling node 19 in as well makes it two.
    angles[0, 19] = 0.2
    assert bottom_decile_false_close_count(angles, cosine_order, true_top5) == 2


def test_unit_norm_max_error_detects_off_sphere_rows():
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert unit_norm_max_error(coords) == pytest.approx(1.0)
    assert unit_norm_max_error(normalise_rows(coords)) <= 1e-15


# --------------------------------------------------------------------------
# optimiser
# --------------------------------------------------------------------------


def test_normalise_rows_produces_unit_vectors_including_degenerate_rows():
    coords = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    unit = normalise_rows(coords)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
    assert np.allclose(unit[1], [0.0, 0.0, 1.0])


def test_objective_gradient_matches_finite_differences():
    cosines, ids = _synthetic_similarity(n=10, seed=3)
    ranks, order = similarity_rank_matrix(cosines, ids)
    target = RESCALERS["rank_uniform_area"](ranks, cosines)
    weights = rank_stress_weights(ranks)
    config = LayoutConfig(hinge_weight=0.7, hinge_margin=0.1)
    objective = _make_objective(target, weights, order[:, :TOP_K], config, 0.05)
    rng = np.random.default_rng(11)
    point = rng.standard_normal(10 * 3)
    _, analytic = objective(point)
    step = 1e-6
    numeric = np.zeros_like(point)
    for index in range(point.size):
        shifted = point.copy()
        shifted[index] += step
        high, _ = objective(shifted)
        shifted[index] -= 2 * step
        low, _ = objective(shifted)
        numeric[index] = (high - low) / (2 * step)
    assert np.allclose(analytic, numeric, rtol=2e-4, atol=2e-5)


def test_optimise_layout_returns_unit_norm_coordinates():
    cosines, ids = _synthetic_similarity(n=16, seed=5)
    ranks, order = similarity_rank_matrix(cosines, ids)
    target = RESCALERS["rank_uniform_area"](ranks, cosines)
    weights = rank_stress_weights(ranks)
    config = LayoutConfig(multistart_count=2, max_iterations=40)
    result = optimise_layout(target, weights, order[:, :TOP_K], config)
    assert result.coordinates.shape == (16, 3)
    assert unit_norm_max_error(result.coordinates) <= 1e-12
    assert len(result.start_objectives) == 2
    assert result.start_objectives[result.selected_start] == min(
        result.start_objectives
    )


def test_optimise_layout_is_deterministic_for_a_fixed_seed():
    cosines, ids = _synthetic_similarity(n=14, seed=9)
    ranks, order = similarity_rank_matrix(cosines, ids)
    target = RESCALERS["rank_uniform_area"](ranks, cosines)
    weights = rank_stress_weights(ranks)
    config = LayoutConfig(multistart_count=2, max_iterations=30)
    first = optimise_layout(target, weights, order[:, :TOP_K], config)
    second = optimise_layout(target, weights, order[:, :TOP_K], config)
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.start_objectives == second.start_objectives


def test_rank_stress_weights_decay_with_rank_and_zero_the_diagonal():
    cosines, ids = _synthetic_similarity(n=12)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    weights = rank_stress_weights(ranks)
    assert np.allclose(np.diag(weights), 0.0)
    for i in range(12):
        others = np.array([j for j in range(12) if j != i])
        by_rank = others[np.argsort(ranks[i, others])]
        assert np.all(np.diff(weights[i, by_rank]) < 0.0)
