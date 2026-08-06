"""Tests for the globally stretched S² layouts (linear and convex modes)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import s2_ratio_scaled_layout as layout

THETA_MAX = 1.4695933665545104


# ---------------------------------------------------------------------------
# convex stretch contract — the four conditions the target function must meet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1.1, 1.2, 1.4, 1.6, 1.8])
@pytest.mark.parametrize("gamma", [1.5, 2.0, 3.0])
def test_convex_stretch_is_strictly_increasing(a: float, gamma: float) -> None:
    rng = np.random.default_rng(11)
    left = rng.uniform(1e-6, THETA_MAX, 5000)
    right = rng.uniform(1e-6, THETA_MAX, 5000)
    f_left = layout.convex_stretch(left, a, gamma, THETA_MAX)
    f_right = layout.convex_stretch(right, a, gamma, THETA_MAX)
    smaller = left < right
    assert np.all(f_left[smaller] < f_right[smaller])
    assert np.all(f_left[~smaller] >= f_right[~smaller])


@pytest.mark.parametrize("a", [1.1, 1.4, 1.8])
@pytest.mark.parametrize("gamma", [1.5, 2.0, 3.0])
def test_convex_stretch_is_convex(a: float, gamma: float) -> None:
    grid = np.linspace(0.0, THETA_MAX, 4001)
    values = layout.convex_stretch(grid, a, gamma, THETA_MAX)
    assert np.min(np.diff(values, 2)) >= -1e-15


@pytest.mark.parametrize("a", [1.1, 1.4, 1.8])
@pytest.mark.parametrize("gamma", [1.5, 2.0, 3.0])
def test_convex_stretch_expands_even_the_closest_pairs(a: float, gamma: float) -> None:
    grid = np.linspace(1e-12, THETA_MAX, 4001)
    values = layout.convex_stretch(grid, a, gamma, THETA_MAX)
    assert np.all(values > grid)
    assert float(np.min(values / grid)) == pytest.approx(a, abs=1e-5)
    assert float(np.min(values / grid)) >= a


@pytest.mark.parametrize("a", [1.1, 1.4, 1.8])
@pytest.mark.parametrize("gamma", [1.5, 2.0, 3.0])
def test_convex_stretch_reaches_pi_at_theta_max(a: float, gamma: float) -> None:
    value = layout.convex_stretch(np.array([THETA_MAX]), a, gamma, THETA_MAX)[0]
    assert value == pytest.approx(math.pi, abs=1e-12)


def test_convex_stretch_rejects_a_at_or_above_pi_over_theta_max() -> None:
    limit = math.pi / THETA_MAX
    with pytest.raises(ValueError, match="pi/theta_max"):
        layout.convex_stretch_coefficient(limit, 2.0, THETA_MAX)
    with pytest.raises(ValueError, match="pi/theta_max"):
        layout.convex_stretch_coefficient(limit + 0.5, 2.0, THETA_MAX)


def test_convex_stretch_rejects_non_expanding_or_non_convex_parameters() -> None:
    with pytest.raises(ValueError, match="a must exceed 1"):
        layout.convex_stretch_coefficient(0.9, 2.0, THETA_MAX)
    with pytest.raises(ValueError, match="gamma must exceed 1"):
        layout.convex_stretch_coefficient(1.2, 1.0, THETA_MAX)


def test_convex_contract_report_flags_all_four_conditions() -> None:
    report = layout.convex_contract_report(1.4, 2.0, THETA_MAX)
    assert report["monotone_increasing"] is True
    assert report["convex"] is True
    assert report["expands_everywhere"] is True
    assert report["b"] > 0.0
    assert report["f_at_theta_max"] == pytest.approx(math.pi, abs=1e-12)
    assert report["a_upper_bound"] == pytest.approx(math.pi / THETA_MAX)


def test_target_monotonicity_has_no_rank_inversions() -> None:
    rng = np.random.default_rng(3)
    angles = rng.uniform(0.05, THETA_MAX, 20000)
    targets = layout.convex_stretch(angles, 1.4, 2.0, THETA_MAX)
    assert layout.target_monotonicity_violations(angles, targets) == 0
    scrambled = targets.copy()
    scrambled[0], scrambled[1] = float(np.max(targets)), float(np.min(targets))
    assert layout.target_monotonicity_violations(angles, scrambled) > 0


# ---------------------------------------------------------------------------
# linear mode — the global scalar candidates
# ---------------------------------------------------------------------------


def test_linear_targets_preserve_every_pair_ratio_exactly() -> None:
    rng = np.random.default_rng(5)
    angles = rng.uniform(0.1, THETA_MAX, 4000)
    k = layout.k_max_antipodal(angles)
    targets = layout.scaled_targets(angles, k)
    assert layout.ratio_preservation_error(angles, targets) < 1e-12
    left = rng.integers(0, angles.size, 500)
    right = rng.integers(0, angles.size, 500)
    assert np.allclose(
        targets[left] / targets[right], angles[left] / angles[right], rtol=1e-12
    )


def test_k_candidates_follow_their_definitions() -> None:
    angles = np.array([0.2, 0.5, 0.9, 1.2, 1.4])
    assert layout.k_max_antipodal(angles) == pytest.approx(math.pi / 1.4)
    assert layout.k_median_match(angles) == pytest.approx((math.pi / 2) / 0.9)


def test_k_ks_optimal_beats_the_other_candidates_on_ks() -> None:
    rng = np.random.default_rng(7)
    angles = rng.uniform(0.14, THETA_MAX, 20000)
    best_k, best_ks = layout.k_ks_optimal(angles)
    for other in (
        layout.k_max_antipodal(angles),
        layout.k_median_match(angles),
        best_k * 1.02,
        best_k * 0.98,
    ):
        other_ks = layout.ks_statistic_uniform_sphere(
            layout.scaled_targets(angles, other)
        )
        assert best_ks <= other_ks + 1e-12
    assert best_ks == pytest.approx(
        layout.ks_statistic_uniform_sphere(layout.scaled_targets(angles, best_k))
    )


def test_ks_statistic_is_zero_for_an_exact_uniform_sphere_sample() -> None:
    grid = (np.arange(1, 10001) - 0.5) / 10000.0
    sample = np.arccos(1.0 - 2.0 * grid)
    assert layout.ks_statistic_uniform_sphere(sample) < 1e-4
    assert layout.ks_statistic_uniform_sphere(np.full(1000, 0.3)) > 0.4


def test_clamp_report_counts_pairs_pushed_past_pi() -> None:
    angles = np.array([0.5, 1.0, 1.5])
    report = layout.clamp_report(angles, 2.5)
    assert report["clamped_pair_count"] == 1
    assert report["clamped_pair_fraction"] == pytest.approx(1 / 3)
    assert layout.clamp_report(angles, 4.0)["clamped_pair_count"] == 2
    assert layout.clamp_report(angles, math.pi / 1.5)["clamped_pair_count"] == 0


# ---------------------------------------------------------------------------
# coverage metrics — uniform and clustered fixtures must disagree
# ---------------------------------------------------------------------------


def _points_in_polar_cap(count: int, cap_radius: float, seed: int) -> np.ndarray:
    """``count`` points confined to a polar cap of angular radius ``cap_radius``.

    Sampled uniformly in ``(z, phi)`` inside the cap, so the fixture really is a
    tight cluster rather than a whole-sphere sample with a rescaled z column.
    """

    rng = np.random.default_rng(seed)
    z = rng.uniform(math.cos(cap_radius), 1.0, count)
    phi = rng.uniform(0.0, 2.0 * math.pi, count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def test_nearest_neighbour_angles_separate_uniform_from_clustered() -> None:
    spread = layout.fibonacci_sphere(200)
    clustered = _points_in_polar_cap(200, 0.05, seed=1)
    spread_nn = layout.nearest_neighbour_angles(spread)
    clustered_nn = layout.nearest_neighbour_angles(clustered)
    assert float(np.median(spread_nn)) > 5.0 * float(np.median(clustered_nn))
    assert layout.angle_summary(spread_nn)["min"] > 0.0


def test_covering_radius_is_much_larger_for_a_clustered_set() -> None:
    probes = layout.fibonacci_sphere(4096)
    spread = layout.fibonacci_sphere(200)
    clustered = _points_in_polar_cap(200, 0.05, seed=2)
    assert layout.covering_radius(spread, probes) < 0.25
    assert layout.covering_radius(clustered, probes) > 2.5


def test_equal_area_cell_counts_match_hand_computed_cells() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    counts = layout.equal_area_cell_counts(coords, 2, 2)
    assert counts.tolist() == [0, 1, 1, 1]
    assert int(np.sum(counts)) == coords.shape[0]


def test_cell_occupancy_separates_uniform_from_clustered() -> None:
    spread = layout.fibonacci_sphere(602)
    clustered = _points_in_polar_cap(602, 0.05, seed=3)
    spread_summary = layout.cell_occupancy_summary(
        layout.equal_area_cell_counts(spread, 10, 10), 602
    )
    clustered_summary = layout.cell_occupancy_summary(
        layout.equal_area_cell_counts(clustered, 10, 10), 602
    )
    assert spread_summary["empty_cells"] == 0
    assert clustered_summary["empty_cells"] >= 90
    assert clustered_summary["std"] > 10.0 * spread_summary["std"]
    assert spread_summary["uniform_expected_std"] == pytest.approx(
        math.sqrt(602 * 0.01 * 0.99)
    )


def test_uniform_nn_marginal_quantile_matches_the_closed_form() -> None:
    theta = layout.uniform_nn_marginal_quantile(602, 0.5)
    survival = ((1.0 + math.cos(theta)) / 2.0) ** 601
    assert survival == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# cluster separation
# ---------------------------------------------------------------------------


def test_cluster_separation_ratio_rewards_separated_clusters() -> None:
    labels = np.array(["a"] * 20 + ["b"] * 20)
    rng = np.random.default_rng(9)
    tight = np.vstack(
        [
            layout.normalise_rows(
                np.array([0.0, 0.0, 1.0]) + 0.02 * rng.standard_normal((20, 3))
            ),
            layout.normalise_rows(
                np.array([0.0, 0.0, -1.0]) + 0.02 * rng.standard_normal((20, 3))
            ),
        ]
    )
    mixed = layout.normalise_rows(rng.standard_normal((40, 3)))
    separated = layout.cluster_separation_metrics(tight, labels)
    blended = layout.cluster_separation_metrics(mixed, labels)
    assert separated["separation_ratio"] > 10.0 * blended["separation_ratio"]
    assert separated["silhouette_mean"] > 0.9
    assert blended["silhouette_mean"] < 0.2
    assert separated["centroid_angle_min"] > 3.0
    assert layout.separation_ratio(tight, labels) == pytest.approx(
        separated["separation_ratio"]
    )


# ---------------------------------------------------------------------------
# top-k metrics
# ---------------------------------------------------------------------------


def test_recall_metrics_give_three_different_values() -> None:
    true_top_k = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
        ]
    )
    coordinate_top_k = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 20, 21, 22],
            [30, 31, 32, 33, 34],
        ]
    )
    metrics = layout.recall_metrics(true_top_k, coordinate_top_k)
    assert metrics["mean_recall_at_5"] == pytest.approx((1.0 + 0.4 + 0.0) / 3.0)
    assert metrics["full_recovery_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["hit_rate_at_5"] == pytest.approx(2.0 / 3.0)
    assert len({round(value, 6) for value in metrics.values()}) == 3


def test_unit_norm_max_error_detects_off_sphere_rows() -> None:
    coords = layout.fibonacci_sphere(16)
    assert layout.unit_norm_max_error(coords) < 1e-12
    broken = coords.copy()
    broken[3] *= 1.5
    assert layout.unit_norm_max_error(broken) == pytest.approx(0.5)


def test_normalise_rows_returns_unit_vectors() -> None:
    rng = np.random.default_rng(13)
    coords = layout.normalise_rows(rng.standard_normal((50, 3)) * 7.0)
    assert layout.unit_norm_max_error(coords) <= 1e-12
    fallback = layout.normalise_rows(np.zeros((2, 3)))
    assert np.allclose(fallback, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))


def test_similarity_rank_matrix_breaks_ties_on_ascending_id() -> None:
    similarity = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.2],
            [0.5, 0.2, 1.0],
        ]
    )
    ranks, order = layout.similarity_rank_matrix(similarity, [30, 10, 20])
    assert order[0].tolist() == [1, 2]
    assert ranks[0, 1] == 1 and ranks[0, 2] == 2


# ---------------------------------------------------------------------------
# optimiser and refinement machinery
# ---------------------------------------------------------------------------


def test_objective_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(17)
    n = 6
    targets = np.full((n, n), 1.4)
    np.fill_diagonal(targets, 0.0)
    objective = layout.make_objective(targets)
    flat = rng.standard_normal(n * 3)
    value, grad = objective(flat)
    step = 1e-6
    for index in range(flat.size):
        shifted = flat.copy()
        shifted[index] += step
        upper, _ = objective(shifted)
        shifted[index] -= 2.0 * step
        lower, _ = objective(shifted)
        assert grad[index] == pytest.approx((upper - lower) / (2.0 * step), rel=1e-4)
    assert value > 0.0


def test_optimise_layout_is_deterministic_and_on_the_sphere() -> None:
    rng = np.random.default_rng(19)
    angles = layout.angle_matrix(
        layout.exact_cosine_matrix(rng.standard_normal((24, 8)))
    )
    theta_max = float(np.max(angles))
    targets = layout.convex_stretch(angles, 1.1, 2.0, theta_max)
    np.fill_diagonal(targets, 0.0)
    config = layout.LayoutConfig(multistart_count=2, max_iterations=120)
    first = layout.optimise_layout(targets, config)
    second = layout.optimise_layout(targets, config)
    assert np.array_equal(first.coordinates, second.coordinates)
    assert layout.unit_norm_max_error(first.coordinates) <= 1e-12
    assert first.start_objectives == second.start_objectives


def _true_mask(n: int, true_top_k: np.ndarray) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[np.arange(n)[:, None], true_top_k] = True
    return mask


def _brute_force_hits(coords: np.ndarray, mask: np.ndarray) -> int:
    dots = coords @ coords.T
    np.fill_diagonal(dots, -2.0)
    top = np.argpartition(-dots, layout.TOP_K, axis=1)[:, : layout.TOP_K]
    return int(sum(int(np.sum(mask[i, top[i]])) for i in range(coords.shape[0])))


def test_recall_state_incremental_updates_match_a_full_recount() -> None:
    rng = np.random.default_rng(23)
    coords = layout.normalise_rows(rng.standard_normal((40, 3)))
    reference = layout.normalise_rows(rng.standard_normal((40, 3)))
    dots = reference @ reference.T
    np.fill_diagonal(dots, -2.0)
    true_top_k = np.argpartition(-dots, layout.TOP_K, axis=1)[:, : layout.TOP_K]
    mask = _true_mask(40, true_top_k)
    state = layout._RecallState(coords, mask)
    assert state.total_hits() == _brute_force_hits(coords, mask)
    for node in (0, 7, 19, 39):
        point = layout.normalise_rows(rng.standard_normal((1, 3)))[0]
        predicted, dv = state.score_move(node, point)
        state.apply_move(node, point, dv)
        assert state.total_hits() == predicted
        assert state.total_hits() == _brute_force_hits(state.coords, mask)


def test_refinement_improves_recall_without_breaking_the_constraints() -> None:
    rng = np.random.default_rng(29)
    vectors = rng.standard_normal((60, 8))
    _, order = layout.similarity_rank_matrix(
        layout.exact_cosine_matrix(vectors), list(range(60))
    )
    true_top_k = order[:, : layout.TOP_K]
    coords = layout.normalise_rows(rng.standard_normal((60, 3)))
    labels = np.asarray(["a"] * 30 + ["b"] * 30)
    before = layout.recall_metrics(
        true_top_k,
        layout.similarity_rank_matrix(
            np.clip(coords @ coords.T, -1.0, 1.0), list(range(60))
        )[1][:, : layout.TOP_K],
    )
    config = layout.RefineConfig(max_passes=3, directions_per_radius=2)
    refined, report = layout.refine_recall_constrained(
        coords, true_top_k, labels, config
    )
    after = layout.recall_metrics(
        true_top_k,
        layout.similarity_rank_matrix(
            np.clip(refined @ refined.T, -1.0, 1.0), list(range(60))
        )[1][:, : layout.TOP_K],
    )
    assert after["mean_recall_at_5"] >= before["mean_recall_at_5"]
    assert layout.unit_norm_max_error(refined) <= 1e-12
    assert report["sweeps_run"] >= 1
    constraints = report["constraints"]
    assert (
        layout.separation_ratio(refined, labels)
        >= constraints["separation_ratio_floor"] - 1e-12
    )
    assert (
        layout.ks_statistic_uniform_sphere(
            layout.upper_triangle(layout.coordinate_angle_matrix(refined))
        )
        <= constraints["pair_angle_ks_ceiling"] + 1e-12
    )


# ---------------------------------------------------------------------------
# id normalisation — the string/int join hazard
# ---------------------------------------------------------------------------


def test_canonical_id_normalises_strings_and_ints() -> None:
    assert layout.canonical_id("1") == 1
    assert layout.canonical_id(1) == 1
    assert layout.canonical_id(" 42 ") == 42
    with pytest.raises(ValueError):
        layout.canonical_id("")
    with pytest.raises(ValueError):
        layout.canonical_id(True)


def test_coordinates_in_id_order_requires_every_node() -> None:
    lookup = {1: (0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0)}
    coords = layout.coordinates_in_id_order(lookup, [2, 1])
    assert coords.tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="missing"):
        layout.coordinates_in_id_order(lookup, [1, 3])


# ---------------------------------------------------------------------------
# realised (coordinate-level) stretch — the user-visible near-field contract
# ---------------------------------------------------------------------------


def test_realised_stretch_profile_detects_near_field_compression() -> None:
    source = np.linspace(0.1, 1.4, 1000)
    expanded = source * 2.0
    profile = layout.realised_stretch_by_source_decile(source, expanded)
    assert len(profile) == layout.STRETCH_DECILES
    assert all(row["median"] == pytest.approx(2.0) for row in profile)
    assert all(row["fraction_below_1"] == 0.0 for row in profile)

    squeezed = expanded.copy()
    nearest = source <= np.quantile(source, 0.1)
    squeezed[nearest] = source[nearest] * 0.9
    squeezed_profile = layout.realised_stretch_by_source_decile(source, squeezed)
    assert squeezed_profile[0]["median"] == pytest.approx(0.9)
    assert squeezed_profile[0]["fraction_below_1"] == pytest.approx(1.0)
    assert squeezed_profile[-1]["median"] == pytest.approx(2.0)


def test_nearest_fraction_compressed_counts_only_the_closest_pairs() -> None:
    source = np.linspace(0.1, 1.4, 1000)
    observed = source * 2.0
    assert layout.nearest_fraction_compressed(source, observed) == 0.0
    observed[:50] = source[:50] * 0.5
    assert layout.nearest_fraction_compressed(source, observed) == pytest.approx(1.0)
    assert layout.nearest_fraction_compressed(
        source, observed, fraction=0.5
    ) == pytest.approx(0.1)


def test_order_fidelity_always_reports_the_realised_stretch_fields() -> None:
    rng = np.random.default_rng(31)
    coords = layout.normalise_rows(rng.standard_normal((40, 3)))
    angles = layout.angle_matrix(
        layout.exact_cosine_matrix(rng.standard_normal((40, 6)))
    )
    metrics = layout.order_fidelity_metrics(coords, angles, inversion_samples=5000)
    for key in (
        "realised_stretch_by_source_decile",
        "near_decile_median_stretch",
        "nearest_5pct_fraction_compressed",
        "all_pairs_fraction_compressed",
    ):
        assert key in metrics
    # No target supplied: target-relative fields must be None, not a foreign value.
    assert metrics["target_reference"] is None
    assert metrics["angle_stress"] is None
    assert metrics["normalised_angle_stress"] is None
    assert metrics["pearson_coord_vs_target"] is None

    targets = 2.0 * angles
    with_targets = layout.order_fidelity_metrics(
        coords, angles, targets, inversion_samples=5000
    )
    assert with_targets["angle_stress"] > 0.0
    assert with_targets["target_reference"] == "own layout target"


# ---------------------------------------------------------------------------
# gates, floors and selection logic
# ---------------------------------------------------------------------------


def test_check_thresholds_reports_each_failed_metric() -> None:
    row = {"a": 0.6, "b": 10.0}
    thresholds = {"a": (">=", 0.5), "b": ("<=", 5.0)}
    result = layout.check_thresholds(row, thresholds)
    assert result["passed"] is False
    assert result["failed"] == ["b"]
    assert result["checks"]["a"]["passed"] is True
    assert result["checks"]["b"]["value"] == 10.0
    assert layout.check_thresholds({"a": 0.6, "b": 1.0}, thresholds)["passed"] is True


def test_revised_gates_are_looser_than_the_original_ones() -> None:
    row = {
        "mean_recall_at_5": 0.5636,
        "hit_rate_at_5": 0.97,
        "union_edge_rmse_radians_original_acos": 0.24,
        "unit_norm_max_error": 2.3e-16,
        "bottom_decile_false_close_count": 420.0,
    }
    assert layout.check_thresholds(row, layout.REVISED_ACCEPTANCE_GATES)["passed"]
    original = layout.check_thresholds(row, layout.ORIGINAL_ACCEPTANCE_GATES)
    assert original["passed"] is False
    assert set(original["failed"]) == {
        "mean_recall_at_5",
        "bottom_decile_false_close_count",
    }


def test_non_regression_floors_include_the_near_field_bounds() -> None:
    assert layout.NON_REGRESSION_FLOORS["near_decile_median_stretch"] == (">=", 1.00)
    assert layout.NON_REGRESSION_FLOORS["nearest_5pct_fraction_compressed"] == (
        "<=",
        0.20,
    )
    failing = {key: 0.0 for key in layout.NON_REGRESSION_FLOORS}
    failing["nearest_5pct_fraction_compressed"] = 0.60
    failing["near_decile_median_stretch"] = 0.9716
    result = layout.check_thresholds(failing, layout.NON_REGRESSION_FLOORS)
    assert result["passed"] is False
    assert "near_decile_median_stretch" in result["failed"]
    assert "nearest_5pct_fraction_compressed" in result["failed"]


def test_borda_rank_sums_positions_on_all_three_axes() -> None:
    rows = {
        "best": {"separation": 3.0, "recall": 0.9, "ks": 0.10},
        "middle": {"separation": 2.0, "recall": 0.5, "ks": 0.20},
        "worst": {"separation": 1.0, "recall": 0.1, "ks": 0.30},
    }
    totals = layout.borda_rank(rows)
    assert totals == {"best": 0.0, "middle": 3.0, "worst": 6.0}
    # Winning one axis and losing two must lose to the reverse.
    mixed = layout.borda_rank(
        {
            "a": {"separation": 3.0, "recall": 0.1, "ks": 0.30},
            "b": {"separation": 1.0, "recall": 0.9, "ks": 0.10},
        }
    )
    assert mixed["a"] == 2.0
    assert mixed["b"] == 1.0


def test_select_linear_candidate_prefers_lowest_ks_within_the_clamp_budget() -> None:
    def run(clamped: float, ks: float) -> dict:
        return {
            "clamp": {"clamped_pair_fraction": clamped},
            "evaluation": {"coverage": {"pair_angle_ks_vs_uniform_sphere": ks}},
        }

    runs = {
        "k_max_antipodal": run(0.0, 0.11),
        "k_median_match": run(1e-4, 0.09),
        "k_ks_optimal": run(3e-5, 0.10),
    }
    assert layout.select_linear_candidate(runs) == "k_median_match"
    # A candidate that clamps too much is ineligible however good its KS is.
    runs["k_median_match"] = run(0.5, 0.01)
    assert layout.select_linear_candidate(runs) == "k_ks_optimal"
    # Nothing eligible at all falls back to the zero-clamp candidate.
    runs = {name: run(0.5, 0.01) for name in runs}
    assert layout.select_linear_candidate(runs) == "k_max_antipodal"


# ---------------------------------------------------------------------------
# refinement — the rejection path must actually reject
# ---------------------------------------------------------------------------


def test_refinement_rejects_every_move_under_an_impossible_ks_ceiling() -> None:
    rng = np.random.default_rng(37)
    vectors = rng.standard_normal((60, 8))
    _, order = layout.similarity_rank_matrix(
        layout.exact_cosine_matrix(vectors), list(range(60))
    )
    true_top_k = order[:, : layout.TOP_K]
    coords = layout.normalise_rows(rng.standard_normal((60, 3)))
    labels = np.asarray(["a"] * 30 + ["b"] * 30)
    config = layout.RefineConfig(
        max_passes=2, directions_per_radius=2, ks_max_absolute_increase=-1.0
    )
    refined, report = layout.refine_recall_constrained(
        coords, true_top_k, labels, config
    )
    assert report["constraints"]["rejected_by_ks"] > 0
    assert report["moved_node_count"] == 0
    assert np.allclose(refined, coords, rtol=0.0, atol=1e-15)


def test_count_degenerate_rows_sees_the_silent_pole_fallback() -> None:
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1e-15, 0.0]])
    assert layout.count_degenerate_rows(coords) == 2
    normalised = layout.normalise_rows(coords)
    assert layout.unit_norm_max_error(normalised) <= 1e-12
    assert layout.count_degenerate_rows(normalised) == 0
