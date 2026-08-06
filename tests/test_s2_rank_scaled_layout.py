"""Unit tests for the rank-rescaled S² layout script."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc

from scripts.s2_rank_scaled_layout import (
    CONTRACT_RESCALERS,
    RESCALERS,
    TOP_K,
    DegenerateRescalingError,
    LayoutConfig,
    _make_objective,
    acceptance_report,
    baseline_compatible_layout_report,
    bottom_decile_false_close_count,
    check_rescaling_contract,
    coordinate_angles_and_ranks,
    enforce_top_k_band,
    exact_cosine_matrix,
    layout_provenance,
    load_graph48,
    normalise_rows,
    optimise_layout,
    rank_stress_weights,
    recall_metrics,
    rescale_cosine_zcdf_band,
    similarity_rank_matrix,
    standard_normal_tail,
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
    # Pairwise, not chained: `a != b != c` never compares a with c.
    assert result["mean_recall_at_5"] != result["full_recovery_rate"]
    assert result["full_recovery_rate"] != result["hit_rate_at_5"]
    assert result["mean_recall_at_5"] != result["hit_rate_at_5"]
    assert (
        len(
            {
                result["mean_recall_at_5"],
                result["full_recovery_rate"],
                result["hit_rate_at_5"],
            }
        )
        == 3
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


# --------------------------------------------------------------------------
# numerical degeneracy must fail loudly, never silently (review finding M3)
# --------------------------------------------------------------------------


def test_standard_normal_tail_matches_erfc_in_the_safe_range():
    z = np.linspace(-5.0, 5.0, 401)
    assert np.allclose(standard_normal_tail(z), 0.5 * erfc(z / math.sqrt(2.0)))


def test_standard_normal_tail_raises_when_cancellation_destroys_precision():
    """`1 - 0.5*(1 + erf(z/sqrt2))` underflows to exactly 0 for large z."""

    z = np.array([0.0, 1.0, 9.0])
    assert 1.0 - 0.5 * (1.0 + math.erf(9.0 / math.sqrt(2.0))) == 0.0
    with pytest.raises(DegenerateRescalingError, match="cancellation"):
        standard_normal_tail(z)


def test_enforce_top_k_band_raises_instead_of_clamping_a_zero_rank_k_area():
    cosines, ids = _synthetic_similarity(n=24)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    areas = 0.30 + 0.002 * ranks.astype(float)
    areas[3, ranks[3] <= TOP_K] = 0.0  # rank-k area collapses for source 3
    with pytest.raises(DegenerateRescalingError, match="collapse"):
        enforce_top_k_band(areas, ranks)


def test_enforce_top_k_band_raises_on_a_zero_outer_band_span():
    cosines, ids = _synthetic_similarity(n=24)
    ranks, _ = similarity_rank_matrix(cosines, ids)
    areas = 0.30 + 0.002 * ranks.astype(float)
    areas[7, ranks[7] > TOP_K] = 0.9  # every outer-band node gets the same area
    with pytest.raises(DegenerateRescalingError, match="span"):
        enforce_top_k_band(areas, ranks)


def test_cosine_zcdf_band_rejects_a_cohort_whose_top5_tail_underflows():
    """End-to-end regression for the silent all-zero-target failure mode.

    Five identical outlier neighbours among 199 others push their z-score past
    the point where the tail probability underflows, which used to produce five
    target angles of exactly 0 that still satisfied the top-5 cap contract.
    """

    n = 200
    similarity = np.tile(np.linspace(0.0, 0.02, n), (n, 1))
    similarity[:, 10:15] = 1.0
    np.fill_diagonal(similarity, 1.0)
    ranks, _ = similarity_rank_matrix(similarity, [str(i + 1) for i in range(n)])
    with pytest.raises(DegenerateRescalingError):
        rescale_cosine_zcdf_band(ranks, similarity)


def test_cosine_zcdf_band_rejects_a_source_with_zero_cosine_spread():
    n = 20
    similarity = np.tile(np.linspace(0.1, 0.9, n), (n, 1))
    similarity[4, :] = 0.5  # source 4 sees every other node identically
    ranks, _ = similarity_rank_matrix(similarity, [str(i + 1) for i in range(n)])
    with pytest.raises(DegenerateRescalingError, match="spread"):
        rescale_cosine_zcdf_band(ranks, similarity)


# --------------------------------------------------------------------------
# artifact schema and acceptance reporting (review findings I1, I2)
# --------------------------------------------------------------------------


_BASELINE_LAYOUT_REPORT = {
    "acceptance_checks": {},
    "acceptance_passed": True,
    "acceptance_thresholds": {
        "bottom_decile_false_close_count_max": 0,
        "mean_recall_at_5_min": 0.60,
        "node_coverage_at_5_min": 0.90,
        "union_edge_rmse_radians_max": 0.40,
        "unit_norm_max_error_max": 1e-12,
    },
    "algorithm": "baseline",
    "audit_similarity_supplied": True,
    "bottom_decile_false_close_count": 0,
    "bottom_decile_false_close_policy": "baseline policy",
    "clustering_policy": "cosine_k_medoids_v1",
    "edge_target_rmse_radians": 0.3,
    "mean_recall_at_5": 0.0,
    "node_coverage_at_5": 0.0,
    "nodes_with_true_top5_count": 167,
    "private_hub_count": 7,
    "private_hub_edge_count": 608,
    "ranking_constraint_count": 13488,
    "report_only": True,
    "sampled_nonedge_count": 10678,
    "seed": 20260806,
    "union_edge_rmse_radians": 0.3,
    "unit_norm_max_error": 0.0,
}

_PASSING_METRICS = {
    "mean_recall_at_5": 0.75,
    "hit_rate_at_5": 0.95,
    "bottom_decile_false_close_count": 0,
    "union_edge_rmse_radians_original_acos": 0.2,
    "union_edge_rmse_radians_rescaled_target": 0.15,
    "unit_norm_max_error": 2.2e-16,
}


def test_acceptance_report_flags_every_failing_gate():
    thresholds = _BASELINE_LAYOUT_REPORT["acceptance_thresholds"]
    checks, passed = acceptance_report(_PASSING_METRICS, thresholds)
    assert passed
    assert all(checks.values())

    failing = dict(_PASSING_METRICS)
    failing["mean_recall_at_5"] = 0.4309
    failing["hit_rate_at_5"] = 0.8970
    failing["bottom_decile_false_close_count"] = 237
    checks, passed = acceptance_report(failing, thresholds)
    assert passed is False
    assert checks["mean_recall_at_5"] is False
    assert checks["node_coverage_at_5"] is False
    assert checks["bottom_decile_false_close_count"] is False
    # The two gates that genuinely pass must not be dragged down with them.
    assert checks["union_edge_rmse_radians"] is True
    assert checks["unit_norm_max_error"] is True


def test_layout_report_keeps_every_baseline_key_and_reports_failure():
    failing = dict(_PASSING_METRICS)
    failing["mean_recall_at_5"] = 0.4309
    report = baseline_compatible_layout_report(
        failing,
        _BASELINE_LAYOUT_REPORT,
        {"algorithm": "rank_rescaled_spherical_nca_v1", "seed": 20260806},
    )
    assert not set(_BASELINE_LAYOUT_REPORT) - set(report)
    assert report["acceptance_passed"] is False
    assert report["union_edge_rmse_radians"] == 0.2
    assert report["edge_target_rmse_radians"] == 0.15
    assert report["node_coverage_at_5"] == 0.95
    # Keys this algorithm cannot fill are present but null, with a reason.
    for key in (
        "nodes_with_true_top5_count",
        "ranking_constraint_count",
        "sampled_nonedge_count",
    ):
        assert report[key] is None
        assert report[f"{key}_null_reason"]


def test_layout_report_rejects_a_dropped_baseline_key():
    extended = dict(_BASELINE_LAYOUT_REPORT)
    extended["some_future_baseline_key"] = 1
    with pytest.raises(ValueError, match="missing baseline keys"):
        baseline_compatible_layout_report(_PASSING_METRICS, extended, {})


def test_layout_provenance_identifies_this_run_and_keeps_lineage():
    baseline_provenance = {
        "run_id": "full602-vertex-batch-20260806",
        "layout_method": "graph-only; no high-dimensional coordinate projection",
        "registry_sha256": "abc",
        "cohort_ids_sha256": "def",
    }
    provenance = layout_provenance(baseline_provenance, "deadbeef")
    assert provenance["run_id"] != baseline_provenance["run_id"]
    assert provenance["layout_method"] != baseline_provenance["layout_method"]
    assert provenance["source_run_id"] == baseline_provenance["run_id"]
    assert provenance["coordinate_sha256"] == "deadbeef"
    assert provenance["layout_algorithm"]
    # Lineage digests are inherited unchanged.
    assert provenance["registry_sha256"] == "abc"
    assert provenance["cohort_ids_sha256"] == "def"


# --------------------------------------------------------------------------
# real-data smoke test (review finding M2) — skipped when artifacts are absent
# --------------------------------------------------------------------------


_GRAPH48 = Path("/private/tmp/cocktail-mate-sensory-artifacts-602-v1/graph48.csv")
_PUBLISHED = Path("/private/tmp/cocktail-mate-s2-rankscaled-602-v1/coordinates.csv")
_PUBLISHED_MEAN_RECALL_AT_5 = 0.4308970099667774


@pytest.mark.skipif(
    not (_GRAPH48.exists() and _PUBLISHED.exists()),
    reason="602-node artifacts are not present on this machine",
)
def test_published_602_layout_reproduces_its_reported_metrics():
    ids, vectors = load_graph48(_GRAPH48)
    assert len(ids) == 602
    cosines = exact_cosine_matrix(vectors)
    ranks, order = similarity_rank_matrix(cosines, ids)

    with _PUBLISHED.open(newline="") as handle:
        published = {
            row["cocktail_id"]: (float(row["x"]), float(row["y"]), float(row["z"]))
            for row in csv.DictReader(handle)
        }
    coords = np.asarray([published[node_id] for node_id in ids], dtype=np.float64)

    assert unit_norm_max_error(coords) <= 1e-12
    _, _, coord_order = coordinate_angles_and_ranks(coords, ids)
    metrics = recall_metrics(order[:, :TOP_K], coord_order[:, :TOP_K])
    assert metrics["mean_recall_at_5"] == pytest.approx(
        _PUBLISHED_MEAN_RECALL_AT_5, abs=1e-12
    )

    contract = check_rescaling_contract(
        rescale_cosine_zcdf_band(ranks, cosines), ranks, cosines
    )
    assert contract["monotone"]
    assert contract["top_k_inside_cap"]
    assert contract["rest_outside_cap"]
    assert contract["top_k_cap_radians"] == pytest.approx(0.1827, abs=5e-5)
