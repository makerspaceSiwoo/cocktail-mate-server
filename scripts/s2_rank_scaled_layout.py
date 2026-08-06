"""Rank-rescaled S² layout for the 602-cocktail Graph48 sensory embedding.

Offline numerical experiment. Reads read-only artifacts from disk, writes a new
artifact directory. No database access, no ORM, no network calls.

Why rank rescaling: the exact 48D cosines of the true top-5 neighbours are
compressed into ``[0.968, 0.974]``, so ``acos(cosine)`` maps rank-5 to about
0.23 rad. A spherical cap of radius 0.23 rad holds a median of 15.5 of the 602
nodes under uniform density, which makes "exactly the true top-5 are nearest"
geometrically impossible. Rank-based rescaling instead maps rank ``r`` onto the
spherical *area* quantile that a rank-``r`` neighbour would occupy under uniform
density, so the rank-5 target lands exactly on the 5-slot cap radius
``acos(1 - 2*5/(N-1)) = 0.18268 rad``.

Usage::

    python scripts/s2_rank_scaled_layout.py --output-dir /private/tmp/...
    python scripts/s2_rank_scaled_layout.py --compare-all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

DEFAULT_ARTIFACT_DIR = Path("/private/tmp/cocktail-mate-sensory-artifacts-602-v1")
DEFAULT_OUTPUT_DIR = Path("/private/tmp/cocktail-mate-s2-rankscaled-602-v1")
DEFAULT_SEED = 20260806
TOP_K = 5
BOTTOM_DECILE_FRACTION = 0.10


def top_k_cap_radians(node_count: int, k: int = TOP_K) -> float:
    """Cap radius whose uniform-density area holds exactly ``k`` of the nodes.

    A spherical cap of angular radius ``theta`` covers ``(1 - cos theta) / 2`` of
    the sphere. Requiring that fraction of the ``node_count - 1`` non-self nodes
    to equal ``k`` gives ``1 - cos theta = 2 * k / (node_count - 1)``.
    """

    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    area = 2.0 * k / (node_count - 1)
    if not 0.0 < area <= 2.0:
        raise ValueError("k is out of range for node_count")
    return math.acos(max(-1.0, min(1.0, 1.0 - area)))


# ---------------------------------------------------------------------------
# deterministic ordering helpers
# ---------------------------------------------------------------------------


def node_id_sort_key(node_id: str) -> tuple[int, int, str]:
    """Tie-break key matching ``app.spherical_graph.pipeline`` node ordering."""

    try:
        value = int(node_id)
    except ValueError:
        return (1, 0, node_id)
    if value > 0 and str(value) == node_id:
        return (0, value, "")
    return (1, 0, node_id)


def _tiebreak_positions(node_ids: Sequence[str]) -> np.ndarray:
    order = sorted(range(len(node_ids)), key=lambda j: node_id_sort_key(node_ids[j]))
    positions = np.empty(len(node_ids), dtype=np.int64)
    positions[np.asarray(order, dtype=np.int64)] = np.arange(len(node_ids))
    return positions


def similarity_rank_matrix(
    similarity: np.ndarray, node_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Rank every node from every source by descending similarity.

    Returns ``(ranks, order)`` where ``ranks[i, j]`` is the 1-based rank of ``j``
    seen from ``i`` (``ranks[i, i] == 0``) and ``order[i]`` lists the ``N - 1``
    non-self nodes from nearest to farthest. Ties break on ``node_id_sort_key``.
    """

    n = similarity.shape[0]
    if similarity.shape != (n, n):
        raise ValueError("similarity must be square")
    if len(node_ids) != n:
        raise ValueError("node_ids length must match similarity")
    tiebreak = _tiebreak_positions(node_ids)
    ranks = np.zeros((n, n), dtype=np.int64)
    order = np.zeros((n, n - 1), dtype=np.int64)
    all_idx = np.arange(n, dtype=np.int64)
    for i in range(n):
        others = all_idx[all_idx != i]
        sorted_others = others[np.lexsort((tiebreak[others], -similarity[i, others]))]
        order[i] = sorted_others
        ranks[i, sorted_others] = np.arange(1, n)
    return ranks, order


# ---------------------------------------------------------------------------
# rank rescaling candidates
# ---------------------------------------------------------------------------


def _areas_to_target_cosines(areas: np.ndarray) -> np.ndarray:
    """Convert ``1 - cos(theta)`` values into clamped target cosines."""

    return np.clip(1.0 - areas, -1.0, 1.0)


def rescale_rank_uniform_area(
    ranks: np.ndarray, similarity: np.ndarray | None = None
) -> np.ndarray:
    """Candidate 1 — map rank ``r`` onto the uniform spherical area quantile.

    ``1 - cos theta = 2 * r / (N - 1)``. Rank 5 lands exactly on the 5-slot cap
    radius and rank ``N - 1`` lands on ``pi``, so the whole ``[0, pi]`` range is
    used. Depends on ranks only; cosine magnitudes are discarded.
    """

    del similarity
    n = ranks.shape[0]
    areas = 2.0 * ranks.astype(np.float64) / (n - 1)
    np.fill_diagonal(areas, 0.0)
    return _areas_to_target_cosines(areas)


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def enforce_top_k_band(
    areas: np.ndarray, ranks: np.ndarray, k: int = TOP_K
) -> np.ndarray:
    """Monotonically squeeze raw area values into the two required bands.

    Ranks ``1..k`` are rescaled onto ``(0, area_cap]`` and ranks ``k+1..N-1``
    onto ``(area_cap, 2]``, preserving the relative spacing of the raw values
    inside each band. ``area_cap = 2 * k / (N - 1)``.
    """

    n = ranks.shape[0]
    area_cap = 2.0 * k / (n - 1)
    out = np.zeros_like(areas, dtype=np.float64)
    # Strictly positive offset so rank k+1 lands strictly outside the cap.
    offset = 1.0 / (n - k - 1)
    for i in range(n):
        row_rank = ranks[i]
        top = (row_rank >= 1) & (row_rank <= k)
        rest = row_rank > k
        area_k = float(areas[i, row_rank == k][0])
        out[i, top] = area_cap * areas[i, top] / max(area_k, 1e-15)
        area_first = float(areas[i, row_rank == k + 1][0])
        area_last = float(areas[i, row_rank == n - 1][0])
        span = max(area_last - area_first, 1e-15)
        t = (areas[i, rest] - area_first) / span
        out[i, rest] = area_cap + (2.0 - area_cap) * (t + offset) / (1.0 + offset)
    np.fill_diagonal(out, 0.0)
    return out


def rescale_cosine_zcdf_band(ranks: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    """Candidate 2 — per-source z-normalised cosine flattened by a normal CDF.

    ``z = (cos - mean_i) / std_i`` is turned into an estimated tail fraction
    ``q = 1 - Phi(z)`` and read as a spherical area fraction. The raw values are
    then squeezed into the two required bands, so the candidate keeps the
    *relative cosine gaps* that candidate 1 throws away.
    """

    work = similarity.astype(np.float64).copy()
    np.fill_diagonal(work, np.nan)
    mean = np.nanmean(work, axis=1, keepdims=True)
    std = np.nanstd(work, axis=1, keepdims=True)
    z = (work - mean) / np.maximum(std, 1e-15)
    tail = 1.0 - _standard_normal_cdf(np.nan_to_num(z, nan=0.0))
    areas = enforce_top_k_band(2.0 * tail, ranks)
    return _areas_to_target_cosines(areas)


def rescale_rank_area_margin(
    ranks: np.ndarray,
    similarity: np.ndarray | None = None,
    *,
    top_shrink: float = 0.5,
    rank_gap: float = 5.0,
) -> np.ndarray:
    """Candidate 3 — candidate 1 plus an explicit angular gap after rank ``k``.

    Ranks ``1..5`` are pulled inside a cap shrunk by ``top_shrink`` and ranks
    ``6..N-1`` are spread linearly in area from ``2 * (6 + rank_gap) / (N - 1)``
    up to ``2`` (``theta = pi``). The gap is a deliberate exclusivity margin.
    """

    del similarity
    n = ranks.shape[0]
    r = ranks.astype(np.float64)
    top_area = 2.0 * r / (n - 1) * top_shrink
    first_rest = 2.0 * (TOP_K + 1.0 + rank_gap) / (n - 1)
    rest_area = first_rest + (2.0 - first_rest) * (r - (TOP_K + 1.0)) / (
        n - 1 - (TOP_K + 1.0)
    )
    areas = np.where(r <= TOP_K, top_area, rest_area)
    np.fill_diagonal(areas, 0.0)
    return _areas_to_target_cosines(areas)


def rescale_raw_acos(ranks: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    """Reference only — the baseline policy of using the 48D cosine directly.

    This is *not* a valid candidate: it violates the top-5 cap band because the
    true rank-5 cosines sit near 0.97 (about 0.23 rad). It exists so ablations
    can isolate the effect of rescaling under an otherwise identical optimiser.
    """

    del ranks
    return np.clip(similarity.astype(np.float64), -1.0, 1.0)


RescaleFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

RESCALERS: dict[str, RescaleFn] = {
    "rank_uniform_area": rescale_rank_uniform_area,
    "cosine_zcdf_band": rescale_cosine_zcdf_band,
    "rank_area_margin": rescale_rank_area_margin,
    "raw_acos": rescale_raw_acos,
}

#: Candidates that must satisfy the rescaling contract; ``raw_acos`` is excluded.
CONTRACT_RESCALERS: tuple[str, ...] = (
    "rank_uniform_area",
    "cosine_zcdf_band",
    "rank_area_margin",
)

RESCALING_POLICY: dict[str, str] = {
    "rank_uniform_area": "1 - cos(theta) = 2 * rank / (N - 1)",
    "cosine_zcdf_band": (
        "1 - cos(theta) = 2 * (1 - Phi(z)) with per-source z-normalised cosine, "
        "monotonically squeezed into the rank<=5 and rank>5 angular bands"
    ),
    "rank_area_margin": (
        "1 - cos(theta) = 2 * rank / (N - 1) * 0.5 for rank <= 5, then a linear "
        "area ramp from 2 * 11 / (N - 1) to 2 for rank >= 6"
    ),
    "raw_acos": "acos(clamped 48D cosine) — reference only, violates the cap band",
}


def check_rescaling_contract(
    target_cosines: np.ndarray,
    ranks: np.ndarray,
    similarity: np.ndarray,
    *,
    k: int = TOP_K,
) -> dict[str, object]:
    """Verify monotonicity, the top-k band and full ``[0, pi]`` range usage."""

    n = ranks.shape[0]
    cap = top_k_cap_radians(n, k)
    angles = np.arccos(np.clip(target_cosines, -1.0, 1.0))
    top = (ranks >= 1) & (ranks <= k)
    rest = ranks > k
    monotone_violations = 0
    for i in range(n):
        row_rank = ranks[i]
        keep = row_rank >= 1
        idx = np.argsort(row_rank[keep])
        ordered = angles[i, keep][idx]
        monotone_violations += int(np.sum(np.diff(ordered) < -1e-12))
        ordered_cos = similarity[i, keep][idx]
        monotone_violations += int(np.sum(np.diff(ordered_cos) > 1e-12))
    return {
        "top_k_cap_radians": cap,
        "max_target_angle_rank_le_k": float(np.max(angles[top])),
        "min_target_angle_rank_gt_k": float(np.min(angles[rest])),
        "max_target_angle": float(np.max(angles[rest])),
        "min_target_angle": float(np.min(angles[top])),
        "top_k_inside_cap": bool(np.max(angles[top]) <= cap + 1e-12),
        "rest_outside_cap": bool(np.min(angles[rest]) > cap),
        "monotone_violations": monotone_violations,
        "monotone": monotone_violations == 0,
    }


# ---------------------------------------------------------------------------
# objective and optimiser
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    """Deterministic optimiser configuration."""

    seed: int = DEFAULT_SEED
    multistart_count: int = 8
    stress_weight: float = 1.0
    neighbourhood_weight: float = 1.0
    hinge_weight: float = 0.0
    hinge_margin: float = 0.20
    temperature: float = 0.01
    temperature_schedule: tuple[float, ...] = (20.0, 5.0, 1.0)
    max_iterations: int = 3000
    ftol: float = 1e-14
    gtol: float = 1e-10
    rank_weight_power: float = 1.0


def rank_stress_weights(ranks: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Stress weights ``1 / rank**power``; the diagonal is zeroed."""

    weights = 1.0 / np.power(np.maximum(ranks, 1).astype(np.float64), power)
    np.fill_diagonal(weights, 0.0)
    return weights


def _make_objective(
    target_cosines: np.ndarray,
    weights: np.ndarray,
    top_k_index: np.ndarray,
    config: LayoutConfig,
    temperature: float,
):
    n = target_cosines.shape[0]
    rows = np.arange(n)[:, None]
    excluded = np.zeros((n, n), dtype=bool)
    excluded[rows, top_k_index] = True
    excluded[np.arange(n), np.arange(n)] = True
    self_mask = np.eye(n, dtype=bool)
    k = top_k_index.shape[1]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        raw = flat.reshape(n, 3)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        coords = raw / norms
        dots = coords @ coords.T
        loss = 0.0
        grad_dots = np.zeros((n, n), dtype=np.float64)

        if config.stress_weight:
            residual = dots - target_cosines
            np.fill_diagonal(residual, 0.0)
            weighted = weights * residual
            loss += config.stress_weight * float(np.sum(weighted * residual))
            grad_dots += 2.0 * config.stress_weight * weighted

        if config.hinge_weight:
            near = np.take_along_axis(dots, top_k_index, axis=1)
            slack = config.hinge_margin + dots[:, None, :] - near[:, :, None]
            np.maximum(slack, 0.0, out=slack)
            slack *= ~excluded[:, None, :]
            loss += config.hinge_weight * float(np.sum(slack * slack))
            per_far = slack.sum(axis=1)
            per_near = slack.sum(axis=2)
            hinge_grad = 2.0 * config.hinge_weight * per_far
            np.add.at(
                hinge_grad, (rows, top_k_index), -2.0 * config.hinge_weight * per_near
            )
            grad_dots += hinge_grad

        if config.neighbourhood_weight:
            scaled = np.where(self_mask, -np.inf, dots / temperature)
            shift = scaled.max(axis=1, keepdims=True)
            exponent = np.exp(scaled - shift)
            partition = exponent.sum(axis=1, keepdims=True)
            log_prob = (scaled - shift) - np.log(partition)
            picked = np.take_along_axis(log_prob, top_k_index, axis=1)
            loss += config.neighbourhood_weight * float(-np.sum(picked))
            prob = exponent / partition
            scale = config.neighbourhood_weight / temperature
            nca_grad = scale * k * prob
            np.add.at(nca_grad, (rows, top_k_index), -scale)
            np.fill_diagonal(nca_grad, 0.0)
            grad_dots += nca_grad

        grad_coords = (grad_dots + grad_dots.T) @ coords
        radial = np.sum(grad_coords * coords, axis=1, keepdims=True)
        grad_raw = (grad_coords - radial * coords) / norms
        return loss, grad_raw.ravel()

    return objective


def spectral_initialisation(target_cosines: np.ndarray) -> np.ndarray:
    """Deterministic start from the top-3 eigenvectors of the target Gram."""

    gram = 0.5 * (target_cosines + target_cosines.T)
    np.fill_diagonal(gram, 1.0)
    values, vectors = np.linalg.eigh(gram)
    coords = vectors[:, -3:] * np.sqrt(np.maximum(values[-3:], 1e-9))
    return normalise_rows(coords)


def normalise_rows(coords: np.ndarray) -> np.ndarray:
    """Project rows onto the unit sphere; zero rows fall back to the pole."""

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    out = coords / safe
    degenerate = (norms < 1e-12).ravel()
    if np.any(degenerate):
        out[degenerate] = np.array([0.0, 0.0, 1.0])
    return out


@dataclass
class LayoutResult:
    coordinates: np.ndarray
    start_objectives: list[float] = field(default_factory=list)
    selected_start: int = 0
    iterations: list[int] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def optimise_layout(
    target_cosines: np.ndarray,
    weights: np.ndarray,
    top_k_index: np.ndarray,
    config: LayoutConfig,
) -> LayoutResult:
    """Multistart L-BFGS-B with a deterministic temperature annealing schedule."""

    started = time.perf_counter()
    n = target_cosines.shape[0]
    stages = [
        _make_objective(
            target_cosines, weights, top_k_index, config, config.temperature * factor
        )
        for factor in config.temperature_schedule
    ]
    starts = [spectral_initialisation(target_cosines)]
    seeds = [config.seed]
    for offset in range(1, config.multistart_count):
        seed = config.seed + offset
        rng = np.random.default_rng(seed)
        starts.append(normalise_rows(rng.standard_normal((n, 3))))
        seeds.append(seed)

    options = {
        "maxiter": config.max_iterations,
        "maxfun": config.max_iterations * 2,
        "ftol": config.ftol,
        "gtol": config.gtol,
    }
    result = LayoutResult(coordinates=np.zeros((n, 3)))
    best_value = math.inf
    for index, start in enumerate(starts):
        flat = start.ravel().copy()
        iterations = 0
        value = math.inf
        for objective in stages:
            solved = minimize(
                objective, flat, jac=True, method="L-BFGS-B", options=options
            )
            flat = solved.x
            iterations += int(solved.nit)
            value = float(solved.fun)
        result.start_objectives.append(value)
        result.iterations.append(iterations)
        if value < best_value:
            best_value = value
            result.selected_start = index
            result.coordinates = normalise_rows(flat.reshape(n, 3))
    result.seeds = seeds
    result.elapsed_seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def unit_norm_max_error(coords: np.ndarray) -> float:
    """Largest deviation of any row norm from 1."""

    return float(np.max(np.abs(np.linalg.norm(coords, axis=1) - 1.0)))


def recall_metrics(
    true_top_k: np.ndarray, coordinate_top_k: np.ndarray
) -> dict[str, float]:
    """Exact definitions from the task brief.

    ``mean_recall_at_5 = mean_i |T_i & C_i| / 5``,
    ``full_recovery_rate = |{i : T_i subset C_i}| / N``,
    ``hit_rate_at_5 = |{i : |T_i & C_i| >= 1}| / N``.
    """

    if true_top_k.shape != coordinate_top_k.shape:
        raise ValueError("true and coordinate top-k must have the same shape")
    n, k = true_top_k.shape
    if n == 0:
        raise ValueError("need at least one source")
    hits = np.array(
        [
            len(set(true_top_k[i].tolist()) & set(coordinate_top_k[i].tolist()))
            for i in range(n)
        ],
        dtype=np.int64,
    )
    return {
        "mean_recall_at_5": float(np.mean(hits / k)),
        "full_recovery_rate": float(np.mean(hits == k)),
        "hit_rate_at_5": float(np.mean(hits >= 1)),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def bottom_decile_false_close_count(
    coordinate_angles: np.ndarray,
    cosine_order: np.ndarray,
    true_top_k: np.ndarray,
    fraction: float = BOTTOM_DECILE_FRACTION,
) -> int:
    """Count cosine bottom-decile non-neighbours placed inside the top-k radius.

    For each source the bottom ``fraction`` of the ``N - 1`` non-self nodes by
    48D cosine is taken; every one of those placed strictly closer than the
    source's farthest true top-k coordinate neighbour counts once.
    """

    n = coordinate_angles.shape[0]
    per_source = int(math.floor(fraction * (n - 1)))
    total = 0
    for i in range(n):
        farthest = float(np.max(coordinate_angles[i, true_top_k[i]]))
        worst = cosine_order[i, -per_source:] if per_source else cosine_order[i, :0]
        total += int(np.sum(coordinate_angles[i, worst] < farthest))
    return total


def coordinate_angles_and_ranks(
    coords: np.ndarray, node_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(angles, ranks, order)`` in coordinate space."""

    dots = np.clip(coords @ coords.T, -1.0, 1.0)
    angles = np.arccos(dots)
    np.fill_diagonal(angles, 0.0)
    ranks, order = similarity_rank_matrix(dots, node_ids)
    return angles, ranks, order


def evaluate_layout(
    coords: np.ndarray,
    node_ids: Sequence[str],
    cosine_ranks: np.ndarray,
    cosine_order: np.ndarray,
    union_pairs: np.ndarray,
    union_cosines: np.ndarray,
    target_cosines: np.ndarray,
) -> dict[str, object]:
    """All brief §D metrics for one coordinate set."""

    n = coords.shape[0]
    true_top_k = cosine_order[:, :TOP_K]
    angles, coord_ranks, coord_order = coordinate_angles_and_ranks(coords, node_ids)
    coord_top_k = coord_order[:, :TOP_K]

    metrics: dict[str, object] = dict(recall_metrics(true_top_k, coord_top_k))

    true_coord_ranks = np.take_along_axis(coord_ranks, true_top_k, axis=1)
    for k in (10, 20, 50):
        metrics[f"recall_at_{k}"] = float(np.mean(true_coord_ranks <= k))
    metrics["true_top5_coordinate_rank"] = distribution_summary(
        true_coord_ranks.ravel()
    )

    intruder_ranks: list[int] = []
    intruder_sources = 0
    for i in range(n):
        truth = set(true_top_k[i].tolist())
        found = [j for j in coord_top_k[i].tolist() if j not in truth]
        if found:
            intruder_sources += 1
        intruder_ranks.extend(int(cosine_ranks[i, j]) for j in found)
    intruder_array = np.asarray(intruder_ranks or [0], dtype=np.float64)
    metrics["intruder_cosine_rank"] = {
        "count": len(intruder_ranks),
        "median": float(np.median(intruder_array)),
        "p90": float(np.percentile(intruder_array, 90)),
        "max": float(np.max(intruder_array)),
        "sources_with_intruder": intruder_sources,
        "slot_share": float(len(intruder_ranks) / (n * TOP_K)),
    }

    left = union_pairs[:, 0]
    right = union_pairs[:, 1]
    observed = angles[left, right]
    original_target = np.arccos(np.clip(union_cosines, -1.0, 1.0))
    rescaled_pair = 0.5 * (
        np.arccos(np.clip(target_cosines[left, right], -1.0, 1.0))
        + np.arccos(np.clip(target_cosines[right, left], -1.0, 1.0))
    )
    metrics["union_edge_rmse_radians_original_acos"] = float(
        np.sqrt(np.mean((observed - original_target) ** 2))
    )
    metrics["union_edge_rmse_radians_rescaled_target"] = float(
        np.sqrt(np.mean((observed - rescaled_pair) ** 2))
    )
    metrics["unit_norm_max_error"] = unit_norm_max_error(coords)
    metrics["bottom_decile_false_close_count"] = bottom_decile_false_close_count(
        angles, cosine_order, true_top_k
    )
    offdiag = angles[~np.eye(n, dtype=bool)]
    metrics["coordinate_angle_spread"] = {
        "min": float(np.min(offdiag)),
        "median": float(np.median(offdiag)),
        "max": float(np.max(offdiag)),
    }
    return metrics


# ---------------------------------------------------------------------------
# artifact I/O
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph48(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 48:
            raise ValueError("graph48 header is too short")
        for row in reader:
            ids.append(row[0])
            rows.append([float(value) for value in row[-48:]])
    return ids, np.asarray(rows, dtype=np.float64)


def load_union_edges(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def exact_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosines = np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(cosines, 1.0)
    return cosines


def write_coordinates_csv(path: Path, node_ids: Sequence[str], coords: np.ndarray):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cocktail_id", "x", "y", "z"])
        for node_id, row in zip(node_ids, coords):
            writer.writerow(
                [node_id, repr(float(row[0])), repr(float(row[1])), repr(float(row[2]))]
            )


def coordinate_digest(node_ids: Sequence[str], coords: np.ndarray) -> str:
    """SHA-256 over ``node_id:x:y:z`` lines in the graph48 row order."""

    digest = hashlib.sha256()
    for node_id, row in zip(node_ids, coords):
        digest.update(
            f"{node_id}:{float(row[0])!r}:{float(row[1])!r}:{float(row[2])!r}\n".encode()
        )
    return digest.hexdigest()


def build_public_json(
    baseline: dict,
    node_ids: Sequence[str],
    coords: np.ndarray,
    layout_report: dict,
) -> dict:
    """Reproduce the baseline public schema with new coordinates."""

    index = {node_id: i for i, node_id in enumerate(node_ids)}
    graph = dict(baseline["graph"])
    nodes = []
    for node in baseline["graph"]["nodes"]:
        node_id = node["node_id"]
        if node_id not in index:
            raise ValueError(f"unknown node in baseline json: {node_id}")
        if node.get("node_kind") != "cocktail":
            raise ValueError("baseline json contains a non-cocktail node")
        row = coords[index[node_id]]
        updated = dict(node)
        updated["x"] = float(row[0])
        updated["y"] = float(row[1])
        updated["z"] = float(row[2])
        nodes.append(updated)
    graph["nodes"] = nodes
    graph["layout_report"] = layout_report
    return {
        "graph": graph,
        "provenance": dict(baseline["provenance"]),
        "public_hub_edge_count": 0,
        "public_hub_node_count": 0,
        "schema_version": baseline["schema_version"],
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_candidate(
    name: str,
    ids: Sequence[str],
    cosines: np.ndarray,
    ranks: np.ndarray,
    order: np.ndarray,
    union_pairs: np.ndarray,
    union_cosines: np.ndarray,
    config: LayoutConfig,
    *,
    enforce_contract: bool = True,
) -> tuple[np.ndarray, np.ndarray, LayoutResult, dict, dict]:
    target_cosines = RESCALERS[name](ranks, cosines)
    contract = check_rescaling_contract(target_cosines, ranks, cosines)
    if enforce_contract:
        if not contract["monotone"]:
            raise ValueError(f"{name} rescaling is not monotone")
        if not contract["top_k_inside_cap"] or not contract["rest_outside_cap"]:
            raise ValueError(f"{name} rescaling violates the top-5 cap band")
    weights = rank_stress_weights(ranks, config.rank_weight_power)
    result = optimise_layout(target_cosines, weights, order[:, :TOP_K], config)
    metrics = evaluate_layout(
        result.coordinates,
        ids,
        ranks,
        order,
        union_pairs,
        union_cosines,
        target_cosines,
    )
    return result.coordinates, target_cosines, result, contract, metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate", default="rank_uniform_area", choices=sorted(RESCALERS)
    )
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stress-weight", type=float, default=1.0)
    parser.add_argument("--neighbourhood-weight", type=float, default=1.0)
    parser.add_argument("--hinge-weight", type=float, default=0.0)
    parser.add_argument("--hinge-margin", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir: Path = args.artifact_dir
    graph48_path = artifact_dir / "graph48.csv"
    union_path = artifact_dir / "graph48-union-edges.csv"
    directed_path = artifact_dir / "graph48-directed-top5.csv"
    baseline_path = artifact_dir / "spherical-graph-public.json"

    ids, vectors = load_graph48(graph48_path)
    n = len(ids)
    cosines = exact_cosine_matrix(vectors)
    ranks, order = similarity_rank_matrix(cosines, ids)
    index = {node_id: i for i, node_id in enumerate(ids)}

    union_rows = load_union_edges(union_path)
    union_pairs = np.asarray(
        [[index[row["a_id"]], index[row["b_id"]]] for row in union_rows],
        dtype=np.int64,
    )
    union_cosines = np.asarray(
        [float(row["cosine"]) for row in union_rows], dtype=np.float64
    )

    config = LayoutConfig(
        seed=args.seed,
        multistart_count=args.starts,
        stress_weight=args.stress_weight,
        neighbourhood_weight=args.neighbourhood_weight,
        hinge_weight=args.hinge_weight,
        hinge_margin=args.hinge_margin,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
    )

    names = list(CONTRACT_RESCALERS) if args.compare_all else [args.candidate]
    runs: dict[str, dict] = {}
    artifacts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        coords, target_cosines, result, contract, metrics = run_candidate(
            name,
            ids,
            cosines,
            ranks,
            order,
            union_pairs,
            union_cosines,
            config,
            enforce_contract=name in CONTRACT_RESCALERS,
        )
        artifacts[name] = (coords, target_cosines)
        runs[name] = {
            "rescaling": name,
            "contract": contract,
            "metrics": metrics,
            "multistart_objectives": result.start_objectives,
            "multistart_seeds": result.seeds,
            "multistart_iterations": result.iterations,
            "selected_start": result.selected_start,
            "elapsed_seconds": result.elapsed_seconds,
        }
        print(
            f"{name}: recall@5={metrics['mean_recall_at_5']:.4f} "
            f"full={metrics['full_recovery_rate']:.4f} "
            f"hit={metrics['hit_rate_at_5']:.4f} "
            f"({result.elapsed_seconds:.1f}s)"
        )

    # Selection rule fixed up front: highest mean Recall@5, the task objective.
    chosen = max(names, key=lambda name: runs[name]["metrics"]["mean_recall_at_5"])
    coords, target_cosines = artifacts[chosen]
    if not args.no_write and chosen not in CONTRACT_RESCALERS:
        raise ValueError(
            f"{chosen} is a reference-only rescaling; rerun with --no-write"
        )

    with baseline_path.open() as handle:
        baseline = json.load(handle)
    baseline_ids = [node["node_id"] for node in baseline["graph"]["nodes"]]
    baseline_coords = np.asarray(
        [
            [
                baseline["graph"]["nodes"][baseline_ids.index(node_id)]["x"],
                baseline["graph"]["nodes"][baseline_ids.index(node_id)]["y"],
                baseline["graph"]["nodes"][baseline_ids.index(node_id)]["z"],
            ]
            for node_id in ids
        ],
        dtype=np.float64,
    )
    baseline_metrics = evaluate_layout(
        baseline_coords,
        ids,
        ranks,
        order,
        union_pairs,
        union_cosines,
        target_cosines,
    )

    directed_agreement = _directed_top5_agreement(directed_path, ids, order, index)

    if args.no_write:
        payload = {
            "candidates": {name: runs[name]["metrics"] for name in names},
            "baseline": baseline_metrics,
        }
        print(json.dumps(payload, indent=2, default=str, sort_keys=True))
        return 0

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = out_dir / "coordinates.csv"
    write_coordinates_csv(coordinates_path, ids, coords)

    non_edge_pairs = n * (n - 1) // 2 - len(union_rows)
    layout_report = {
        "algorithm": "rank_rescaled_spherical_nca_v1",
        "rescaling": chosen,
        "edge_target_policy": (
            f"rank-based rescaled angular target ({chosen}): {RESCALING_POLICY[chosen]}"
        ),
        "negative_sampling_policy": (
            "complete deterministic enumeration of all C(602, 2) = 180901 "
            "unordered pairs; repulsion strength derives from the pair's exact "
            "48D cosine rank, no uniform margin"
        ),
        "sampled_nonedge_count": non_edge_pairs,
        "seed": config.seed,
        "multistart_count": config.multistart_count,
        "multistart_seeds": runs[chosen]["multistart_seeds"],
        "candidate_objectives": runs[chosen]["multistart_objectives"],
        "multistart_iterations": runs[chosen]["multistart_iterations"],
        "selected_start": runs[chosen]["selected_start"],
        "iterations": runs[chosen]["multistart_iterations"][
            runs[chosen]["selected_start"]
        ],
        "temperature": config.temperature,
        "temperature_schedule": [
            config.temperature * factor for factor in config.temperature_schedule
        ],
        "stress_weight": config.stress_weight,
        "neighbourhood_weight": config.neighbourhood_weight,
        "hinge_weight": config.hinge_weight,
        "convergence": {
            "ftol": config.ftol,
            "gtol": config.gtol,
            "max_iterations": config.max_iterations,
        },
        "coordinate_sha256": coordinate_digest(ids, coords),
        **{key: value for key, value in runs[chosen]["metrics"].items()},
    }

    public = build_public_json(baseline, ids, coords, layout_report)
    public_path = out_dir / "spherical-graph-public.json"
    with public_path.open("w") as handle:
        json.dump(public, handle, indent=2, sort_keys=True)

    metrics_payload = {
        "node_count": n,
        "top5_cap_radians": top_k_cap_radians(n),
        "selected_rescaling": chosen,
        "candidates": runs,
        "baseline": {
            "source": str(baseline_path),
            "metrics": baseline_metrics,
        },
        "directed_top5_csv_agreement": directed_agreement,
        "selection_rule": (
            "highest mean Recall@5; ties break on CONTRACT_RESCALERS order"
        ),
        "rescaling_policy": {
            name: RESCALING_POLICY[name] for name in CONTRACT_RESCALERS
        },
        "config": {
            "seed": config.seed,
            "multistart_count": config.multistart_count,
            "stress_weight": config.stress_weight,
            "neighbourhood_weight": config.neighbourhood_weight,
            "hinge_weight": config.hinge_weight,
            "hinge_margin": config.hinge_margin,
            "temperature_final": config.temperature,
            "temperature_schedule_factors": list(config.temperature_schedule),
            "temperature_schedule_absolute": [
                config.temperature * factor for factor in config.temperature_schedule
            ],
            "max_iterations": config.max_iterations,
            "ftol": config.ftol,
            "gtol": config.gtol,
            "rank_weight_power": config.rank_weight_power,
        },
        "nonedge_policy": {
            "policy": "complete deterministic enumeration",
            "unordered_pairs_total": n * (n - 1) // 2,
            "union_edge_pairs": len(union_rows),
            "nonedge_pairs": non_edge_pairs,
        },
        "input_file_sha256": {
            "graph48.csv": sha256_file(graph48_path),
            "graph48-union-edges.csv": sha256_file(union_path),
            "graph48-directed-top5.csv": sha256_file(directed_path),
            "spherical-graph-public.json": sha256_file(baseline_path),
        },
        "output_file_sha256": {
            "coordinates.csv": sha256_file(coordinates_path),
            "spherical-graph-public.json": sha256_file(public_path),
        },
        "coordinate_sha256": coordinate_digest(ids, coords),
        "database_reads": 0,
        "database_writes": 0,
        "network_calls": 0,
    }
    metrics_path = out_dir / "layout-metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)

    print(f"wrote {coordinates_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {public_path}")
    return 0


def _directed_top5_agreement(
    directed_path: Path,
    ids: Sequence[str],
    order: np.ndarray,
    index: dict[str, int],
) -> dict[str, object]:
    """Cross-check the recomputed top-5 against the published directed CSV."""

    published: dict[str, set[int]] = {}
    with directed_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["rank"]) <= TOP_K:
                published.setdefault(row["source_id"], set()).add(
                    index[row["target_id"]]
                )
    matches = sum(
        1
        for i, node_id in enumerate(ids)
        if published.get(node_id, set()) == set(order[i, :TOP_K].tolist())
    )
    return {"sources": len(ids), "identical_top5_sets": matches}


if __name__ == "__main__":
    raise SystemExit(main())
