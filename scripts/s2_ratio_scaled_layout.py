"""Globally stretched S² layouts for the 602-cocktail Graph48 sensory embedding.

Offline numerical experiment. Reads read-only artifacts from disk and writes new
artifact directories. No database access, no ORM, no network calls.

Why stretch at all: the exact 48D cosines span ``[0.101, 0.990]``, so ``arccos``
maps every pair into ``[0.139, 1.470]`` rad. A layout that reads those values as
absolute angular distances can never use more than half of the sphere, which is
exactly what the rendered baseline showed.

Two target modes are implemented.

``linear`` — the ratio-preserving global rescale::

    theta_target(i, j) = k * angle(i, j),  one single scalar k

Every pair angle ratio is preserved exactly, and ``k`` is picked from three
candidates (max-antipodal, median-match, KS-optimal).

``convex`` — the convex monotone stretch requested afterwards, which pushes far
pairs apart much harder than near pairs::

    f(theta) = a * theta + b * theta ** gamma,
    b = (pi - a * theta_max) / theta_max ** gamma,  a > 1, gamma > 1

``f`` is strictly increasing (no rank inversions), convex (``f'`` increasing),
expanding everywhere (``f(theta) > theta``, with ``f(theta)/theta -> a > 1`` as
``theta -> 0``) and ends exactly at ``f(theta_max) = pi``. ``a < pi / theta_max``
is required for ``b > 0``; larger ``a`` is rejected because it breaks convexity.

Both modes give targets to all ``C(602, 2) = 180901`` pairs, not just the 2248
union edges. The 48D vectors are never dimensionally reduced — 3D is only the
visualisation manifold.

After the surrogate stress optimiser converges, ``convex`` mode runs a
constrained greedy refinement on the *real* metric (mean Recall@5). The
refinement only accepts moves that keep the cluster separation ratio and the
uniform-sphere KS statistic within a declared tolerance, so the coverage and
separation goals are not traded away for recall. **Every** layout in the
comparison table receives the identical refinement (matched treatment), because
comparing a recall-refined layout against unrefined ones is not like-for-like.

The contract conditions hold for ``f`` by construction, but what the user sees
is the projected 3D coordinates, where near pairs can end up *compressed*. That
is measured directly (``realised_stretch_by_source_decile``,
``nearest_5pct_fraction_compressed``), reported for every layout, and enforced:
a candidate whose realised near-field decile median stretch is below
``--min-near-decile-stretch`` is eliminated before ranking. If no candidate
clears the floors the run reports ``BLOCKED`` and emits a frontier table rather
than forcing a pick.

Usage — these defaults reproduce the published artifacts, no extra flags::

    python scripts/s2_ratio_scaled_layout.py --mode convex
    python scripts/s2_ratio_scaled_layout.py --mode linear
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

#: Durable in-repo artifact copy (``/private/tmp`` is volatile). ``sensory-batch/``
#: is gitignored, so these inputs survive reboots without entering the commit.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "sensory-batch/run-20260806-full602-v1/artifacts"
DEFAULT_RANKSCALED_DIR = Path("/private/tmp/cocktail-mate-s2-rankscaled-602-v1")
DEFAULT_LINEAR_OUTPUT_DIR = Path("/private/tmp/cocktail-mate-s2-ratio-602-v1")
DEFAULT_CONVEX_OUTPUT_DIR = Path("/private/tmp/cocktail-mate-s2-convex-602-v2")
DEFAULT_DURABLE_COPY_DIR = (
    REPO_ROOT / "sensory-batch/run-20260806-full602-v1/s2-convex-v2"
)
DEFAULT_SEED = 20260806
TOP_K = 5
BOTTOM_DECILE_FRACTION = 0.10
#: ``d(arccos)/dx`` diverges at ±1, so dot products are clipped before arccos.
DOT_CLIP = 1.0 - 1e-12
#: Maximum clamped-pair fraction still accepted as "ratio preserving" (linear).
MAX_CLAMPED_FRACTION = 1e-3
#: Fibonacci probe count for the covering-radius estimate.
DEFAULT_PROBE_POINTS = 131072
#: Monte-Carlo replicates for the uniform-sphere reference values.
DEFAULT_UNIFORM_REPLICATES = 32
CELL_LAYOUTS: tuple[tuple[int, int], ...] = ((10, 10), (20, 20))
#: Convex sweep grid. ``a`` is the near-field expansion ratio and the separation
#: lever, so the grid deliberately runs past ``pi / theta_max``; the combinations
#: above the limit are rejected by contract and reported as rejected, never
#: silently dropped.
SWEEP_A: tuple[float, ...] = (1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)
SWEEP_GAMMA: tuple[float, ...] = (1.5, 2.0, 3.0)
#: Deterministic sample size for the coordinate pair-order inversion estimate.
INVERSION_SAMPLES = 2_000_000
#: Deciles used for the realised (coordinate-level) stretch profile.
STRETCH_DECILES = 10
#: "Nearest pairs" fraction for the compression check.
NEAREST_FRACTION = 0.05

#: Pipeline acceptance gates as they stood before the 2026-08-06 revision.
ORIGINAL_ACCEPTANCE_GATES: dict[str, tuple[str, float]] = {
    "mean_recall_at_5": (">=", 0.60),
    "hit_rate_at_5": (">=", 0.90),
    "union_edge_rmse_radians_original_acos": ("<=", 0.40),
    "unit_norm_max_error": ("<=", 1e-12),
    "bottom_decile_false_close_count": ("<=", 0.0),
}
#: Revised hard gates (user-approved 2026-08-06). Both sets are always reported.
REVISED_ACCEPTANCE_GATES: dict[str, tuple[str, float]] = {
    "mean_recall_at_5": (">=", 0.55),
    "hit_rate_at_5": (">=", 0.90),
    "union_edge_rmse_radians_original_acos": ("<=", 0.40),
    "unit_norm_max_error": ("<=", 1e-12),
    "bottom_decile_false_close_count": ("<=", 500.0),
}
#: Non-regression floors against the v1 convex layout. A candidate that misses
#: any of these is eliminated before the Borda ranking — no silent lowering.
NON_REGRESSION_FLOORS: dict[str, tuple[str, float]] = {
    "mean_recall_at_5": (">=", 0.5236),
    "hit_rate_at_5": (">=", 0.9684),
    "bottom_decile_false_close_count": ("<=", 424.0),
    "pair_angle_ks_vs_uniform_sphere": ("<=", 0.2040),
    "covering_radius_radians": ("<=", 0.7100),
    "empty_cells_100": ("<=", 21.0),
    "unit_norm_max_error": ("<=", 1e-12),
    "union_edge_rmse_radians_original_acos": ("<=", 0.40),
    "near_decile_median_stretch": (">=", 1.00),
    "nearest_5pct_fraction_compressed": ("<=", 0.20),
}
#: Separation goals (aspirational, not filters).
SEPARATION_TARGETS: dict[str, tuple[str, float]] = {
    "separation_ratio": (">=", 2.45),
    "silhouette_mean": (">=", 0.28),
    "centroid_angle_min": (">=", 0.80),
}


# ---------------------------------------------------------------------------
# id normalisation and deterministic ordering
# ---------------------------------------------------------------------------


def canonical_id(value: object) -> int:
    """Normalise a cocktail id to ``int``.

    The public JSON stores ``node_id`` as a string while every CSV stores an
    integer. Joining the two without this normalisation silently matches nothing.
    """

    if isinstance(value, bool):
        raise ValueError("cocktail id must not be a bool")
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty cocktail id")
    return int(text)


def _tiebreak_positions(node_ids: Sequence[int]) -> np.ndarray:
    """Positions of each node in ascending-id order, used to break rank ties."""

    order = sorted(range(len(node_ids)), key=lambda j: node_ids[j])
    positions = np.empty(len(node_ids), dtype=np.int64)
    positions[np.asarray(order, dtype=np.int64)] = np.arange(len(node_ids))
    return positions


def similarity_rank_matrix(
    similarity: np.ndarray, node_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Rank every node from every source by descending similarity.

    Returns ``(ranks, order)`` where ``ranks[i, j]`` is the 1-based rank of ``j``
    seen from ``i`` (``ranks[i, i] == 0``) and ``order[i]`` lists the ``N - 1``
    non-self nodes from nearest to farthest. Ties break on ascending id.
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
# angles
# ---------------------------------------------------------------------------


def exact_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Exact pairwise cosine matrix of row-normalised vectors."""

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosines = np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(cosines, 1.0)
    return cosines


def angle_matrix(cosines: np.ndarray) -> np.ndarray:
    """``arccos`` of a cosine matrix with an exactly zero diagonal."""

    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    np.fill_diagonal(angles, 0.0)
    return angles


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """All ``C(n, 2)`` off-diagonal upper-triangle values as a flat array."""

    rows, cols = np.triu_indices(matrix.shape[0], 1)
    return matrix[rows, cols]


def coordinate_angle_matrix(coords: np.ndarray) -> np.ndarray:
    """Great-circle angle matrix of unit coordinates."""

    return angle_matrix(np.clip(coords @ coords.T, -1.0, 1.0))


# ---------------------------------------------------------------------------
# linear mode — one global scalar k
# ---------------------------------------------------------------------------


def uniform_sphere_cdf(theta: np.ndarray) -> np.ndarray:
    """``F(theta) = (1 - cos theta) / 2`` — pair-angle CDF of a uniform sphere."""

    return (1.0 - np.cos(np.clip(theta, 0.0, math.pi))) / 2.0


def ks_statistic_uniform_sphere(sample: np.ndarray) -> float:
    """Two-sided Kolmogorov–Smirnov distance to the uniform-sphere pair CDF."""

    ordered = np.sort(np.asarray(sample, dtype=np.float64))
    count = ordered.size
    if count == 0:
        raise ValueError("sample must be non-empty")
    cdf = uniform_sphere_cdf(ordered)
    steps = np.arange(1, count + 1, dtype=np.float64) / count
    above = float(np.max(steps - cdf))
    below = float(np.max(cdf - (steps - 1.0 / count)))
    return max(above, below)


def scaled_targets(angles: np.ndarray, k: float) -> np.ndarray:
    """``min(k * angle, pi)`` — the global rescale with a hard clamp at ``pi``."""

    return np.minimum(k * np.asarray(angles, dtype=np.float64), math.pi)


def clamp_report(angles: np.ndarray, k: float) -> dict[str, float | int]:
    """How many pairs the ``pi`` clamp touches (the only ratio-breaking event)."""

    values = np.asarray(angles, dtype=np.float64)
    clamped = k * values > math.pi + 1e-15
    return {
        "clamped_pair_count": int(np.sum(clamped)),
        "total_pair_count": int(values.size),
        "clamped_pair_fraction": float(np.mean(clamped)),
        "max_unclamped_target_radians": float(k * np.max(values)),
    }


def k_max_antipodal(pair_angles: np.ndarray) -> float:
    """Candidate 1 — ``k = pi / max_angle``; the widest pair becomes antipodal."""

    return math.pi / float(np.max(pair_angles))


def k_median_match(pair_angles: np.ndarray) -> float:
    """Candidate 2 — ``k = (pi / 2) / median_angle``; medians of the two match."""

    return (math.pi / 2.0) / float(np.median(pair_angles))


def k_ks_optimal(
    pair_angles: np.ndarray,
    *,
    low: float = 0.5,
    high: float = 6.0,
    coarse_step: float = 0.005,
    refinements: int = 3,
    refine_points: int = 201,
) -> tuple[float, float]:
    """Candidate 3 — the ``k`` minimising the KS distance to the uniform sphere.

    Deterministic nested grid search: one coarse sweep followed by
    ``refinements`` bracketed sweeps, each 100x finer than the previous one.
    Returns ``(k, ks)``. A grid search rather than a unimodal solver because the
    KS statistic in ``k`` is only piecewise smooth.
    """

    ordered = np.sort(np.asarray(pair_angles, dtype=np.float64))

    def ks_for(k: float) -> float:
        return ks_statistic_uniform_sphere(scaled_targets(ordered, k))

    grid = np.arange(low, high + 0.5 * coarse_step, coarse_step)
    values = np.asarray([ks_for(float(k)) for k in grid])
    best = float(grid[int(np.argmin(values))])
    best_value = float(np.min(values))
    step = coarse_step
    for _ in range(refinements):
        grid = np.linspace(best - step, best + step, refine_points)
        values = np.asarray([ks_for(float(k)) for k in grid])
        index = int(np.argmin(values))
        best = float(grid[index])
        best_value = float(values[index])
        step = 2.0 * step / (refine_points - 1)
    return best, best_value


def ratio_preservation_error(angles: np.ndarray, targets: np.ndarray) -> float:
    """Largest relative violation of ``target(i,j)/target(p,q) == a(i,j)/a(p,q)``.

    Equivalent to checking that ``target / angle`` is one single constant, which
    is the contract of the linear mode only.
    """

    a = np.asarray(angles, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    keep = a > 0.0
    ratios = t[keep] / a[keep]
    return float(np.max(np.abs(ratios - ratios[0])) / ratios[0])


# ---------------------------------------------------------------------------
# convex mode — monotone convex stretch
# ---------------------------------------------------------------------------


def convex_stretch_coefficient(a: float, gamma: float, theta_max: float) -> float:
    """``b`` of ``f(theta) = a*theta + b*theta**gamma`` with ``f(theta_max)=pi``.

    Raises when the requested near-field stretch ``a`` is too large to leave any
    room for the convex term: ``b > 0`` requires ``a < pi / theta_max``.
    """

    if theta_max <= 0.0:
        raise ValueError("theta_max must be positive")
    if a <= 1.0:
        raise ValueError("a must exceed 1 so that near pairs expand, not shrink")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed 1 for the stretch to be convex")
    limit = math.pi / theta_max
    if a >= limit:
        raise ValueError(
            f"a={a} must be below pi/theta_max={limit:.6f}; otherwise b <= 0 "
            "and the stretch stops being convex"
        )
    return (math.pi - a * theta_max) / theta_max**gamma


def convex_stretch(
    angles: np.ndarray, a: float, gamma: float, theta_max: float
) -> np.ndarray:
    """Monotone convex stretch ``f(theta) = a*theta + b*theta**gamma``."""

    b = convex_stretch_coefficient(a, gamma, theta_max)
    values = np.asarray(angles, dtype=np.float64)
    return np.minimum(a * values + b * np.power(values, gamma), math.pi)


def convex_contract_report(
    a: float, gamma: float, theta_max: float, points: int = 20001
) -> dict[str, object]:
    """Numerically verify the four contract conditions on ``[0, theta_max]``."""

    b = convex_stretch_coefficient(a, gamma, theta_max)
    grid = np.linspace(0.0, theta_max, points)
    values = convex_stretch(grid, a, gamma, theta_max)
    first = np.diff(values)
    second = np.diff(values, 2)
    expansion = values[1:] / grid[1:]
    return {
        "a": a,
        "gamma": gamma,
        "b": b,
        "theta_max": theta_max,
        "a_upper_bound": math.pi / theta_max,
        "monotone_increasing": bool(np.all(first > 0.0)),
        "min_first_difference": float(np.min(first)),
        "convex": bool(np.all(second >= -1e-15)),
        "min_second_difference": float(np.min(second)),
        "expands_everywhere": bool(np.all(expansion > 1.0)),
        "min_expansion_ratio": float(np.min(expansion)),
        "expansion_ratio_limit_at_zero": a,
        "f_at_theta_max": float(values[-1]),
        "endpoint_error": float(abs(values[-1] - math.pi)),
        "grid_points": points,
    }


def target_monotonicity_violations(pair_angles: np.ndarray, targets: np.ndarray) -> int:
    """Pairs whose target ordering disagrees with the source angle ordering."""

    order = np.argsort(np.asarray(pair_angles, dtype=np.float64), kind="stable")
    ordered = np.asarray(targets, dtype=np.float64)[order]
    return int(np.sum(np.diff(ordered) < -1e-12))


# ---------------------------------------------------------------------------
# frontier target families
# ---------------------------------------------------------------------------


def globally_scaled_targets(targets: np.ndarray, scale: float) -> np.ndarray:
    """``min(scale * target, pi)`` — push every target further apart at once.

    This is the most direct test of "what happens to top-k if high-similarity
    pairs are pushed further apart": the shape of the target curve is unchanged,
    only its overall size grows, and everything that would exceed ``pi`` piles up
    at the antipode.
    """

    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return np.minimum(scale * np.asarray(targets, dtype=np.float64), math.pi)


def scale_clamp_report(targets: np.ndarray, scale: float) -> dict[str, float | int]:
    """Pairs pushed past ``pi`` by a global scale, counted on unordered pairs."""

    values = upper_triangle(np.asarray(targets, dtype=np.float64))
    clamped = scale * values > math.pi + 1e-15
    return {
        "scale": scale,
        "clamped_pair_count": int(np.sum(clamped)),
        "total_pair_count": int(values.size),
        "clamped_pair_fraction": float(np.mean(clamped)),
        "max_unclamped_target_radians": float(scale * np.max(values)),
    }


def rank_uniform_area_targets(ranks: np.ndarray) -> np.ndarray:
    """``1 - cos(theta) = 2 * rank / (N - 1)`` — the coverage upper bound.

    Every source spreads its neighbours over the sphere by equal area, so the
    layout fills the sphere by construction. It is included as the extreme end
    of the frontier: maximum coverage, and whatever Recall@5 comes with it.
    The per-source matrix is asymmetric, so it is symmetrised (the stress term
    is symmetric anyway).
    """

    n = ranks.shape[0]
    areas = 2.0 * ranks.astype(np.float64) / (n - 1)
    np.fill_diagonal(areas, 0.0)
    angles = np.arccos(np.clip(1.0 - areas, -1.0, 1.0))
    np.fill_diagonal(angles, 0.0)
    symmetric = 0.5 * (angles + angles.T)
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


# ---------------------------------------------------------------------------
# optimiser
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    """Deterministic optimiser configuration."""

    seed: int = DEFAULT_SEED
    multistart_count: int = 8
    max_iterations: int = 4000
    ftol: float = 1e-15
    gtol: float = 1e-12


def count_degenerate_rows(coords: np.ndarray, tolerance: float = 1e-12) -> int:
    """Rows whose norm is too small to define a direction.

    :func:`normalise_rows` replaces these with the north pole. A silent fallback
    would hide a collapsed optimisation behind a perfect ``unit_norm_max_error``,
    so every caller that can report it counts them first.
    """

    return int(np.sum(np.linalg.norm(coords, axis=1) < tolerance))


def normalise_rows(coords: np.ndarray) -> np.ndarray:
    """Project rows onto the unit sphere; zero rows fall back to the pole.

    Use :func:`count_degenerate_rows` to detect the fallback — it is reported in
    ``degenerate_fallback_count`` rather than being swallowed.
    """

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    out = coords / safe
    degenerate = (norms < 1e-12).ravel()
    if np.any(degenerate):
        out[degenerate] = np.array([0.0, 0.0, 1.0])
    return out


def angle_stress(coords: np.ndarray, targets: np.ndarray) -> float:
    """``sum_{i<j} (theta_coord - theta_target)^2`` — unweighted over all pairs."""

    residual = coordinate_angle_matrix(coords) - targets
    np.fill_diagonal(residual, 0.0)
    return float(np.sum(residual * residual)) / 2.0


def make_objective(targets: np.ndarray, centering_weight: float = 0.0):
    """Objective and gradient of the unweighted all-pairs angle stress.

    Every one of the ``C(n, 2)`` pairs carries weight 1.0 — no rank weighting, no
    edge/non-edge distinction — because the stretch treats all pairs alike.
    Coordinates are parametrised by unnormalised rows normalised inside, so
    L-BFGS-B runs unconstrained.

    ``centering_weight`` adds ``lambda * ||(1/N) sum x_i||^2``. Pair angles are
    invariant to it in the ideal case (it only penalises the layout leaning to
    one side of the sphere), so it attacks the one-sidedness directly and is
    almost free of rank distortion.
    """

    n = targets.shape[0]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        raw = flat.reshape(n, 3)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        coords = raw / norms
        dots = np.clip(coords @ coords.T, -DOT_CLIP, DOT_CLIP)
        residual = np.arccos(dots) - targets
        np.fill_diagonal(residual, 0.0)
        loss = float(np.sum(residual * residual)) / 2.0
        grad_dots = -residual / np.sqrt(1.0 - dots * dots)
        np.fill_diagonal(grad_dots, 0.0)
        grad_coords = (grad_dots + grad_dots.T) @ coords
        if centering_weight:
            mean = np.mean(coords, axis=0)
            loss += centering_weight * float(mean @ mean)
            grad_coords = grad_coords + (2.0 * centering_weight / n) * mean
        radial = np.sum(grad_coords * coords, axis=1, keepdims=True)
        return loss, ((grad_coords - radial * coords) / norms).ravel()

    return objective


def spectral_initialisation(targets: np.ndarray) -> np.ndarray:
    """Deterministic start from the top-3 eigenvectors of ``cos(target)``."""

    gram = np.cos(0.5 * (targets + targets.T))
    np.fill_diagonal(gram, 1.0)
    values, vectors = np.linalg.eigh(gram)
    coords = vectors[:, -3:] * np.sqrt(np.maximum(values[-3:], 1e-9))
    return normalise_rows(coords)


@dataclass
class LayoutResult:
    coordinates: np.ndarray
    start_objectives: list[float] = field(default_factory=list)
    start_kinds: list[str] = field(default_factory=list)
    selected_start: int = 0
    iterations: list[int] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def optimise_layout(
    targets: np.ndarray, config: LayoutConfig, centering_weight: float = 0.0
) -> LayoutResult:
    """Multistart L-BFGS-B; start 0 is spectral, the rest are seeded gaussians."""

    started = time.perf_counter()
    n = targets.shape[0]
    objective = make_objective(targets, centering_weight)
    starts = [spectral_initialisation(targets)]
    kinds = ["spectral"]
    seeds = [config.seed]
    for offset in range(1, config.multistart_count):
        seed = config.seed + offset
        rng = np.random.default_rng(seed)
        starts.append(normalise_rows(rng.standard_normal((n, 3))))
        kinds.append("gaussian")
        seeds.append(seed)

    options = {
        "maxiter": config.max_iterations,
        "maxfun": config.max_iterations * 2,
        "ftol": config.ftol,
        "gtol": config.gtol,
    }
    result = LayoutResult(coordinates=np.zeros((n, 3)), start_kinds=kinds, seeds=seeds)
    best_value = math.inf
    for index, start in enumerate(starts):
        solved = minimize(
            objective,
            start.ravel().copy(),
            jac=True,
            method="L-BFGS-B",
            options=options,
        )
        value = float(solved.fun)
        result.start_objectives.append(value)
        result.iterations.append(int(solved.nit))
        if value < best_value:
            best_value = value
            result.selected_start = index
            result.coordinates = normalise_rows(solved.x.reshape(n, 3))
    result.elapsed_seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# metrics — A. sphere coverage
# ---------------------------------------------------------------------------


def nearest_neighbour_angles(coords: np.ndarray) -> np.ndarray:
    """Angular distance from each node to its closest other node."""

    angles = coordinate_angle_matrix(coords)
    np.fill_diagonal(angles, math.inf)
    return np.min(angles, axis=1)


def angle_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def uniform_nn_marginal_quantile(node_count: int, quantile: float) -> float:
    """Analytic nearest-neighbour quantile for one point among ``n`` uniform ones.

    ``P(NN > theta) = ((1 + cos theta) / 2) ** (n - 1)``.
    """

    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must be in [0, 1)")
    cosine = 2.0 * (1.0 - quantile) ** (1.0 / (node_count - 1)) - 1.0
    return math.acos(max(-1.0, min(1.0, cosine)))


def fibonacci_sphere(count: int) -> np.ndarray:
    """Deterministic quasi-uniform probe points used to sample empty space."""

    if count < 1:
        raise ValueError("count must be positive")
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = index * math.pi * (1.0 + math.sqrt(5.0))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def covering_radius(coords: np.ndarray, probes: np.ndarray, chunk: int = 8192) -> float:
    """Largest probe-to-nearest-node angle — a sampled covering radius.

    With a finite probe set this is a lower bound on the true covering radius;
    the probe count is reported alongside so the estimate stays comparable.
    """

    worst = 0.0
    for start in range(0, probes.shape[0], chunk):
        block = probes[start : start + chunk]
        nearest = np.max(np.clip(block @ coords.T, -1.0, 1.0), axis=1)
        worst = max(worst, float(np.max(np.arccos(nearest))))
    return worst


def equal_area_cell_counts(coords: np.ndarray, bands: int, sectors: int) -> np.ndarray:
    """Node counts of ``bands * sectors`` exactly equal-area spherical cells.

    The sphere's area element is uniform in ``(z, phi)``, so equal ``z`` slabs
    crossed with equal longitude sectors give exactly equal-area cells.
    """

    z = np.clip(coords[:, 2], -1.0, 1.0)
    band = np.clip(((z + 1.0) / 2.0 * bands).astype(np.int64), 0, bands - 1)
    phi = np.arctan2(coords[:, 1], coords[:, 0]) + math.pi
    sector = np.clip((phi / (2.0 * math.pi) * sectors).astype(np.int64), 0, sectors - 1)
    return np.bincount(band * sectors + sector, minlength=bands * sectors)


def cell_occupancy_summary(counts: np.ndarray, node_count: int) -> dict[str, float]:
    cells = int(counts.size)
    share = 1.0 / cells
    return {
        "cells": cells,
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "empty_cells": int(np.sum(counts == 0)),
        "uniform_expected_std": math.sqrt(node_count * share * (1.0 - share)),
        "uniform_expected_empty_cells": cells * (1.0 - share) ** node_count,
    }


def mean_vector_norm(coords: np.ndarray) -> float:
    """``|| (1/N) sum x_i ||`` — how far the point set leans to one side.

    Zero means perfectly balanced over the sphere; one means every node sits on
    the same spot. A uniform 602-point sample averages about 0.041. This is the
    single most direct measure of the "everything is crammed onto one side of
    the ball" complaint.
    """

    return float(np.linalg.norm(np.mean(coords, axis=0)))


def best_hemisphere_fraction(coords: np.ndarray) -> float:
    """Share of nodes inside the hemisphere centred on the mean direction.

    About 0.53 for a uniform sample (a random point set always leans a little);
    1.0 means every node fits in one half of the sphere.
    """

    mean = np.mean(coords, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-15:
        return 0.5
    return float(np.mean(coords @ (mean / norm) >= 0.0))


def bias_metrics(coords: np.ndarray) -> dict[str, float]:
    """One-sidedness of the layout, with the uniform-sphere expectations."""

    return {
        "mean_vector_norm": mean_vector_norm(coords),
        "best_hemisphere_fraction": best_hemisphere_fraction(coords),
    }


def coverage_metrics(coords: np.ndarray, probes: np.ndarray) -> dict[str, object]:
    """Section A of the brief for one coordinate set."""

    n = coords.shape[0]
    pair_angles = upper_triangle(coordinate_angle_matrix(coords))
    metrics: dict[str, object] = {
        "nearest_neighbour_angle": angle_summary(nearest_neighbour_angles(coords)),
        "covering_radius_radians": covering_radius(coords, probes),
        "covering_radius_probe_points": int(probes.shape[0]),
        "pair_angle_ks_vs_uniform_sphere": ks_statistic_uniform_sphere(pair_angles),
        "pair_angle": angle_summary(pair_angles),
        **bias_metrics(coords),
    }
    cells: dict[str, object] = {}
    for bands, sectors in CELL_LAYOUTS:
        counts = equal_area_cell_counts(coords, bands, sectors)
        cells[f"cells_{bands * sectors}"] = {
            "partition": f"{bands} equal-area z bands x {sectors} longitude sectors",
            **cell_occupancy_summary(counts, n),
        }
    metrics["equal_area_cells"] = cells
    return metrics


def uniform_reference(
    node_count: int,
    probes: np.ndarray,
    *,
    replicates: int = DEFAULT_UNIFORM_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Monte-Carlo section-A values for ``replicates`` uniform point sets."""

    rng = np.random.default_rng(seed)
    nn: list[dict[str, float]] = []
    covering: list[float] = []
    ks: list[float] = []
    bias: list[dict[str, float]] = []
    median_pair: list[float] = []
    cells: dict[str, list[dict[str, float]]] = {
        f"cells_{b * s}": [] for b, s in CELL_LAYOUTS
    }
    for _ in range(replicates):
        coords = normalise_rows(rng.standard_normal((node_count, 3)))
        nn.append(angle_summary(nearest_neighbour_angles(coords)))
        covering.append(covering_radius(coords, probes))
        pairs = upper_triangle(coordinate_angle_matrix(coords))
        ks.append(ks_statistic_uniform_sphere(pairs))
        median_pair.append(float(np.median(pairs)))
        bias.append(bias_metrics(coords))
        for bands, sectors in CELL_LAYOUTS:
            counts = equal_area_cell_counts(coords, bands, sectors)
            cells[f"cells_{bands * sectors}"].append(
                cell_occupancy_summary(counts, node_count)
            )
    keys = ("min", "p10", "median", "p90", "max", "mean")
    return {
        "replicates": replicates,
        "seed": seed,
        "nearest_neighbour_angle_mean": {
            key: float(np.mean([row[key] for row in nn])) for key in keys
        },
        "nearest_neighbour_angle_analytic": {
            "p10": uniform_nn_marginal_quantile(node_count, 0.10),
            "median": uniform_nn_marginal_quantile(node_count, 0.50),
            "p90": uniform_nn_marginal_quantile(node_count, 0.90),
        },
        "covering_radius_radians_mean": float(np.mean(covering)),
        "covering_radius_radians_max": float(np.max(covering)),
        "pair_angle_ks_vs_uniform_sphere_mean": float(np.mean(ks)),
        "pair_angle_median_mean": float(np.mean(median_pair)),
        "mean_vector_norm_mean": float(
            np.mean([row["mean_vector_norm"] for row in bias])
        ),
        "best_hemisphere_fraction_mean": float(
            np.mean([row["best_hemisphere_fraction"] for row in bias])
        ),
        "equal_area_cells": {
            name: {
                "std_mean": float(np.mean([row["std"] for row in rows])),
                "empty_cells_mean": float(
                    np.mean([row["empty_cells"] for row in rows])
                ),
                "max_mean": float(np.mean([row["max"] for row in rows])),
                "uniform_expected_std": rows[0]["uniform_expected_std"],
                "uniform_expected_empty_cells": rows[0]["uniform_expected_empty_cells"],
            }
            for name, rows in cells.items()
        },
    }


# ---------------------------------------------------------------------------
# metrics — cluster separation
# ---------------------------------------------------------------------------


def cluster_separation_metrics(
    coords: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    """Intra/inter cluster angular separation, silhouette and centroid spread.

    ``labels`` are the 48D ``cosine_k_medoids_v1:k=7:iterations=100:seed=20260806``
    assignments; all angles are great-circle angles on the layout sphere.
    """

    n = coords.shape[0]
    angles = coordinate_angle_matrix(coords)
    same = labels[:, None] == labels[None, :]
    offdiag = ~np.eye(n, dtype=bool)
    intra_mask = same & offdiag
    inter_mask = (~same) & offdiag
    intra = float(np.mean(angles[intra_mask]))
    inter = float(np.mean(angles[inter_mask]))

    unique = np.unique(labels)
    silhouette = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own = labels[i]
        own_mask = intra_mask[i] & (labels == own)
        if not np.any(own_mask):
            silhouette[i] = 0.0
            continue
        a_i = float(np.mean(angles[i, own_mask]))
        b_i = math.inf
        for other in unique:
            if other == own:
                continue
            b_i = min(b_i, float(np.mean(angles[i, labels == other])))
        silhouette[i] = (b_i - a_i) / max(a_i, b_i)

    centroids = normalise_rows(
        np.asarray([coords[labels == c].sum(axis=0) for c in unique], dtype=np.float64)
    )
    centroid_angles = upper_triangle(coordinate_angle_matrix(centroids))
    return {
        "cluster_count": int(unique.size),
        "cluster_sizes": [int(np.sum(labels == c)) for c in unique],
        "mean_intra_cluster_angle": intra,
        "mean_inter_cluster_angle": inter,
        "separation_ratio": inter / intra,
        "silhouette_mean": float(np.mean(silhouette)),
        "silhouette_median": float(np.median(silhouette)),
        "centroid_angle_min": float(np.min(centroid_angles)),
        "centroid_angle_mean": float(np.mean(centroid_angles)),
        "centroid_angle_max": float(np.max(centroid_angles)),
    }


def separation_ratio(coords: np.ndarray, labels: np.ndarray) -> float:
    """Fast ``mean inter-cluster angle / mean intra-cluster angle``."""

    n = coords.shape[0]
    angles = coordinate_angle_matrix(coords)
    same = labels[:, None] == labels[None, :]
    offdiag = ~np.eye(n, dtype=bool)
    intra = float(np.mean(angles[same & offdiag]))
    inter = float(np.mean(angles[(~same) & offdiag]))
    return inter / intra


# ---------------------------------------------------------------------------
# metrics — B. order fidelity (monotonicity, no longer exact ratio preservation)
# ---------------------------------------------------------------------------


def pair_order_inversion_rate(
    source: np.ndarray,
    observed: np.ndarray,
    *,
    samples: int = INVERSION_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Fraction of sampled pair-of-pairs whose angular ordering is inverted.

    The exact count needs ``C(180901, 2)`` comparisons, so a seeded uniform
    sample of pair-of-pairs is used. ``kendall_tau_estimate = 1 - 2 * rate``.
    """

    rng = np.random.default_rng(seed)
    left = rng.integers(0, source.size, samples)
    right = rng.integers(0, source.size, samples)
    keep = source[left] != source[right]
    inverted = np.sign(source[left] - source[right]) != np.sign(
        observed[left] - observed[right]
    )
    rate = float(np.mean(inverted[keep]))
    return {
        "samples": int(np.sum(keep)),
        "inversion_rate": rate,
        "kendall_tau_estimate": 1.0 - 2.0 * rate,
        "seed": seed,
    }


def realised_stretch_by_source_decile(
    source: np.ndarray, observed: np.ndarray, deciles: int = STRETCH_DECILES
) -> list[dict[str, float]]:
    """Realised ``observed / source`` profile per source-angle decile.

    This is the coordinate-level version of the contract's "near pairs open up a
    little, far pairs open up a lot". The target function ``f`` satisfies it by
    construction; the projected 3D coordinates need not, so it is measured here
    and reported for every layout.
    """

    src = np.asarray(source, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    ratio = obs / src
    edges = np.quantile(src, np.linspace(0.0, 1.0, deciles + 1))
    index = np.clip(np.searchsorted(edges, src, side="right") - 1, 0, deciles - 1)
    profile: list[dict[str, float]] = []
    for decile in range(deciles):
        mask = index == decile
        values = ratio[mask]
        profile.append(
            {
                "decile": decile + 1,
                "pair_count": int(values.size),
                "source_angle_low": float(edges[decile]),
                "source_angle_high": float(edges[decile + 1]),
                "min": float(np.min(values)),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "fraction_below_1": float(np.mean(values < 1.0)),
            }
        )
    return profile


def nearest_fraction_compressed(
    source: np.ndarray, observed: np.ndarray, fraction: float = NEAREST_FRACTION
) -> float:
    """Share of the closest ``fraction`` of source pairs that ended up compressed."""

    src = np.asarray(source, dtype=np.float64)
    count = max(1, int(round(fraction * src.size)))
    nearest = np.argsort(src, kind="stable")[:count]
    return float(np.mean(np.asarray(observed)[nearest] / src[nearest] < 1.0))


def order_fidelity_metrics(
    coords: np.ndarray,
    angles: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    inversion_samples: int = INVERSION_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Section B: how faithfully the coordinates keep the source angle order.

    ``targets`` is optional: stress and target correlations are only meaningful
    for a layout that actually aimed at those targets, so layouts that never did
    (baseline, rank-scaled) report ``None`` instead of a number computed against
    a foreign target.
    """

    observed = upper_triangle(coordinate_angle_matrix(coords))
    source = upper_triangle(angles)
    ratios = observed / source
    profile = realised_stretch_by_source_decile(source, observed)
    metrics: dict[str, object] = {
        "spearman_coord_vs_source_angle": float(spearmanr(observed, source).statistic),
        "pearson_coord_vs_source_angle": float(np.corrcoef(observed, source)[0, 1]),
        "coordinate_pair_order_inversions": pair_order_inversion_rate(
            source, observed, samples=inversion_samples, seed=seed
        ),
        "realised_stretch_by_source_decile": profile,
        "near_decile_median_stretch": profile[0]["median"],
        "near_decile_fraction_compressed": profile[0]["fraction_below_1"],
        "nearest_5pct_fraction_compressed": nearest_fraction_compressed(
            source, observed
        ),
        "all_pairs_fraction_compressed": float(np.mean(ratios < 1.0)),
        "realised_stretch_monotone_across_deciles": bool(
            all(
                profile[i]["median"] <= profile[i + 1]["median"]
                for i in range(len(profile) - 1)
            )
        ),
        "angle_ratio": {
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios)),
            "coefficient_of_variation": float(np.std(ratios) / np.mean(ratios)),
            "min": float(np.min(ratios)),
            "p10": float(np.percentile(ratios, 10)),
            "median": float(np.median(ratios)),
            "p90": float(np.percentile(ratios, 90)),
            "max": float(np.max(ratios)),
        },
        "pair_count": int(source.size),
    }
    if targets is None:
        metrics["target_reference"] = None
        for key in (
            "spearman_coord_vs_target",
            "pearson_coord_vs_target",
            "angle_stress",
            "normalised_angle_stress",
            "angle_rmse_radians",
        ):
            metrics[key] = None
        return metrics

    target = upper_triangle(targets)
    residual = observed - target
    stress = float(np.sum(residual * residual))
    metrics["target_reference"] = "own layout target"
    metrics["spearman_coord_vs_target"] = float(spearmanr(observed, target).statistic)
    metrics["pearson_coord_vs_target"] = float(np.corrcoef(observed, target)[0, 1])
    metrics["angle_stress"] = stress
    metrics["normalised_angle_stress"] = math.sqrt(
        stress / float(np.sum(target * target))
    )
    metrics["angle_rmse_radians"] = math.sqrt(stress / target.size)
    return metrics


# ---------------------------------------------------------------------------
# metrics — C. top-k preservation (definitions identical to task 2)
# ---------------------------------------------------------------------------


def unit_norm_max_error(coords: np.ndarray) -> float:
    """Largest deviation of any row norm from 1."""

    return float(np.max(np.abs(np.linalg.norm(coords, axis=1) - 1.0)))


def recall_metrics(
    true_top_k: np.ndarray, coordinate_top_k: np.ndarray
) -> dict[str, float]:
    """``mean_i |T_i & C_i| / k``, ``|{i: T_i subset C_i}| / N``, ``>=1 hit`` rate."""

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
    """Cosine bottom-decile non-neighbours placed inside the top-k radius."""

    n = coordinate_angles.shape[0]
    per_source = int(math.floor(fraction * (n - 1)))
    total = 0
    for i in range(n):
        farthest = float(np.max(coordinate_angles[i, true_top_k[i]]))
        worst = cosine_order[i, -per_source:] if per_source else cosine_order[i, :0]
        total += int(np.sum(coordinate_angles[i, worst] < farthest))
    return total


def topk_metrics(
    coords: np.ndarray,
    node_ids: Sequence[int],
    cosine_ranks: np.ndarray,
    cosine_order: np.ndarray,
    union_pairs: np.ndarray,
    union_cosines: np.ndarray,
) -> dict[str, object]:
    """Section C of the brief for one coordinate set."""

    n = coords.shape[0]
    true_top_k = cosine_order[:, :TOP_K]
    angles = coordinate_angle_matrix(coords)
    dots = np.clip(coords @ coords.T, -1.0, 1.0)
    coord_ranks, coord_order = similarity_rank_matrix(dots, node_ids)
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
    metrics["union_edge_rmse_radians_original_acos"] = float(
        np.sqrt(np.mean((observed - original_target) ** 2))
    )
    metrics["unit_norm_max_error"] = unit_norm_max_error(coords)
    metrics["bottom_decile_false_close_count"] = bottom_decile_false_close_count(
        angles, cosine_order, true_top_k
    )
    return metrics


# ---------------------------------------------------------------------------
# constrained greedy refinement on the real metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefineConfig:
    """Deterministic refinement budget and constraint tolerances."""

    seed: int = DEFAULT_SEED
    max_passes: int = 60
    pull_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    step_radii: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4)
    directions_per_radius: int = 3
    #: Separation ratio may fall by at most this relative amount.
    separation_max_relative_drop: float = 0.01
    #: Uniform-sphere KS may rise by at most this absolute amount.
    ks_max_absolute_increase: float = 0.005
    #: Coverage guard: ``"ks"`` bounds the uniform-sphere KS, ``"mean_vector_norm"``
    #: bounds the one-sidedness directly instead.
    constraint_mode: str = "ks"
    #: ``mean_vector_norm`` may rise by at most this absolute amount in that mode.
    mean_vector_norm_max_absolute_increase: float = 0.005


class _RecallState:
    """Incremental top-5 hit bookkeeping for single-node moves.

    Moving one node rewrites one row and one column of the dot matrix, so only
    the moved node's own top-5 plus the rows where it enters or leaves the top-5
    need rescoring.
    """

    def __init__(self, coords: np.ndarray, true_mask: np.ndarray):
        self.n = coords.shape[0]
        self.coords = coords.copy()
        self.true_mask = true_mask
        self.dots = np.clip(coords @ coords.T, -1.0, 1.0)
        np.fill_diagonal(self.dots, -2.0)
        self.top5 = np.zeros((self.n, TOP_K), dtype=np.int64)
        self.fifth = np.zeros(self.n, dtype=np.float64)
        self.hits = np.zeros(self.n, dtype=np.int64)
        for u in range(self.n):
            self._rescore(u, self.dots[u])

    def _rescore(self, u: int, row: np.ndarray) -> None:
        idx = np.argpartition(-row, TOP_K)[:TOP_K]
        self.top5[u] = idx
        self.fifth[u] = float(np.min(row[idx]))
        self.hits[u] = int(np.sum(self.true_mask[u, idx]))

    def total_hits(self) -> int:
        return int(np.sum(self.hits))

    def _affected(self, v: int, dv: np.ndarray) -> np.ndarray:
        touched = (dv >= self.fifth) | np.any(self.top5 == v, axis=1)
        touched[v] = False
        return np.nonzero(touched)[0]

    def score_move(self, v: int, point: np.ndarray) -> tuple[int, np.ndarray]:
        """Total hits if ``v`` moved to ``point``; also returns the new row."""

        dv = np.clip(self.coords @ point, -1.0, 1.0)
        dv[v] = -2.0
        idx = np.argpartition(-dv, TOP_K)[:TOP_K]
        total = self.total_hits() - int(self.hits[v])
        total += int(np.sum(self.true_mask[v, idx]))
        for u in self._affected(v, dv):
            row = self.dots[u].copy()
            row[v] = dv[u]
            best = np.argpartition(-row, TOP_K)[:TOP_K]
            total += int(np.sum(self.true_mask[u, best])) - int(self.hits[u])
        return total, dv

    def apply_move(self, v: int, point: np.ndarray, dv: np.ndarray) -> None:
        affected = self._affected(v, dv)
        self.coords[v] = point
        self.dots[v, :] = dv
        self.dots[:, v] = dv
        self.dots[v, v] = -2.0
        self._rescore(v, self.dots[v])
        for u in affected:
            self._rescore(int(u), self.dots[u])


def _tangent_candidates(
    current: np.ndarray, pull: np.ndarray, config: RefineConfig, rng
) -> list[np.ndarray]:
    candidates = [
        normalise_rows(((1.0 - t) * current + t * pull)[None, :])[0]
        for t in config.pull_fractions
    ]
    for radius in config.step_radii:
        for _ in range(config.directions_per_radius):
            g = rng.standard_normal(3)
            g -= float(g @ current) * current
            norm = float(np.linalg.norm(g))
            if norm < 1e-12:
                continue
            g /= norm
            candidates.append(math.cos(radius) * current + math.sin(radius) * g)
    return candidates


def refine_recall_constrained(
    coords: np.ndarray,
    true_top_k: np.ndarray,
    labels: np.ndarray,
    config: RefineConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Greedy coordinate-wise search on mean Recall@5 under coverage constraints.

    Node visit order is ascending index (graph48 row order) and the candidate
    perturbations come from a single seeded generator, so the whole pass is
    deterministic. A move is accepted only when it strictly increases the total
    top-5 hit count *and* keeps the cluster separation ratio and the configured
    coverage guard inside tolerance. The guard is either the uniform-sphere KS
    statistic (``constraint_mode="ks"``) or the layout's one-sidedness
    (``constraint_mode="mean_vector_norm"``) — the refinement pulls neighbours
    together, which makes a leaning layout lean further, so which quantity is
    guarded changes what the refinement is allowed to spend.
    """

    n = coords.shape[0]
    true_mask = np.zeros((n, n), dtype=bool)
    true_mask[np.arange(n)[:, None], true_top_k] = True
    state = _RecallState(coords, true_mask)
    rng = np.random.default_rng(config.seed)

    if config.constraint_mode not in ("ks", "mean_vector_norm"):
        raise ValueError(f"unknown constraint_mode {config.constraint_mode!r}")
    base_separation = separation_ratio(coords, labels)
    base_ks = ks_statistic_uniform_sphere(
        upper_triangle(coordinate_angle_matrix(coords))
    )
    base_mean_norm = mean_vector_norm(coords)
    min_separation = base_separation * (1.0 - config.separation_max_relative_drop)
    max_ks = base_ks + config.ks_max_absolute_increase
    max_mean_norm = base_mean_norm + config.mean_vector_norm_max_absolute_increase

    started = time.perf_counter()
    history: list[dict[str, object]] = []
    moved_nodes: set[int] = set()
    rejected_separation = 0
    rejected_coverage = 0
    for sweep in range(config.max_passes):
        accepted = 0
        for v in range(n):
            current = state.coords[v]
            pull = state.coords[true_top_k[v]].sum(axis=0)
            if float(np.linalg.norm(pull)) < 1e-12:
                pull = current
            else:
                pull = pull / float(np.linalg.norm(pull))
            scored = []
            for point in _tangent_candidates(current, pull, config, rng):
                total, dv = state.score_move(v, point)
                scored.append((total, point, dv))
            scored.sort(key=lambda item: -item[0])
            baseline_total = state.total_hits()
            for total, point, dv in scored:
                if total <= baseline_total:
                    break
                previous = state.coords[v].copy()
                state.apply_move(v, point, dv)
                candidate_separation = separation_ratio(state.coords, labels)
                if candidate_separation < min_separation:
                    rejected_separation += 1
                    _, back = state.score_move(v, previous)
                    state.apply_move(v, previous, back)
                    continue
                if config.constraint_mode == "ks":
                    breached = (
                        ks_statistic_uniform_sphere(
                            upper_triangle(coordinate_angle_matrix(state.coords))
                        )
                        > max_ks
                    )
                else:
                    breached = mean_vector_norm(state.coords) > max_mean_norm
                if breached:
                    rejected_coverage += 1
                    _, back = state.score_move(v, previous)
                    state.apply_move(v, previous, back)
                    continue
                accepted += 1
                moved_nodes.add(v)
                break
        history.append(
            {
                "sweep": sweep,
                "accepted_moves": accepted,
                "total_hits": state.total_hits(),
                "mean_recall_at_5": state.total_hits() / (n * TOP_K),
            }
        )
        if accepted == 0:
            break

    report = {
        "seed": config.seed,
        "max_passes": config.max_passes,
        "sweeps_run": len(history),
        "history": history,
        "moved_node_count": len(moved_nodes),
        "candidates_per_node": len(config.pull_fractions)
        + len(config.step_radii) * config.directions_per_radius,
        "constraints": {
            "constraint_mode": config.constraint_mode,
            "separation_ratio_before": base_separation,
            "separation_ratio_floor": min_separation,
            "separation_max_relative_drop": config.separation_max_relative_drop,
            "pair_angle_ks_before": base_ks,
            "pair_angle_ks_ceiling": max_ks,
            "ks_max_absolute_increase": config.ks_max_absolute_increase,
            "mean_vector_norm_before": base_mean_norm,
            "mean_vector_norm_ceiling": max_mean_norm,
            "mean_vector_norm_max_absolute_increase": (
                config.mean_vector_norm_max_absolute_increase
            ),
            "rejected_by_separation": rejected_separation,
            "rejected_by_coverage_guard": rejected_coverage,
            "rejected_by_ks": (
                rejected_coverage if config.constraint_mode == "ks" else 0
            ),
            "binding": rejected_separation + rejected_coverage > 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return normalise_rows(state.coords), report


# ---------------------------------------------------------------------------
# artifact I/O
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph48(path: Path) -> tuple[list[int], np.ndarray]:
    ids: list[int] = []
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 48:
            raise ValueError("graph48 header is too short")
        for row in reader:
            ids.append(canonical_id(row[0]))
            rows.append([float(value) for value in row[-48:]])
    return ids, np.asarray(rows, dtype=np.float64)


def load_union_edges(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def coordinates_in_id_order(
    lookup: dict[int, tuple[float, float, float]], node_ids: Sequence[int]
) -> np.ndarray:
    """Reorder a ``{int id: (x, y, z)}`` map into the graph48 row order."""

    missing = [node_id for node_id in node_ids if node_id not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} ids missing from coordinate source")
    return np.asarray([lookup[node_id] for node_id in node_ids], dtype=np.float64)


def load_public_json_coordinates(path: Path) -> dict[int, tuple[float, float, float]]:
    """Read ``graph.nodes[]`` coordinates keyed by **int** id.

    ``node_id`` is a string in this file and an integer everywhere else, so the
    key is normalised here rather than at every call site.
    """

    with path.open() as handle:
        payload = json.load(handle)
    return {
        canonical_id(node["node_id"]): (
            float(node["x"]),
            float(node["y"]),
            float(node["z"]),
        )
        for node in payload["graph"]["nodes"]
    }


def load_cluster_labels(path: Path) -> tuple[dict[int, str], str]:
    """Read the published ``component_id`` per node plus the clusterer policy.

    These labels come from ``cosine_k_medoids_v1:k=7:iterations=100:seed=20260806``
    applied to the 48D vectors, which is exactly the clustering the brief asks
    for, so they are reused rather than recomputed.
    """

    with path.open() as handle:
        payload = json.load(handle)
    labels = {
        canonical_id(node["node_id"]): str(node["component_id"])
        for node in payload["graph"]["nodes"]
    }
    return labels, str(payload["graph"]["clusterer"])


def load_coordinates_csv(path: Path) -> dict[int, tuple[float, float, float]]:
    with path.open(newline="") as handle:
        return {
            canonical_id(row["cocktail_id"]): (
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            )
            for row in csv.DictReader(handle)
        }


def write_coordinates_csv(path: Path, node_ids: Sequence[int], coords: np.ndarray):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cocktail_id", "x", "y", "z"])
        for node_id, row in zip(node_ids, coords):
            writer.writerow(
                [node_id, repr(float(row[0])), repr(float(row[1])), repr(float(row[2]))]
            )


def coordinate_digest(node_ids: Sequence[int], coords: np.ndarray) -> str:
    """SHA-256 over ``node_id:x:y:z`` lines in the graph48 row order."""

    digest = hashlib.sha256()
    for node_id, row in zip(node_ids, coords):
        digest.update(
            f"{node_id}:{float(row[0])!r}:{float(row[1])!r}:{float(row[2])!r}\n".encode()
        )
    return digest.hexdigest()


def build_public_json(
    baseline: dict,
    node_ids: Sequence[int],
    coords: np.ndarray,
    layout_report: dict,
) -> dict:
    """Reproduce the baseline public schema with the new coordinates."""

    index = {node_id: i for i, node_id in enumerate(node_ids)}
    graph = dict(baseline["graph"])
    nodes = []
    for node in baseline["graph"]["nodes"]:
        node_id = canonical_id(node["node_id"])
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
# shared evaluation
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    ids: list[int]
    angles: np.ndarray
    pair_angles: np.ndarray
    ranks: np.ndarray
    order: np.ndarray
    union_pairs: np.ndarray
    union_cosines: np.ndarray
    labels: np.ndarray
    cluster_policy: str
    probes: np.ndarray


def evaluate_layout(
    coords: np.ndarray, data: Dataset, targets: np.ndarray | None = None
) -> dict[str, object]:
    """A + B + C + cluster metrics for one coordinate set.

    ``targets`` is optional so that layouts which never aimed at a given target
    do not get a meaningless stress number (see :func:`order_fidelity_metrics`).
    """

    return {
        "coverage": coverage_metrics(coords, data.probes),
        "cluster": cluster_separation_metrics(coords, data.labels),
        "order_fidelity": order_fidelity_metrics(coords, data.angles, targets),
        "topk": topk_metrics(
            coords,
            data.ids,
            data.ranks,
            data.order,
            data.union_pairs,
            data.union_cosines,
        ),
        "degenerate_fallback_count": count_degenerate_rows(coords),
    }


def gate_row(evaluation: dict[str, object]) -> dict[str, float]:
    """Flatten the metrics a gate or floor can be evaluated against."""

    coverage = evaluation["coverage"]
    cluster = evaluation["cluster"]
    order = evaluation["order_fidelity"]
    topk = evaluation["topk"]
    return {
        "mean_recall_at_5": float(topk["mean_recall_at_5"]),
        "hit_rate_at_5": float(topk["hit_rate_at_5"]),
        "full_recovery_rate": float(topk["full_recovery_rate"]),
        "bottom_decile_false_close_count": float(
            topk["bottom_decile_false_close_count"]
        ),
        "union_edge_rmse_radians_original_acos": float(
            topk["union_edge_rmse_radians_original_acos"]
        ),
        "unit_norm_max_error": float(topk["unit_norm_max_error"]),
        "pair_angle_ks_vs_uniform_sphere": float(
            coverage["pair_angle_ks_vs_uniform_sphere"]
        ),
        "covering_radius_radians": float(coverage["covering_radius_radians"]),
        "empty_cells_100": float(
            coverage["equal_area_cells"]["cells_100"]["empty_cells"]
        ),
        "mean_vector_norm": float(coverage["mean_vector_norm"]),
        "best_hemisphere_fraction": float(coverage["best_hemisphere_fraction"]),
        "pair_angle_median": float(coverage["pair_angle"]["median"]),
        "near_decile_median_stretch": float(order["near_decile_median_stretch"]),
        "nearest_5pct_fraction_compressed": float(
            order["nearest_5pct_fraction_compressed"]
        ),
        "separation_ratio": float(cluster["separation_ratio"]),
        "silhouette_mean": float(cluster["silhouette_mean"]),
        "centroid_angle_min": float(cluster["centroid_angle_min"]),
    }


def check_thresholds(
    row: dict[str, float], thresholds: dict[str, tuple[str, float]]
) -> dict[str, object]:
    """Evaluate ``{metric: (op, bound)}`` against a flattened metric row."""

    checks: dict[str, object] = {}
    for metric, (op, bound) in thresholds.items():
        value = row[metric]
        passed = value >= bound if op == ">=" else value <= bound
        checks[metric] = {
            "value": value,
            "operator": op,
            "threshold": bound,
            "passed": bool(passed),
        }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {"checks": checks, "passed": not failed, "failed": failed}


def load_dataset(
    artifact_dir: Path, probe_points: int
) -> tuple[Dataset, dict[str, Path]]:
    paths = {
        "graph48": artifact_dir / "graph48.csv",
        "union": artifact_dir / "graph48-union-edges.csv",
        "directed": artifact_dir / "graph48-directed-top5.csv",
        "baseline": artifact_dir / "spherical-graph-public.json",
    }
    ids, vectors = load_graph48(paths["graph48"])
    cosines = exact_cosine_matrix(vectors)
    angles = angle_matrix(cosines)
    ranks, order = similarity_rank_matrix(cosines, ids)
    index = {node_id: i for i, node_id in enumerate(ids)}
    union_rows = load_union_edges(paths["union"])
    union_pairs = np.asarray(
        [
            [index[canonical_id(row["a_id"])], index[canonical_id(row["b_id"])]]
            for row in union_rows
        ],
        dtype=np.int64,
    )
    union_cosines = np.asarray(
        [float(row["cosine"]) for row in union_rows], dtype=np.float64
    )
    label_map, policy = load_cluster_labels(paths["baseline"])
    labels = np.asarray([label_map[node_id] for node_id in ids])
    data = Dataset(
        ids=ids,
        angles=angles,
        pair_angles=upper_triangle(angles),
        ranks=ranks,
        order=order,
        union_pairs=union_pairs,
        union_cosines=union_cosines,
        labels=labels,
        cluster_policy=policy,
        probes=fibonacci_sphere(probe_points),
    )
    return data, paths


# ---------------------------------------------------------------------------
# linear mode driver
# ---------------------------------------------------------------------------

LINEAR_SELECTION_RULE = (
    "lowest coordinate pair-angle KS distance to F(theta)=(1-cos theta)/2 "
    "(sphere coverage is the primary goal), subject to clamped_pair_fraction "
    f"<= {MAX_CLAMPED_FRACTION}; if no candidate qualifies, fall back to "
    "k = pi / max_angle, which clamps nothing by construction"
)


def select_linear_candidate(runs: dict[str, dict]) -> str:
    """Apply :data:`LINEAR_SELECTION_RULE` to the finished candidate runs."""

    eligible = [
        name
        for name in runs
        if runs[name]["clamp"]["clamped_pair_fraction"] <= MAX_CLAMPED_FRACTION
    ]
    if not eligible:
        return "k_max_antipodal"
    return min(
        eligible,
        key=lambda name: runs[name]["evaluation"]["coverage"][
            "pair_angle_ks_vs_uniform_sphere"
        ],
    )


def run_linear_mode(args: argparse.Namespace) -> int:
    data, paths = load_dataset(args.artifact_dir, args.probe_points)
    n = len(data.ids)
    ks_k, ks_value = k_ks_optimal(data.pair_angles)
    candidates = {
        "k_max_antipodal": k_max_antipodal(data.pair_angles),
        "k_median_match": k_median_match(data.pair_angles),
        "k_ks_optimal": ks_k,
    }
    notes = {
        "k_max_antipodal": "k = pi / max_angle; widest pair becomes antipodal",
        "k_median_match": "k = (pi/2) / median_angle; median matches a uniform sphere",
        "k_ks_optimal": (
            "k minimising the KS distance between the scaled target pair-angle "
            "distribution and F(theta) = (1 - cos theta) / 2"
        ),
    }
    config = LayoutConfig(
        seed=args.seed,
        multistart_count=args.starts,
        max_iterations=args.max_iterations,
    )
    runs: dict[str, dict] = {}
    coords_by_candidate: dict[str, np.ndarray] = {}
    targets_by_candidate: dict[str, np.ndarray] = {}
    for name, k in candidates.items():
        targets = scaled_targets(data.angles, k)
        np.fill_diagonal(targets, 0.0)
        result = optimise_layout(targets, config)
        coords_by_candidate[name] = result.coordinates
        targets_by_candidate[name] = targets
        runs[name] = {
            "k": k,
            "policy": notes[name],
            "target_pair_angle_ks_vs_uniform_sphere": ks_statistic_uniform_sphere(
                scaled_targets(data.pair_angles, k)
            ),
            "clamp": clamp_report(data.pair_angles, k),
            "ratio_preservation_max_relative_error": ratio_preservation_error(
                data.pair_angles, scaled_targets(data.pair_angles, k)
            ),
            "evaluation": evaluate_layout(result.coordinates, data, targets),
            "multistart_objectives": result.start_objectives,
            "multistart_seeds": result.seeds,
            "multistart_start_kinds": result.start_kinds,
            "multistart_iterations": result.iterations,
            "selected_start": result.selected_start,
            "elapsed_seconds": result.elapsed_seconds,
        }
        print(
            f"{name}: k={k:.6f} clamped="
            f"{runs[name]['clamp']['clamped_pair_count']} "
            f"coordKS={runs[name]['evaluation']['coverage']['pair_angle_ks_vs_uniform_sphere']:.4f} "
            f"recall@5={runs[name]['evaluation']['topk']['mean_recall_at_5']:.4f} "
            f"({result.elapsed_seconds:.1f}s)"
        )

    chosen = select_linear_candidate(runs)
    coords = coords_by_candidate[chosen]
    chosen_k = candidates[chosen]
    print(f"selected {chosen} (k={chosen_k:.6f})")
    if args.no_write:
        return 0

    out_dir: Path = args.output_dir or DEFAULT_LINEAR_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = out_dir / "coordinates.csv"
    write_coordinates_csv(coordinates_path, data.ids, coords)
    layout_report = {
        "algorithm": "ratio_preserving_global_scale_spherical_stress_v1",
        "mode": "linear",
        "selected_k_candidate": chosen,
        "k": chosen_k,
        "k_candidates": candidates,
        "edge_target_policy": (
            "theta_target(i,j) = k * arccos(cos48(i,j)) for all C(602,2) pairs, "
            "clamped at pi; one global scalar k"
        ),
        "objective": (
            "sum over all 180901 unordered pairs of "
            "(theta_coord - theta_target)^2, uniform weight 1.0 per pair"
        ),
        "seed": config.seed,
        "multistart_count": config.multistart_count,
        "multistart_seeds": runs[chosen]["multistart_seeds"],
        "multistart_objectives": runs[chosen]["multistart_objectives"],
        "multistart_iterations": runs[chosen]["multistart_iterations"],
        "selected_start": runs[chosen]["selected_start"],
        "clamp": runs[chosen]["clamp"],
        "coordinate_sha256": coordinate_digest(data.ids, coords),
        **runs[chosen]["evaluation"],
    }
    with paths["baseline"].open() as handle:
        baseline_payload = json.load(handle)
    public = build_public_json(baseline_payload, data.ids, coords, layout_report)
    public_path = out_dir / "spherical-graph-public.json"
    with public_path.open("w") as handle:
        json.dump(public, handle, indent=2, sort_keys=True)

    payload = {
        "mode": "linear",
        "node_count": n,
        "pair_count": int(data.pair_angles.size),
        "selected_k_candidate": chosen,
        "selected_k": chosen_k,
        "selection_rule": LINEAR_SELECTION_RULE,
        "k_candidates": candidates,
        "k_ks_optimal_search": {
            "ks_at_optimum": ks_value,
            "method": "nested deterministic grid: 0.5..6.0 step 0.005, then "
            "3 bracketed refinements of 201 points",
        },
        "source_angle_summary": angle_summary(data.pair_angles),
        "candidates": runs,
        "cluster_policy": data.cluster_policy,
        "config": {
            "seed": config.seed,
            "multistart_count": config.multistart_count,
            "max_iterations": config.max_iterations,
            "probe_points": int(args.probe_points),
            "weighting_policy": "uniform weight 1.0 on every unordered pair",
        },
        "input_file_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "output_file_sha256": {
            "coordinates.csv": sha256_file(coordinates_path),
            "spherical-graph-public.json": sha256_file(public_path),
        },
        "coordinate_sha256": coordinate_digest(data.ids, coords),
        "database_reads": 0,
        "database_writes": 0,
        "network_calls": 0,
    }
    metrics_path = out_dir / "layout-metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"wrote {coordinates_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {public_path}")
    return 0


# ---------------------------------------------------------------------------
# convex mode driver
# ---------------------------------------------------------------------------

CONVEX_SELECTION_RULE = (
    "Two stages. STAGE 1 (filter, applied before any ranking): a candidate is "
    "eliminated if the realised near-field decile median stretch of its "
    "coordinates is below --min-near-decile-stretch, or if it misses any "
    "non-regression floor in NON_REGRESSION_FLOORS, or if it misses a revised "
    "hard acceptance gate in REVISED_ACCEPTANCE_GATES. Passing the analytic "
    "contract is not enough — the filter is evaluated on realised coordinates. "
    "STAGE 2 (rank the survivors): Borda rank sum over three axes, lowest total "
    "wins: (1) cluster separation ratio, higher is better; (2) mean Recall@5, "
    "higher is better; (3) coordinate pair-angle KS distance to the uniform "
    "sphere, lower is better. Ties break on the lower KS. If no candidate "
    "survives stage 1, nothing is selected: the run reports BLOCKED and emits "
    "the frontier instead of forcing a pick."
)


def borda_rank(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    """Rank-sum over ``separation`` (max), ``recall`` (max) and ``ks`` (min)."""

    names = sorted(rows)
    totals = {name: 0.0 for name in names}
    for key, descending in (("separation", True), ("recall", True), ("ks", False)):
        ordered = sorted(names, key=lambda name: rows[name][key], reverse=descending)
        for position, name in enumerate(ordered):
            totals[name] += position
    return totals


def matched_treatment(
    coords: np.ndarray,
    data: Dataset,
    true_top_k: np.ndarray,
    refine_config: RefineConfig,
    targets: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate a layout before and after the *same* refinement every row gets.

    The refinement optimises mean Recall@5 directly, so comparing a refined
    layout with unrefined ones is not a like-for-like comparison. Every row in
    the comparison table therefore receives an identical ``RefineConfig``.
    """

    before = evaluate_layout(coords, data, targets)
    refined, report = refine_recall_constrained(
        coords, true_top_k, data.labels, refine_config
    )
    after = evaluate_layout(refined, data, targets)
    return {
        "before_refinement": before,
        "after_refinement": after,
        "refinement": report,
        "coordinates": refined,
        "coordinates_before_refinement": coords,
    }


def _borda_axes(evaluation: dict[str, object]) -> dict[str, float]:
    return {
        "separation": float(evaluation["cluster"]["separation_ratio"]),
        "recall": float(evaluation["topk"]["mean_recall_at_5"]),
        "ks": float(evaluation["coverage"]["pair_angle_ks_vs_uniform_sphere"]),
    }


def run_convex_mode(args: argparse.Namespace) -> int:
    data, paths = load_dataset(args.artifact_dir, args.probe_points)
    n = len(data.ids)
    theta_max = float(np.max(data.pair_angles))
    true_top_k = data.order[:, :TOP_K]
    near_gate = float(args.min_near_decile_stretch)

    grid = [(a, gamma) for a in SWEEP_A for gamma in SWEEP_GAMMA]
    contracts: dict[str, dict] = {}
    rejected_contracts: dict[str, dict] = {}
    for a, gamma in grid:
        name = f"a{a}_g{gamma}"
        try:
            contracts[name] = convex_contract_report(a, gamma, theta_max)
        except ValueError as exc:
            rejected_contracts[name] = {
                "a": a,
                "gamma": gamma,
                "reason": str(exc),
                "a_upper_bound": math.pi / theta_max,
            }
            print(f"rejected {name}: {exc}")

    preliminary_config = LayoutConfig(
        seed=args.seed,
        multistart_count=args.preliminary_starts,
        max_iterations=args.preliminary_iterations,
    )
    full_config = LayoutConfig(
        seed=args.seed,
        multistart_count=args.starts,
        max_iterations=args.max_iterations,
    )
    refine_config = RefineConfig(
        seed=args.seed,
        max_passes=args.refine_passes,
        separation_max_relative_drop=args.separation_drop,
        ks_max_absolute_increase=args.ks_increase,
    )

    source_pairs = data.pair_angles
    preliminary: dict[str, dict] = {}
    axes: dict[str, dict[str, float]] = {}
    for name, contract in contracts.items():
        a = contract["a"]
        gamma = contract["gamma"]
        targets = convex_stretch(data.angles, a, gamma, theta_max)
        np.fill_diagonal(targets, 0.0)
        result = optimise_layout(targets, preliminary_config)
        coords = result.coordinates
        cluster = cluster_separation_metrics(coords, data.labels)
        observed = upper_triangle(coordinate_angle_matrix(coords))
        ks = ks_statistic_uniform_sphere(observed)
        profile = realised_stretch_by_source_decile(source_pairs, observed)
        near_median = float(profile[0]["median"])
        nearest_compressed = nearest_fraction_compressed(source_pairs, observed)
        dots = np.clip(coords @ coords.T, -1.0, 1.0)
        _, coord_order = similarity_rank_matrix(dots, data.ids)
        recall = recall_metrics(true_top_k, coord_order[:, :TOP_K])
        preliminary[name] = {
            "a": a,
            "gamma": gamma,
            "contract": contract,
            "separation_ratio": cluster["separation_ratio"],
            "silhouette_mean": cluster["silhouette_mean"],
            "centroid_angle_min": cluster["centroid_angle_min"],
            "pair_angle_ks_vs_uniform_sphere": ks,
            "covering_radius_radians": covering_radius(coords, data.probes),
            "near_decile_median_stretch": near_median,
            "nearest_5pct_fraction_compressed": nearest_compressed,
            "near_field_gate": near_gate,
            "near_field_gate_passed": bool(near_median >= near_gate),
            "realised_stretch_by_source_decile": profile,
            **recall,
            "objective": min(result.start_objectives),
            "degenerate_fallback_count": count_degenerate_rows(coords),
            "elapsed_seconds": result.elapsed_seconds,
        }
        axes[name] = {
            "separation": cluster["separation_ratio"],
            "recall": recall["mean_recall_at_5"],
            "ks": ks,
        }
        print(
            f"prelim {name}: sep={cluster['separation_ratio']:.4f} "
            f"recall@5={recall['mean_recall_at_5']:.4f} ks={ks:.4f} "
            f"near_stretch={near_median:.4f} "
            f"near5%_compressed={nearest_compressed:.4f} "
            f"gate={'pass' if near_median >= near_gate else 'FAIL'} "
            f"({result.elapsed_seconds:.1f}s)"
        )

    eligible = [
        name for name in preliminary if preliminary[name]["near_field_gate_passed"]
    ]
    near_gate_eliminated_all = not eligible
    pool = eligible or sorted(preliminary)
    totals = borda_rank({name: axes[name] for name in pool})
    shortlist = sorted(pool, key=lambda name: (totals[name], axes[name]["ks"]))[
        : args.shortlist
    ]
    if near_gate_eliminated_all:
        print(
            "near-field gate eliminated every candidate; continuing with a "
            "best-effort frontier shortlist (final status will be BLOCKED unless "
            "the full runs clear the floors)"
        )
    print(f"shortlist: {shortlist}")

    finalists: dict[str, dict] = {}
    coords_by_name: dict[str, np.ndarray] = {}
    before_coords_by_name: dict[str, np.ndarray] = {}
    targets_by_name: dict[str, np.ndarray] = {}
    for name in shortlist:
        a = preliminary[name]["a"]
        gamma = preliminary[name]["gamma"]
        targets = convex_stretch(data.angles, a, gamma, theta_max)
        np.fill_diagonal(targets, 0.0)
        result = optimise_layout(targets, full_config)
        treated = matched_treatment(
            result.coordinates, data, true_top_k, refine_config, targets
        )
        after = treated["after_refinement"]
        coords_by_name[name] = treated["coordinates"]
        before_coords_by_name[name] = result.coordinates
        targets_by_name[name] = targets
        row = gate_row(after)
        finalists[name] = {
            "a": a,
            "gamma": gamma,
            "contract": contracts[name],
            "target_monotonicity_violations": target_monotonicity_violations(
                data.pair_angles, upper_triangle(targets)
            ),
            "before_refinement": treated["before_refinement"],
            "after_refinement": after,
            "refinement": treated["refinement"],
            "gate_row": row,
            "non_regression": check_thresholds(row, NON_REGRESSION_FLOORS),
            "revised_gates": check_thresholds(row, REVISED_ACCEPTANCE_GATES),
            "original_gates": check_thresholds(row, ORIGINAL_ACCEPTANCE_GATES),
            "separation_targets": check_thresholds(row, SEPARATION_TARGETS),
            "multistart_objectives": result.start_objectives,
            "multistart_seeds": result.seeds,
            "multistart_start_kinds": result.start_kinds,
            "multistart_iterations": result.iterations,
            "selected_start": result.selected_start,
            "elapsed_seconds": result.elapsed_seconds,
        }
        print(
            f"final {name}: recall@5 "
            f"{treated['before_refinement']['topk']['mean_recall_at_5']:.4f} -> "
            f"{row['mean_recall_at_5']:.4f} sep {row['separation_ratio']:.4f} "
            f"sil {row['silhouette_mean']:.4f} ks "
            f"{row['pair_angle_ks_vs_uniform_sphere']:.4f} near "
            f"{row['near_decile_median_stretch']:.4f} near5% "
            f"{row['nearest_5pct_fraction_compressed']:.4f} floors "
            f"{'PASS' if finalists[name]['non_regression']['passed'] else 'FAIL'} "
            f"{finalists[name]['non_regression']['failed']}"
        )

    qualified = [
        name
        for name in shortlist
        if finalists[name]["non_regression"]["passed"]
        and finalists[name]["revised_gates"]["passed"]
    ]
    final_axes = {
        name: _borda_axes(finalists[name]["after_refinement"]) for name in shortlist
    }
    final_totals = borda_rank(final_axes)
    if qualified:
        chosen = sorted(
            {name: final_totals[name] for name in qualified},
            key=lambda name: (final_totals[name], final_axes[name]["ks"]),
        )[0]
        status = "SELECTED"
    else:
        chosen = None
        status = "BLOCKED"
    frontier = {
        name: {
            **finalists[name]["gate_row"],
            "non_regression_failed": finalists[name]["non_regression"]["failed"],
            "revised_gates_failed": finalists[name]["revised_gates"]["failed"],
            "separation_targets_failed": finalists[name]["separation_targets"][
                "failed"
            ],
            "borda_total": final_totals[name],
        }
        for name in shortlist
    }
    print(f"status {status}; selected {chosen}")

    # reference layouts — identical code path and identical refinement budget
    linear_k = k_max_antipodal(data.pair_angles)
    linear_targets = scaled_targets(data.angles, linear_k)
    np.fill_diagonal(linear_targets, 0.0)
    linear_result = optimise_layout(linear_targets, full_config)
    baseline_coords = coordinates_in_id_order(
        load_public_json_coordinates(paths["baseline"]), data.ids
    )
    rankscaled_path = args.rankscaled_dir / "coordinates.csv"
    rankscaled_coords = coordinates_in_id_order(
        load_coordinates_csv(rankscaled_path), data.ids
    )
    reference_layouts = {
        "baseline": (baseline_coords, None, str(paths["baseline"])),
        "rank_scaled": (rankscaled_coords, None, str(rankscaled_path)),
        "linear_k_max_antipodal": (
            linear_result.coordinates,
            linear_targets,
            "persisted by this run",
        ),
    }
    linear_v1_path = DEFAULT_LINEAR_OUTPUT_DIR / "coordinates.csv"
    if linear_v1_path.exists():
        reference_layouts["linear_v1_on_disk_k_median_match"] = (
            coordinates_in_id_order(load_coordinates_csv(linear_v1_path), data.ids),
            None,
            str(linear_v1_path),
        )

    comparison: dict[str, dict] = {}
    comparison_coords: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (layout, targets, source) in reference_layouts.items():
        treated = matched_treatment(layout, data, true_top_k, refine_config, targets)
        comparison[label] = {
            "source": source,
            "before_refinement": treated["before_refinement"],
            "after_refinement": treated["after_refinement"],
            "refinement": treated["refinement"],
            "gate_row_before": gate_row(treated["before_refinement"]),
            "gate_row_after": gate_row(treated["after_refinement"]),
        }
        comparison_coords[label] = (layout, treated["coordinates"])
        print(
            f"matched {label}: recall@5 "
            f"{comparison[label]['gate_row_before']['mean_recall_at_5']:.4f} -> "
            f"{comparison[label]['gate_row_after']['mean_recall_at_5']:.4f}"
        )
    comparison["linear_k_max_antipodal"]["k"] = linear_k
    for name in shortlist:
        comparison[f"convex_{name}"] = {
            "source": "this run",
            "before_refinement": finalists[name]["before_refinement"],
            "after_refinement": finalists[name]["after_refinement"],
            "refinement": finalists[name]["refinement"],
            "gate_row_before": gate_row(finalists[name]["before_refinement"]),
            "gate_row_after": finalists[name]["gate_row"],
        }

    reference = uniform_reference(
        n, data.probes, replicates=args.uniform_replicates, seed=args.seed
    )

    if args.no_write:
        print(json.dumps(frontier, indent=2, sort_keys=True, default=str))
        return 0

    out_dir: Path = args.output_dir or DEFAULT_CONVEX_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for label, (raw, refined) in comparison_coords.items():
        path = out_dir / f"{label.replace('_', '-')}-coordinates.csv"
        write_coordinates_csv(path, data.ids, raw)
        written[path.name] = path
        refined_path = out_dir / f"{label.replace('_', '-')}-coordinates-refined.csv"
        write_coordinates_csv(refined_path, data.ids, refined)
        written[refined_path.name] = refined_path
    for name in shortlist:
        path = out_dir / f"coordinates-{name}-before-refinement.csv"
        write_coordinates_csv(path, data.ids, before_coords_by_name[name])
        written[path.name] = path
        path = out_dir / f"coordinates-{name}-after-refinement.csv"
        write_coordinates_csv(path, data.ids, coords_by_name[name])
        written[path.name] = path

    sources = {
        "baseline": str(paths["baseline"]),
        "rank_scaled": str(rankscaled_path),
        "linear_k_max_antipodal": str(
            out_dir / "linear-k-max-antipodal-coordinates.csv"
        ),
    }
    if linear_v1_path.exists():
        sources["linear_v1_on_disk_k_median_match"] = str(linear_v1_path)

    public_path = None
    coordinates_path = None
    if chosen is not None:
        coordinates_path = out_dir / "coordinates.csv"
        write_coordinates_csv(coordinates_path, data.ids, coords_by_name[chosen])
        written[coordinates_path.name] = coordinates_path
        before_path = out_dir / "coordinates-before-refinement.csv"
        write_coordinates_csv(before_path, data.ids, before_coords_by_name[chosen])
        written[before_path.name] = before_path
        layout_report = {
            "algorithm": "convex_monotone_stretch_spherical_stress_v2",
            "mode": "convex",
            "status": status,
            "selected_combination": chosen,
            "a": finalists[chosen]["a"],
            "gamma": finalists[chosen]["gamma"],
            "b": contracts[chosen]["b"],
            "theta_max": theta_max,
            "edge_target_policy": (
                "theta_target(i,j) = a*angle + b*angle**gamma with "
                "b = (pi - a*theta_max)/theta_max**gamma, applied to all "
                "C(602,2) pairs; strictly increasing, convex, f(theta) > theta "
                "everywhere, f(theta_max) = pi"
            ),
            "objective": (
                "sum over all 180901 unordered pairs of "
                "(theta_coord - theta_target)^2, uniform weight 1.0 per pair, "
                "followed by a constrained greedy refinement on mean Recall@5"
            ),
            "contract": contracts[chosen],
            "target_monotonicity_violations": finalists[chosen][
                "target_monotonicity_violations"
            ],
            "seed": full_config.seed,
            "multistart_count": full_config.multistart_count,
            "multistart_seeds": finalists[chosen]["multistart_seeds"],
            "multistart_objectives": finalists[chosen]["multistart_objectives"],
            "multistart_iterations": finalists[chosen]["multistart_iterations"],
            "selected_start": finalists[chosen]["selected_start"],
            "refinement": finalists[chosen]["refinement"],
            "non_regression": finalists[chosen]["non_regression"],
            "revised_gates": finalists[chosen]["revised_gates"],
            "original_gates": finalists[chosen]["original_gates"],
            "separation_targets": finalists[chosen]["separation_targets"],
            "cluster_policy": data.cluster_policy,
            "coordinate_sha256": coordinate_digest(data.ids, coords_by_name[chosen]),
            **finalists[chosen]["after_refinement"],
        }
        with paths["baseline"].open() as handle:
            baseline_payload = json.load(handle)
        public = build_public_json(
            baseline_payload, data.ids, coords_by_name[chosen], layout_report
        )
        public_path = out_dir / "spherical-graph-public.json"
        with public_path.open("w") as handle:
            json.dump(public, handle, indent=2, sort_keys=True)
        written[public_path.name] = public_path

    procedure = (
        f"contract check on all {len(grid)} (a, gamma) combinations "
        f"({len(rejected_contracts)} rejected for a >= pi/theta_max), "
        f"preliminary sweep of the {len(contracts)} accepted ones with "
        f"{args.preliminary_starts} starts and {args.preliminary_iterations} "
        f"L-BFGS-B iterations, near-field filter at {near_gate}, then the top "
        f"{args.shortlist} by the selection rule rerun with {args.starts} starts "
        f"and {args.max_iterations} iterations plus the constrained recall "
        "refinement; every comparison row receives the identical refinement"
    )
    sweep_payload = {
        "grid": {"a": list(SWEEP_A), "gamma": list(SWEEP_GAMMA)},
        "procedure": procedure,
        "selection_rule": CONVEX_SELECTION_RULE,
        "status": status,
        "contracts": contracts,
        "rejected_contracts": rejected_contracts,
        "preliminary": preliminary,
        "preliminary_borda_totals": totals,
        "near_field_gate": near_gate,
        "near_field_gate_eliminated_all": near_gate_eliminated_all,
        "shortlist": shortlist,
        "finalists": finalists,
        "final_borda_totals": final_totals,
        "frontier": frontier,
        "qualified": qualified,
        "selected_combination": chosen,
        "thresholds": {
            "original_acceptance_gates": ORIGINAL_ACCEPTANCE_GATES,
            "revised_acceptance_gates": REVISED_ACCEPTANCE_GATES,
            "non_regression_floors": NON_REGRESSION_FLOORS,
            "separation_targets": SEPARATION_TARGETS,
        },
        "database_reads": 0,
        "database_writes": 0,
        "network_calls": 0,
    }
    sweep_path = out_dir / "sweep-metrics.json"
    with sweep_path.open("w") as handle:
        json.dump(sweep_payload, handle, indent=2, sort_keys=True, default=str)
    written[sweep_path.name] = sweep_path

    frontier_path = out_dir / "frontier.json"
    with frontier_path.open("w") as handle:
        json.dump(
            {
                "status": status,
                "selected_combination": chosen,
                "frontier": frontier,
                "thresholds": {
                    "non_regression_floors": NON_REGRESSION_FLOORS,
                    "revised_acceptance_gates": REVISED_ACCEPTANCE_GATES,
                    "separation_targets": SEPARATION_TARGETS,
                    "near_field_gate": near_gate,
                },
            },
            handle,
            indent=2,
            sort_keys=True,
            default=str,
        )
    written[frontier_path.name] = frontier_path

    payload = {
        "mode": "convex",
        "status": status,
        "node_count": n,
        "pair_count": int(data.pair_angles.size),
        "theta_max": theta_max,
        "a_upper_bound": math.pi / theta_max,
        "selected_combination": chosen,
        "selected_a": finalists[chosen]["a"] if chosen else None,
        "selected_gamma": finalists[chosen]["gamma"] if chosen else None,
        "selected_b": contracts[chosen]["b"] if chosen else None,
        "selection_rule": CONVEX_SELECTION_RULE,
        "procedure": procedure,
        "contracts": contracts,
        "rejected_contracts": rejected_contracts,
        "preliminary": preliminary,
        "preliminary_borda_totals": totals,
        "near_field_gate": near_gate,
        "near_field_gate_eliminated_all": near_gate_eliminated_all,
        "shortlist": shortlist,
        "finalists": finalists,
        "final_borda_totals": final_totals,
        "frontier": frontier,
        "qualified": qualified,
        "comparison": comparison,
        "uniform_sphere_reference": reference,
        "cluster_policy": data.cluster_policy,
        "source_angle_summary": angle_summary(data.pair_angles),
        "sources": sources,
        "thresholds": {
            "original_acceptance_gates": ORIGINAL_ACCEPTANCE_GATES,
            "revised_acceptance_gates": REVISED_ACCEPTANCE_GATES,
            "non_regression_floors": NON_REGRESSION_FLOORS,
            "separation_targets": SEPARATION_TARGETS,
        },
        "config": {
            "seed": full_config.seed,
            "multistart_count": full_config.multistart_count,
            "max_iterations": full_config.max_iterations,
            "preliminary_starts": args.preliminary_starts,
            "preliminary_iterations": args.preliminary_iterations,
            "shortlist": args.shortlist,
            "min_near_decile_stretch": near_gate,
            "ftol": full_config.ftol,
            "gtol": full_config.gtol,
            "dot_clip": DOT_CLIP,
            "probe_points": int(args.probe_points),
            "uniform_replicates": int(args.uniform_replicates),
            "weighting_policy": "uniform weight 1.0 on every unordered pair",
            "refinement": {
                "seed": refine_config.seed,
                "max_passes": refine_config.max_passes,
                "pull_fractions": list(refine_config.pull_fractions),
                "step_radii": list(refine_config.step_radii),
                "directions_per_radius": refine_config.directions_per_radius,
                "separation_max_relative_drop": (
                    refine_config.separation_max_relative_drop
                ),
                "ks_max_absolute_increase": refine_config.ks_max_absolute_increase,
            },
        },
        "input_file_sha256": {
            **{key: sha256_file(path) for key, path in paths.items()},
            "rankscaled-coordinates.csv": sha256_file(rankscaled_path),
            **(
                {"linear-v1-coordinates.csv": sha256_file(linear_v1_path)}
                if linear_v1_path.exists()
                else {}
            ),
        },
        "output_file_sha256": {
            name: sha256_file(path) for name, path in sorted(written.items())
        },
        "coordinate_sha256": (
            coordinate_digest(data.ids, coords_by_name[chosen]) if chosen else None
        ),
        "database_reads": 0,
        "database_writes": 0,
        "network_calls": 0,
    }
    metrics_path = out_dir / "layout-metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    for name in sorted(written):
        print(f"wrote {written[name]}")
    print(f"wrote {metrics_path}")

    if chosen is not None and args.durable_copy_dir is not None:
        durable: Path = args.durable_copy_dir
        durable.mkdir(parents=True, exist_ok=True)
        for path in (coordinates_path, public_path, sweep_path, metrics_path):
            if path is not None:
                shutil.copyfile(path, durable / path.name)
                print(f"copied {path.name} -> {durable}")
    elif chosen is None:
        print(
            "BLOCKED: no candidate cleared the non-regression floors; "
            "no canonical coordinates.csv and no durable copy were written"
        )
    return 0


# ---------------------------------------------------------------------------
# frontier mode driver
# ---------------------------------------------------------------------------

FRONTIER_RULE = (
    "The frontier is the deliverable, not a single winner. Every experiment "
    "point is computed and reported; the non-regression numbers are reference "
    "lines, not filters. Only three things are hard: unit_norm_max_error <= "
    "1e-12, zero target rank inversions, and the user's near-field requirement "
    "(near-decile median realised stretch >= 1.00 and nearest-5% compressed "
    "fraction <= 0.20). Points failing a hard gate stay in the table, flagged."
)

FRONTIER_A: tuple[float, ...] = (1.4, 1.6, 1.8, 2.0, 2.1)
FRONTIER_GAMMA: tuple[float, ...] = (1.5, 2.0)
FRONTIER_SCALES: tuple[float, ...] = (1.15, 1.3, 1.5, 1.75, 2.0)
FRONTIER_SCALE_BASE: tuple[float, float] = (2.0, 1.5)
FRONTIER_CENTERING_BASE: tuple[float, float] = (2.0, 1.5)
#: Combined levers: global scale together with the centering penalty.
FRONTIER_COMBINED_SCALES: tuple[float, ...] = (1.3, 1.5)
FRONTIER_COMBINED_CENTERING: tuple[float, ...] = (400000.0,)

#: Hard gates. Everything else is a reference line in this mode.
HARD_GATES: dict[str, tuple[str, float]] = {
    "unit_norm_max_error": ("<=", 1e-12),
    "near_decile_median_stretch": (">=", 1.00),
    "nearest_5pct_fraction_compressed": ("<=", 0.20),
}


def build_frontier_specs(
    data: Dataset, theta_max: float, centering_weights: Sequence[float]
) -> tuple[list[dict], dict[str, dict]]:
    """All experiment points: convex grid, global scales, rank-area, centering."""

    specs: list[dict] = []
    rejected: dict[str, dict] = {}
    for a in FRONTIER_A:
        for gamma in FRONTIER_GAMMA:
            label = f"convex-a{a}-g{gamma}"
            try:
                contract = convex_contract_report(a, gamma, theta_max)
            except ValueError as exc:
                rejected[label] = {
                    "a": a,
                    "gamma": gamma,
                    "reason": str(exc),
                    "a_upper_bound": math.pi / theta_max,
                }
                print(f"rejected {label}: {exc}")
                continue
            targets = convex_stretch(data.angles, a, gamma, theta_max)
            np.fill_diagonal(targets, 0.0)
            specs.append(
                {
                    "label": label,
                    "family": "convex",
                    "a": a,
                    "gamma": gamma,
                    "scale": 1.0,
                    "centering_weight": 0.0,
                    "contract": contract,
                    "targets": targets,
                    "clamp": scale_clamp_report(targets, 1.0),
                }
            )

    base_a, base_gamma = FRONTIER_SCALE_BASE
    base_targets = convex_stretch(data.angles, base_a, base_gamma, theta_max)
    np.fill_diagonal(base_targets, 0.0)
    for scale in FRONTIER_SCALES:
        scaled = globally_scaled_targets(base_targets, scale)
        np.fill_diagonal(scaled, 0.0)
        specs.append(
            {
                "label": f"convex-a{base_a}-g{base_gamma}-s{scale}",
                "family": "convex_global_scale",
                "a": base_a,
                "gamma": base_gamma,
                "scale": scale,
                "centering_weight": 0.0,
                "contract": convex_contract_report(base_a, base_gamma, theta_max),
                "targets": scaled,
                "clamp": scale_clamp_report(base_targets, scale),
            }
        )

    rank_targets = rank_uniform_area_targets(data.ranks)
    specs.append(
        {
            "label": "rank-uniform-area",
            "family": "rank_uniform_area",
            "a": None,
            "gamma": None,
            "scale": 1.0,
            "centering_weight": 0.0,
            "contract": None,
            "targets": rank_targets,
            "clamp": scale_clamp_report(rank_targets, 1.0),
        }
    )

    for scale in FRONTIER_COMBINED_SCALES:
        for weight in FRONTIER_COMBINED_CENTERING:
            scaled = globally_scaled_targets(base_targets, scale)
            np.fill_diagonal(scaled, 0.0)
            specs.append(
                {
                    "label": f"convex-a{base_a}-g{base_gamma}-s{scale}-centering{weight:g}",
                    "family": "convex_scale_plus_centering",
                    "a": base_a,
                    "gamma": base_gamma,
                    "scale": scale,
                    "centering_weight": float(weight),
                    "contract": convex_contract_report(base_a, base_gamma, theta_max),
                    "targets": scaled,
                    "clamp": scale_clamp_report(base_targets, scale),
                }
            )

    centre_a, centre_gamma = FRONTIER_CENTERING_BASE
    centre_targets = convex_stretch(data.angles, centre_a, centre_gamma, theta_max)
    np.fill_diagonal(centre_targets, 0.0)
    for weight in centering_weights:
        specs.append(
            {
                "label": f"convex-a{centre_a}-g{centre_gamma}-centering{weight:g}",
                "family": "convex_centering_penalty",
                "a": centre_a,
                "gamma": centre_gamma,
                "scale": 1.0,
                "centering_weight": float(weight),
                "contract": convex_contract_report(centre_a, centre_gamma, theta_max),
                "targets": centre_targets,
                "clamp": scale_clamp_report(centre_targets, 1.0),
            }
        )
    return specs, rejected


def _frontier_row(
    label: str,
    spec: dict | None,
    treated: dict,
    monotonicity_violations: int | None,
) -> dict[str, object]:
    before = gate_row(treated["before_refinement"])
    after = gate_row(treated["after_refinement"])
    hard = check_thresholds(after, HARD_GATES)
    hard_before = check_thresholds(before, HARD_GATES)
    return {
        "label": label,
        "family": spec["family"] if spec else "reference_layout",
        "a": spec["a"] if spec else None,
        "gamma": spec["gamma"] if spec else None,
        "scale": spec["scale"] if spec else None,
        "centering_weight": spec["centering_weight"] if spec else None,
        "clamped_pair_count": (spec["clamp"]["clamped_pair_count"] if spec else None),
        "clamped_pair_fraction": (
            spec["clamp"]["clamped_pair_fraction"] if spec else None
        ),
        "target_monotonicity_violations": monotonicity_violations,
        "before_refinement": before,
        "after_refinement": after,
        "hard_gates_after": hard,
        "hard_gates_before": hard_before,
        "reference_lines_after": check_thresholds(after, NON_REGRESSION_FLOORS),
        "separation_targets_after": check_thresholds(after, SEPARATION_TARGETS),
        "refinement": treated["refinement"],
    }


def run_frontier_mode(args: argparse.Namespace) -> int:
    data, paths = load_dataset(args.artifact_dir, args.probe_points)
    n = len(data.ids)
    theta_max = float(np.max(data.pair_angles))
    true_top_k = data.order[:, :TOP_K]
    layout_config = LayoutConfig(
        seed=args.seed,
        multistart_count=args.frontier_starts,
        max_iterations=args.max_iterations,
    )
    refine_config = RefineConfig(
        seed=args.seed,
        max_passes=args.refine_passes,
        separation_max_relative_drop=args.separation_drop,
        ks_max_absolute_increase=args.ks_increase,
    )
    meannorm_refine_config = RefineConfig(
        seed=args.seed,
        max_passes=args.refine_passes,
        separation_max_relative_drop=args.separation_drop,
        constraint_mode="mean_vector_norm",
        mean_vector_norm_max_absolute_increase=args.mean_norm_increase,
    )

    specs, rejected = build_frontier_specs(data, theta_max, args.centering_weights)
    rows: dict[str, dict] = {}
    coords_by_label: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    optimisation: dict[str, dict] = {}

    for spec in specs:
        label = spec["label"]
        result = optimise_layout(
            spec["targets"], layout_config, spec["centering_weight"]
        )
        treated = matched_treatment(
            result.coordinates, data, true_top_k, refine_config, spec["targets"]
        )
        violations = target_monotonicity_violations(
            data.pair_angles, upper_triangle(spec["targets"])
        )
        rows[label] = _frontier_row(label, spec, treated, violations)
        coords_by_label[label] = (result.coordinates, treated["coordinates"])
        optimisation[label] = {
            "multistart_objectives": result.start_objectives,
            "multistart_seeds": result.seeds,
            "multistart_iterations": result.iterations,
            "selected_start": result.selected_start,
            "elapsed_seconds": result.elapsed_seconds,
            "degenerate_fallback_count": count_degenerate_rows(result.coordinates),
        }
        row = rows[label]
        print(
            f"{label}: bias {row['before_refinement']['mean_vector_norm']:.4f}->"
            f"{row['after_refinement']['mean_vector_norm']:.4f} "
            f"hemi {row['after_refinement']['best_hemisphere_fraction']:.3f} "
            f"cover {row['after_refinement']['covering_radius_radians']:.4f} "
            f"empty {row['after_refinement']['empty_cells_100']:.0f} "
            f"ks {row['after_refinement']['pair_angle_ks_vs_uniform_sphere']:.4f} "
            f"recall {row['before_refinement']['mean_recall_at_5']:.4f}->"
            f"{row['after_refinement']['mean_recall_at_5']:.4f} "
            f"near {row['after_refinement']['near_decile_median_stretch']:.4f}/"
            f"{row['after_refinement']['nearest_5pct_fraction_compressed']:.4f} "
            f"hard {'PASS' if row['hard_gates_after']['passed'] else row['hard_gates_after']['failed']}"
        )

    # alternative refinement guard on the scale base point
    base_label = f"convex-a{FRONTIER_SCALE_BASE[0]}-g{FRONTIER_SCALE_BASE[1]}"
    if base_label in coords_by_label:
        spec = next(item for item in specs if item["label"] == base_label)
        alt_label = f"{base_label}-refine-meannorm"
        treated = matched_treatment(
            coords_by_label[base_label][0],
            data,
            true_top_k,
            meannorm_refine_config,
            spec["targets"],
        )
        rows[alt_label] = _frontier_row(
            alt_label,
            {**spec, "family": "convex_refine_meannorm_guard"},
            treated,
            rows[base_label]["target_monotonicity_violations"],
        )
        coords_by_label[alt_label] = (
            coords_by_label[base_label][0],
            treated["coordinates"],
        )
        print(
            f"{alt_label}: bias "
            f"{rows[alt_label]['after_refinement']['mean_vector_norm']:.4f} "
            f"recall {rows[alt_label]['after_refinement']['mean_recall_at_5']:.4f} "
            f"(KS guard gave "
            f"{rows[base_label]['after_refinement']['mean_vector_norm']:.4f} / "
            f"{rows[base_label]['after_refinement']['mean_recall_at_5']:.4f})"
        )

    # existing layouts folded in as frontier points, same matched treatment
    references: dict[str, tuple[np.ndarray, str]] = {}
    references["baseline"] = (
        coordinates_in_id_order(
            load_public_json_coordinates(paths["baseline"]), data.ids
        ),
        str(paths["baseline"]),
    )
    rankscaled_path = args.rankscaled_dir / "coordinates.csv"
    references["rank-scaled"] = (
        coordinates_in_id_order(load_coordinates_csv(rankscaled_path), data.ids),
        str(rankscaled_path),
    )
    linear_v1 = DEFAULT_LINEAR_OUTPUT_DIR / "coordinates.csv"
    if linear_v1.exists():
        references["linear-ratio-v1"] = (
            coordinates_in_id_order(load_coordinates_csv(linear_v1), data.ids),
            str(linear_v1),
        )
    convex_v1 = Path("/private/tmp/cocktail-mate-s2-convex-602-v1/coordinates.csv")
    if convex_v1.exists():
        references["convex-v1-a1.4-g1.5"] = (
            coordinates_in_id_order(load_coordinates_csv(convex_v1), data.ids),
            str(convex_v1),
        )
    reference_sources = {label: source for label, (_, source) in references.items()}
    for label, (layout, _) in references.items():
        treated = matched_treatment(layout, data, true_top_k, refine_config, None)
        rows[label] = _frontier_row(label, None, treated, None)
        coords_by_label[label] = (layout, treated["coordinates"])
        row = rows[label]
        print(
            f"{label}: bias {row['before_refinement']['mean_vector_norm']:.4f}->"
            f"{row['after_refinement']['mean_vector_norm']:.4f} "
            f"recall {row['before_refinement']['mean_recall_at_5']:.4f}->"
            f"{row['after_refinement']['mean_recall_at_5']:.4f}"
        )

    reference_uniform = uniform_reference(
        n, data.probes, replicates=args.uniform_replicates, seed=args.seed
    )

    if args.no_write:
        return 0

    out_dir: Path = args.output_dir or DEFAULT_CONVEX_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for label, (raw, refined) in coords_by_label.items():
        path = out_dir / f"coordinates-{label}.csv"
        write_coordinates_csv(path, data.ids, refined)
        written[path.name] = path
        raw_path = out_dir / f"coordinates-{label}-unrefined.csv"
        write_coordinates_csv(raw_path, data.ids, raw)
        written[raw_path.name] = raw_path

    payload = {
        "mode": "frontier",
        "rule": FRONTIER_RULE,
        "node_count": n,
        "pair_count": int(data.pair_angles.size),
        "theta_max": theta_max,
        "a_upper_bound": math.pi / theta_max,
        "grid": {
            "a": list(FRONTIER_A),
            "gamma": list(FRONTIER_GAMMA),
            "global_scales": list(FRONTIER_SCALES),
            "global_scale_base": list(FRONTIER_SCALE_BASE),
            "centering_weights": list(args.centering_weights),
            "centering_base": list(FRONTIER_CENTERING_BASE),
        },
        "rejected_contracts": rejected,
        "frontier": rows,
        "optimisation": optimisation,
        "reference_sources": reference_sources,
        "uniform_sphere_reference": reference_uniform,
        "cluster_policy": data.cluster_policy,
        "thresholds": {
            "hard_gates": HARD_GATES,
            "reference_lines": NON_REGRESSION_FLOORS,
            "separation_targets": SEPARATION_TARGETS,
            "original_acceptance_gates": ORIGINAL_ACCEPTANCE_GATES,
            "revised_acceptance_gates": REVISED_ACCEPTANCE_GATES,
        },
        "config": {
            "seed": layout_config.seed,
            "multistart_count": layout_config.multistart_count,
            "max_iterations": layout_config.max_iterations,
            "probe_points": int(args.probe_points),
            "uniform_replicates": int(args.uniform_replicates),
            "refinement": {
                "seed": refine_config.seed,
                "max_passes": refine_config.max_passes,
                "constraint_mode": refine_config.constraint_mode,
                "separation_max_relative_drop": (
                    refine_config.separation_max_relative_drop
                ),
                "ks_max_absolute_increase": refine_config.ks_max_absolute_increase,
                "mean_vector_norm_max_absolute_increase": (
                    meannorm_refine_config.mean_vector_norm_max_absolute_increase
                ),
            },
        },
        "input_file_sha256": {
            **{key: sha256_file(path) for key, path in paths.items()},
            "rankscaled-coordinates.csv": sha256_file(rankscaled_path),
        },
        "output_file_sha256": {
            name: sha256_file(path) for name, path in sorted(written.items())
        },
        "database_reads": 0,
        "database_writes": 0,
        "network_calls": 0,
    }
    metrics_path = out_dir / "frontier-metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    print(f"wrote {metrics_path} and {len(written)} coordinate files")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("convex", "linear", "frontier"), default="frontier"
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--rankscaled-dir", type=Path, default=DEFAULT_RANKSCALED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--preliminary-starts", type=int, default=2)
    parser.add_argument("--preliminary-iterations", type=int, default=400)
    parser.add_argument("--shortlist", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--refine-passes", type=int, default=60)
    parser.add_argument("--separation-drop", type=float, default=0.01)
    parser.add_argument("--ks-increase", type=float, default=0.005)
    parser.add_argument(
        "--min-near-decile-stretch",
        type=float,
        default=1.00,
        help=(
            "realised near-field decile median stretch a candidate must reach "
            "before it may be ranked; guards the user's 'near pairs open up a "
            "little' requirement at the coordinate level, not just in f"
        ),
    )
    parser.add_argument(
        "--durable-copy-dir", type=Path, default=DEFAULT_DURABLE_COPY_DIR
    )
    parser.add_argument("--probe-points", type=int, default=DEFAULT_PROBE_POINTS)
    parser.add_argument(
        "--uniform-replicates", type=int, default=DEFAULT_UNIFORM_REPLICATES
    )
    parser.add_argument("--frontier-starts", type=int, default=4)
    parser.add_argument("--mean-norm-increase", type=float, default=0.005)
    parser.add_argument(
        "--centering-weights",
        type=float,
        nargs="+",
        default=[10000.0, 100000.0, 400000.0, 800000.0],
    )
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "linear":
        return run_linear_mode(args)
    if args.mode == "convex":
        return run_convex_mode(args)
    return run_frontier_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
