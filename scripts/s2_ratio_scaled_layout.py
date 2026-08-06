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
separation goals are not traded away for recall.

Usage::

    python scripts/s2_ratio_scaled_layout.py --mode convex
    python scripts/s2_ratio_scaled_layout.py --mode linear
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
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

DEFAULT_ARTIFACT_DIR = Path("/private/tmp/cocktail-mate-sensory-artifacts-602-v1")
DEFAULT_RANKSCALED_DIR = Path("/private/tmp/cocktail-mate-s2-rankscaled-602-v1")
DEFAULT_LINEAR_OUTPUT_DIR = Path("/private/tmp/cocktail-mate-s2-ratio-602-v1")
DEFAULT_CONVEX_OUTPUT_DIR = Path("/private/tmp/cocktail-mate-s2-convex-602-v1")
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
#: Convex sweep grid mandated by the brief.
SWEEP_A: tuple[float, ...] = (1.1, 1.2, 1.4, 1.6, 1.8)
SWEEP_GAMMA: tuple[float, ...] = (1.5, 2.0, 3.0)
#: Deterministic sample size for the coordinate pair-order inversion estimate.
INVERSION_SAMPLES = 2_000_000


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


def normalise_rows(coords: np.ndarray) -> np.ndarray:
    """Project rows onto the unit sphere; zero rows fall back to the pole."""

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


def make_objective(targets: np.ndarray):
    """Objective and gradient of the unweighted all-pairs angle stress.

    Every one of the ``C(n, 2)`` pairs carries weight 1.0 — no rank weighting, no
    edge/non-edge distinction — because the stretch treats all pairs alike.
    Coordinates are parametrised by unnormalised rows normalised inside, so
    L-BFGS-B runs unconstrained.
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


def optimise_layout(targets: np.ndarray, config: LayoutConfig) -> LayoutResult:
    """Multistart L-BFGS-B; start 0 is spectral, the rest are seeded gaussians."""

    started = time.perf_counter()
    n = targets.shape[0]
    objective = make_objective(targets)
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
    cells: dict[str, list[dict[str, float]]] = {
        f"cells_{b * s}": [] for b, s in CELL_LAYOUTS
    }
    for _ in range(replicates):
        coords = normalise_rows(rng.standard_normal((node_count, 3)))
        nn.append(angle_summary(nearest_neighbour_angles(coords)))
        covering.append(covering_radius(coords, probes))
        ks.append(
            ks_statistic_uniform_sphere(upper_triangle(coordinate_angle_matrix(coords)))
        )
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


def order_fidelity_metrics(
    coords: np.ndarray,
    angles: np.ndarray,
    targets: np.ndarray,
    *,
    inversion_samples: int = INVERSION_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Section B: how faithfully the coordinates keep the source angle order."""

    observed = upper_triangle(coordinate_angle_matrix(coords))
    source = upper_triangle(angles)
    target = upper_triangle(targets)
    ratios = observed / source
    residual = observed - target
    stress = float(np.sum(residual * residual))
    denominator = float(np.sum(target * target))
    return {
        "spearman_coord_vs_source_angle": float(spearmanr(observed, source).statistic),
        "spearman_coord_vs_target": float(spearmanr(observed, target).statistic),
        "pearson_coord_vs_target": float(np.corrcoef(observed, target)[0, 1]),
        "pearson_coord_vs_source_angle": float(np.corrcoef(observed, source)[0, 1]),
        "coordinate_pair_order_inversions": pair_order_inversion_rate(
            source, observed, samples=inversion_samples, seed=seed
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
        "angle_stress": stress,
        "normalised_angle_stress": math.sqrt(stress / denominator),
        "angle_rmse_radians": math.sqrt(stress / target.size),
        "pair_count": int(target.size),
    }


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
    max_passes: int = 10
    pull_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    step_radii: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4)
    directions_per_radius: int = 3
    #: Separation ratio may fall by at most this relative amount.
    separation_max_relative_drop: float = 0.01
    #: Uniform-sphere KS may rise by at most this absolute amount.
    ks_max_absolute_increase: float = 0.005


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
    top-5 hit count *and* keeps the cluster separation ratio and the
    uniform-sphere KS statistic inside the configured tolerance.
    """

    n = coords.shape[0]
    true_mask = np.zeros((n, n), dtype=bool)
    true_mask[np.arange(n)[:, None], true_top_k] = True
    state = _RecallState(coords, true_mask)
    rng = np.random.default_rng(config.seed)

    base_separation = separation_ratio(coords, labels)
    base_ks = ks_statistic_uniform_sphere(
        upper_triangle(coordinate_angle_matrix(coords))
    )
    min_separation = base_separation * (1.0 - config.separation_max_relative_drop)
    max_ks = base_ks + config.ks_max_absolute_increase

    started = time.perf_counter()
    history: list[dict[str, object]] = []
    moved_nodes: set[int] = set()
    rejected_separation = 0
    rejected_ks = 0
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
                candidate_ks = ks_statistic_uniform_sphere(
                    upper_triangle(coordinate_angle_matrix(state.coords))
                )
                if candidate_ks > max_ks:
                    rejected_ks += 1
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
            "separation_ratio_before": base_separation,
            "separation_ratio_floor": min_separation,
            "separation_max_relative_drop": config.separation_max_relative_drop,
            "pair_angle_ks_before": base_ks,
            "pair_angle_ks_ceiling": max_ks,
            "ks_max_absolute_increase": config.ks_max_absolute_increase,
            "rejected_by_separation": rejected_separation,
            "rejected_by_ks": rejected_ks,
            "binding": rejected_separation + rejected_ks > 0,
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
    cosines: np.ndarray
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
    coords: np.ndarray, data: Dataset, targets: np.ndarray
) -> dict[str, object]:
    """A + B + C + cluster metrics for one coordinate set."""

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
    }


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
        cosines=cosines,
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
    "Borda rank sum over three pre-declared axes, lowest total wins: "
    "(1) cluster separation ratio, higher is better; "
    "(2) mean Recall@5, higher is better; "
    "(3) coordinate pair-angle KS distance to the uniform sphere, lower is "
    "better. Ties break on the lower KS. The same rule is used for the "
    "preliminary shortlist and for the final recommendation."
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


def run_convex_mode(args: argparse.Namespace) -> int:
    data, paths = load_dataset(args.artifact_dir, args.probe_points)
    n = len(data.ids)
    theta_max = float(np.max(data.pair_angles))
    true_top_k = data.order[:, :TOP_K]

    grid = [(a, gamma) for a in SWEEP_A for gamma in SWEEP_GAMMA]
    contracts = {
        f"a{a}_g{gamma}": convex_contract_report(a, gamma, theta_max)
        for a, gamma in grid
    }

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

    preliminary: dict[str, dict] = {}
    axes: dict[str, dict[str, float]] = {}
    for a, gamma in grid:
        name = f"a{a}_g{gamma}"
        targets = convex_stretch(data.angles, a, gamma, theta_max)
        np.fill_diagonal(targets, 0.0)
        result = optimise_layout(targets, preliminary_config)
        coords = result.coordinates
        cluster = cluster_separation_metrics(coords, data.labels)
        pair_angles = upper_triangle(coordinate_angle_matrix(coords))
        ks = ks_statistic_uniform_sphere(pair_angles)
        dots = np.clip(coords @ coords.T, -1.0, 1.0)
        _, coord_order = similarity_rank_matrix(dots, data.ids)
        recall = recall_metrics(true_top_k, coord_order[:, :TOP_K])
        preliminary[name] = {
            "a": a,
            "gamma": gamma,
            "contract": contracts[name],
            "separation_ratio": cluster["separation_ratio"],
            "silhouette_mean": cluster["silhouette_mean"],
            "pair_angle_ks_vs_uniform_sphere": ks,
            "covering_radius_radians": covering_radius(coords, data.probes),
            **recall,
            "objective": min(result.start_objectives),
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
            f"({result.elapsed_seconds:.1f}s)"
        )

    totals = borda_rank(axes)
    shortlist = sorted(totals, key=lambda name: (totals[name], axes[name]["ks"]))[
        : args.shortlist
    ]
    print(f"shortlist: {shortlist}")

    finalists: dict[str, dict] = {}
    coords_by_name: dict[str, np.ndarray] = {}
    targets_by_name: dict[str, np.ndarray] = {}
    for name in shortlist:
        a = preliminary[name]["a"]
        gamma = preliminary[name]["gamma"]
        targets = convex_stretch(data.angles, a, gamma, theta_max)
        np.fill_diagonal(targets, 0.0)
        result = optimise_layout(targets, full_config)
        before = evaluate_layout(result.coordinates, data, targets)
        refined, refine_report = refine_recall_constrained(
            result.coordinates, true_top_k, data.labels, refine_config
        )
        after = evaluate_layout(refined, data, targets)
        coords_by_name[name] = refined
        targets_by_name[name] = targets
        finalists[name] = {
            "a": a,
            "gamma": gamma,
            "contract": contracts[name],
            "target_monotonicity_violations": target_monotonicity_violations(
                data.pair_angles, upper_triangle(targets)
            ),
            "before_refinement": before,
            "after_refinement": after,
            "refinement": refine_report,
            "multistart_objectives": result.start_objectives,
            "multistart_seeds": result.seeds,
            "multistart_start_kinds": result.start_kinds,
            "multistart_iterations": result.iterations,
            "selected_start": result.selected_start,
            "elapsed_seconds": result.elapsed_seconds,
        }
        print(
            f"final {name}: recall@5 {before['topk']['mean_recall_at_5']:.4f} -> "
            f"{after['topk']['mean_recall_at_5']:.4f} sep "
            f"{before['cluster']['separation_ratio']:.4f} -> "
            f"{after['cluster']['separation_ratio']:.4f} ks "
            f"{before['coverage']['pair_angle_ks_vs_uniform_sphere']:.4f} -> "
            f"{after['coverage']['pair_angle_ks_vs_uniform_sphere']:.4f}"
        )

    final_axes = {
        name: {
            "separation": finalists[name]["after_refinement"]["cluster"][
                "separation_ratio"
            ],
            "recall": finalists[name]["after_refinement"]["topk"]["mean_recall_at_5"],
            "ks": finalists[name]["after_refinement"]["coverage"][
                "pair_angle_ks_vs_uniform_sphere"
            ],
        }
        for name in shortlist
    }
    final_totals = borda_rank(final_axes)
    chosen = sorted(
        final_totals, key=lambda name: (final_totals[name], final_axes[name]["ks"])
    )[0]
    coords = coords_by_name[chosen]
    chosen_targets = targets_by_name[chosen]
    print(f"selected {chosen}")

    # reference layouts, all scored with the same code
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
    comparison = {
        "baseline": evaluate_layout(baseline_coords, data, chosen_targets),
        "rank_scaled": evaluate_layout(rankscaled_coords, data, chosen_targets),
        "linear_ratio_k_max_antipodal": evaluate_layout(
            linear_result.coordinates, data, linear_targets
        ),
        "convex_before_refinement": finalists[chosen]["before_refinement"],
        "convex_after_refinement": finalists[chosen]["after_refinement"],
    }
    comparison["linear_ratio_k_max_antipodal"]["k"] = linear_k

    reference = uniform_reference(
        n, data.probes, replicates=args.uniform_replicates, seed=args.seed
    )

    if args.no_write:
        print(json.dumps(final_axes, indent=2, sort_keys=True))
        return 0

    out_dir: Path = args.output_dir or DEFAULT_CONVEX_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = out_dir / "coordinates.csv"
    write_coordinates_csv(coordinates_path, data.ids, coords)

    layout_report = {
        "algorithm": "convex_monotone_stretch_spherical_stress_v1",
        "mode": "convex",
        "selected_combination": chosen,
        "a": finalists[chosen]["a"],
        "gamma": finalists[chosen]["gamma"],
        "b": contracts[chosen]["b"],
        "theta_max": theta_max,
        "edge_target_policy": (
            "theta_target(i,j) = a*angle + b*angle**gamma with "
            "b = (pi - a*theta_max)/theta_max**gamma, applied to all C(602,2) "
            "pairs; strictly increasing, convex, f(theta) > theta everywhere, "
            "f(theta_max) = pi"
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
        "cluster_policy": data.cluster_policy,
        "coordinate_sha256": coordinate_digest(data.ids, coords),
        **finalists[chosen]["after_refinement"],
    }
    with paths["baseline"].open() as handle:
        baseline_payload = json.load(handle)
    public = build_public_json(baseline_payload, data.ids, coords, layout_report)
    public_path = out_dir / "spherical-graph-public.json"
    with public_path.open("w") as handle:
        json.dump(public, handle, indent=2, sort_keys=True)

    sweep_path = out_dir / "sweep-metrics.json"
    with sweep_path.open("w") as handle:
        json.dump(
            {
                "grid": {"a": list(SWEEP_A), "gamma": list(SWEEP_GAMMA)},
                "procedure": (
                    f"preliminary sweep of all {len(grid)} combinations with "
                    f"{args.preliminary_starts} starts and "
                    f"{args.preliminary_iterations} L-BFGS-B iterations, then "
                    f"the top {args.shortlist} by the selection rule rerun with "
                    f"{args.starts} starts and {args.max_iterations} iterations "
                    "plus the constrained recall refinement"
                ),
                "selection_rule": CONVEX_SELECTION_RULE,
                "contracts": contracts,
                "preliminary": preliminary,
                "preliminary_borda_totals": totals,
                "shortlist": shortlist,
                "finalists": finalists,
                "final_borda_totals": final_totals,
                "selected_combination": chosen,
                "database_reads": 0,
                "database_writes": 0,
                "network_calls": 0,
            },
            handle,
            indent=2,
            sort_keys=True,
            default=str,
        )

    payload = {
        "mode": "convex",
        "node_count": n,
        "pair_count": int(data.pair_angles.size),
        "theta_max": theta_max,
        "a_upper_bound": math.pi / theta_max,
        "selected_combination": chosen,
        "selected_a": finalists[chosen]["a"],
        "selected_gamma": finalists[chosen]["gamma"],
        "selected_b": contracts[chosen]["b"],
        "selection_rule": CONVEX_SELECTION_RULE,
        "contracts": contracts,
        "preliminary": preliminary,
        "preliminary_borda_totals": totals,
        "shortlist": shortlist,
        "finalists": finalists,
        "final_borda_totals": final_totals,
        "comparison": comparison,
        "uniform_sphere_reference": reference,
        "cluster_policy": data.cluster_policy,
        "source_angle_summary": angle_summary(data.pair_angles),
        "sources": {
            "baseline": str(paths["baseline"]),
            "rank_scaled": str(rankscaled_path),
            "linear_ratio": str(DEFAULT_LINEAR_OUTPUT_DIR / "coordinates.csv"),
        },
        "config": {
            "seed": full_config.seed,
            "multistart_count": full_config.multistart_count,
            "max_iterations": full_config.max_iterations,
            "preliminary_starts": args.preliminary_starts,
            "preliminary_iterations": args.preliminary_iterations,
            "shortlist": args.shortlist,
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
        },
        "output_file_sha256": {
            "coordinates.csv": sha256_file(coordinates_path),
            "spherical-graph-public.json": sha256_file(public_path),
            "sweep-metrics.json": sha256_file(sweep_path),
        },
        "coordinate_sha256": coordinate_digest(data.ids, coords),
        "database_reads": 0,
        "database_writes": 0,
        "network_calls": 0,
    }
    metrics_path = out_dir / "layout-metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    print(f"wrote {coordinates_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {public_path}")
    print(f"wrote {sweep_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("convex", "linear"), default="convex")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--rankscaled-dir", type=Path, default=DEFAULT_RANKSCALED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--preliminary-starts", type=int, default=2)
    parser.add_argument("--preliminary-iterations", type=int, default=400)
    parser.add_argument("--shortlist", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--refine-passes", type=int, default=10)
    parser.add_argument("--separation-drop", type=float, default=0.01)
    parser.add_argument("--ks-increase", type=float, default=0.005)
    parser.add_argument("--probe-points", type=int, default=DEFAULT_PROBE_POINTS)
    parser.add_argument(
        "--uniform-replicates", type=int, default=DEFAULT_UNIFORM_REPLICATES
    )
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "linear":
        return run_linear_mode(args)
    return run_convex_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
