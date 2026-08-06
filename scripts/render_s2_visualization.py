"""Render a self-contained 3D S² graph visualisation as a single HTML file.

Offline artifact renderer. Reads the read-only layout artifacts from disk and
writes one standalone HTML file with every byte of data inlined. No database
access, no network calls, no external assets, no dimensionality reduction --
the 3D coordinates are read verbatim from the layout artifacts.

The page draws the 602 cocktail nodes on a unit sphere, connects the 2248
Graph48 union edges as great-circle arcs, and lets the reader inspect, per
node, whether the exact-cosine 48D top-5 neighbours survived the projection.
Eight layouts are shown side by side: baseline, rank-scaled, linear-ratio,
convex v1, and the four v2 spread candidates (the recommended v2 point is the
default).

It also plots the whole 27-point spread/recall frontier so the reader can see
that pushing high-similarity pairs apart -- a 10x range in the clumping metric
``mean_vector_norm`` -- barely moves Recall@5.

Metric provenance: every displayed number is read from a ``layout-metrics.json``
or from ``frontier-metrics.json`` and nothing is hardcoded. Because several
artifacts contain more than one candidate layout, each layout's metric block is
chosen by *matching* the recomputed top-k statistics of the coordinates actually
being drawn -- so the panel can never show one candidate's metrics next to
another candidate's coordinates.

Usage::

    python scripts/render_s2_visualization.py
    python scripts/render_s2_visualization.py --output /tmp/other.html
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.s2_rank_scaled_layout import (  # noqa: E402
    coordinate_angles_and_ranks,
    exact_cosine_matrix,
    load_graph48,
    node_id_sort_key,
    similarity_rank_matrix,
)

DEFAULT_BASELINE_DIR = Path("/private/tmp/cocktail-mate-sensory-artifacts-602-v1")
DEFAULT_RANKSCALED_DIR = Path("/private/tmp/cocktail-mate-s2-rankscaled-602-v1")
DEFAULT_RATIO_DIR = Path("/private/tmp/cocktail-mate-s2-ratio-602-v1")
DEFAULT_CONVEX_DIR = Path("/private/tmp/cocktail-mate-s2-convex-602-v1")
DEFAULT_PREP_DIR = Path("/private/tmp/cocktail-mate-vertex-full-602-prep-v1")
DEFAULT_CONVEX_V2_DIR = REPO_ROOT / "sensory-batch/run-20260806-full602-v1/s2-convex-v2"
TOP_K = 5

# The four v2 spread candidates, read-only, from the convex-v2 frontier run.
# ``file`` is the only thing hardcoded here; every number comes from disk.
V2_LAYOUTS: list[dict[str, str]] = [
    {
        "key": "v2rec",
        "file": "coordinates-convex-a2.0-g1.5-s1.3-centering400000.csv",
        "title": "v2 추천",
        "short": "⑤ v2추천",
        "index": "⑤",
    },
    {
        "key": "v2s15",
        "file": "coordinates-convex-a2.0-g1.5-s1.5.csv",
        "title": "v2 s=1.5",
        "short": "⑥ s1.5",
        "index": "⑥",
    },
    {
        "key": "v2s20",
        "file": "coordinates-convex-a2.0-g1.5-s2.0.csv",
        "title": "v2 s=2.0 (덮기 최대)",
        "short": "⑦ s2.0",
        "index": "⑦",
    },
    {
        "key": "v2raw",
        "file": "coordinates-convex-a2.0-g1.5-s1.3-centering400000-unrefined.csv",
        "title": "v2 추천 · 정련 전",
        "short": "⑧ 정련전",
        "index": "⑧",
    },
]

# Toggle order: the reader walks the improvement history left to right.
LAYOUT_ORDER = ["baseline", "rankscaled", "linear", "convex"] + [
    spec["key"] for spec in V2_LAYOUTS
]
DEFAULT_LAYOUT = "v2rec"

# Where each flat frontier metric lands inside the grouped metric structure the
# comparison matrix reads. The two sources agree numerically where they overlap
# (verified at build time); the frontier only ever fills in what is missing.
FRONTIER_TO_GROUPED: dict[str, tuple[str, ...]] = {
    "mean_recall_at_5": ("topk", "mean_recall_at_5"),
    "hit_rate_at_5": ("topk", "hit_rate_at_5"),
    "full_recovery_rate": ("topk", "full_recovery_rate"),
    "union_edge_rmse_radians_original_acos": (
        "topk",
        "union_edge_rmse_radians_original_acos",
    ),
    "unit_norm_max_error": ("topk", "unit_norm_max_error"),
    "bottom_decile_false_close_count": ("topk", "bottom_decile_false_close_count"),
    "pair_angle_ks_vs_uniform_sphere": ("coverage", "pair_angle_ks_vs_uniform_sphere"),
    "covering_radius_radians": ("coverage", "covering_radius_radians"),
    "empty_cells_100": ("coverage", "equal_area_cells", "cells_100", "empty_cells"),
    "separation_ratio": ("cluster", "separation_ratio"),
    "silhouette_mean": ("cluster", "silhouette_mean"),
    "centroid_angle_min": ("cluster", "centroid_angle_min"),
    "mean_vector_norm": ("bias", "mean_vector_norm"),
    "best_hemisphere_fraction": ("bias", "best_hemisphere_fraction"),
    "pair_angle_median": ("bias", "pair_angle_median"),
    "near_decile_median_stretch": ("gate", "near_decile_median_stretch"),
    "nearest_5pct_fraction_compressed": ("gate", "nearest_5pct_fraction_compressed"),
}

# Gate metric -> where to read that metric for the currently drawn layout.
GATE_METRIC_PATHS: dict[str, list[str]] = {
    "mean_recall_at_5": ["topk", "mean_recall_at_5"],
    "hit_rate_at_5": ["topk", "hit_rate_at_5"],
    "bottom_decile_false_close_count": ["topk", "bottom_decile_false_close_count"],
    "union_edge_rmse_radians_original_acos": [
        "topk",
        "union_edge_rmse_radians_original_acos",
    ],
    "unit_norm_max_error": ["topk", "unit_norm_max_error"],
    "near_decile_median_stretch": ["gate", "near_decile_median_stretch"],
    "nearest_5pct_fraction_compressed": ["gate", "nearest_5pct_fraction_compressed"],
}

GATE_LABELS: dict[str, str] = {
    "mean_recall_at_5": "mean Recall@5",
    "hit_rate_at_5": "hit-rate@5",
    "bottom_decile_false_close_count": "false-close 수 (하위 10%)",
    "union_edge_rmse_radians_original_acos": "union-edge RMSE (rad)",
    "unit_norm_max_error": "unit-norm max error",
    "near_decile_median_stretch": "근거리 10% 신축비 중앙값",
    "nearest_5pct_fraction_compressed": "최근접 5% 압축 비율",
}

GATE_FORMATS: dict[str, str] = {
    "mean_recall_at_5": "ratio",
    "hit_rate_at_5": "ratio",
    "bottom_decile_false_close_count": "int",
    "union_edge_rmse_radians_original_acos": "rad",
    "unit_norm_max_error": "sci",
    "near_decile_median_stretch": "num",
    "nearest_5pct_fraction_compressed": "ratio",
}

DEFAULT_OUTPUTS = [
    DEFAULT_CONVEX_DIR / "s2-graph-visualization.html",
    REPO_ROOT / "sensory-batch/run-20260806-full602-v1/s2-graph-visualization.html",
    DEFAULT_RANKSCALED_DIR / "s2-graph-visualization.html",
]

CLUSTER_COLORS = [
    "#6fb4ff",
    "#ff9f43",
    "#35d08a",
    "#e5679c",
    "#c792ea",
    "#ffd93d",
    "#4ecdc4",
]

# ``better`` drives the per-row winner highlight:
#   high    -> largest value wins
#   low     -> smallest value wins
#   uniform -> closest to the uniform-sphere reference wins
METRIC_GROUPS: list[dict[str, object]] = [
    {
        "title": "몰림(편향) — 구 한쪽에 뭉치는 정도",
        "compare": "uniform",
        "lead": True,
        "rows": [
            {
                "label": "‖평균벡터‖",
                "path": ["bias", "mean_vector_norm"],
                "fmt": "ratio",
                "better": "low",
                "uniform": ["mean_vector_norm_mean"],
            },
            {
                "label": "최적 반구 점유율",
                "path": ["bias", "best_hemisphere_fraction"],
                "fmt": "pct",
                "better": "low",
                "uniform": ["best_hemisphere_fraction_mean"],
            },
            {
                "label": "최대 공백 캡 (rad)",
                "path": ["coverage", "covering_radius_radians"],
                "fmt": "rad",
                "better": "low",
                "uniform": ["covering_radius_radians_mean"],
            },
            {
                "label": "100셀 중 빈 셀",
                "path": ["coverage", "equal_area_cells", "cells_100", "empty_cells"],
                "fmt": "int",
                "better": "low",
                "uniform": ["equal_area_cells", "cells_100", "empty_cells_mean"],
            },
            {
                "label": "쌍각도 KS",
                "path": ["coverage", "pair_angle_ks_vs_uniform_sphere"],
                "fmt": "ratio",
                "better": "low",
                "uniform": ["pair_angle_ks_vs_uniform_sphere_mean"],
            },
            {
                "label": "중앙 쌍각도 (rad)",
                "path": ["bias", "pair_angle_median"],
                "fmt": "rad",
                "better": "uniform",
                "uniform": ["pair_angle_median_mean"],
            },
        ],
    },
    {
        "title": "top-k 보존 (48D exact-cosine top-5)",
        "compare": "thresholds",
        "rows": [
            {
                "label": "mean Recall@5",
                "path": ["topk", "mean_recall_at_5"],
                "fmt": "ratio",
                "better": "high",
                "threshold": "mean_recall_at_5_min",
                "cmp": "ge",
            },
            {
                "label": "hit-rate@5",
                "path": ["topk", "hit_rate_at_5"],
                "fmt": "ratio",
                "better": "high",
                "threshold": "node_coverage_at_5_min",
                "cmp": "ge",
            },
            {
                "label": "full-recovery rate",
                "path": ["topk", "full_recovery_rate"],
                "fmt": "ratio",
                "better": "high",
            },
            {
                "label": "Recall@10",
                "path": ["topk", "recall_at_10"],
                "fmt": "ratio",
                "better": "high",
            },
            {
                "label": "Recall@20",
                "path": ["topk", "recall_at_20"],
                "fmt": "ratio",
                "better": "high",
            },
            {
                "label": "Recall@50",
                "path": ["topk", "recall_at_50"],
                "fmt": "ratio",
                "better": "high",
            },
            {
                "label": "진짜 top-5 좌표순위 중앙값",
                "path": ["topk", "true_top5_coordinate_rank", "median"],
                "fmt": "num",
                "better": "low",
            },
            {
                "label": "침입자 48D 순위 중앙값",
                "path": ["topk", "intruder_cosine_rank", "median"],
                "fmt": "num",
                "better": "high",
            },
            {
                "label": "union-edge RMSE (원본 acos, rad)",
                "path": ["topk", "union_edge_rmse_radians_original_acos"],
                "fmt": "rad",
                "better": "low",
                "threshold": "union_edge_rmse_radians_max",
                "cmp": "le",
            },
            {
                "label": "unit-norm max error",
                "path": ["topk", "unit_norm_max_error"],
                "fmt": "sci",
                "better": "low",
                "threshold": "unit_norm_max_error_max",
                "cmp": "le",
            },
            {
                "label": "false-close count (하위 10%)",
                "path": ["topk", "bottom_decile_false_close_count"],
                "fmt": "int",
                "better": "low",
                "threshold": "bottom_decile_false_close_count_max",
                "cmp": "eq",
            },
        ],
    },
    {
        "title": "구면 덮기 (균일 구면 대비)",
        "compare": "uniform",
        "rows": [
            {
                "label": "쌍각도 KS vs 균일",
                "path": ["coverage", "pair_angle_ks_vs_uniform_sphere"],
                "fmt": "ratio",
                "better": "low",
                "uniform": ["pair_angle_ks_vs_uniform_sphere_mean"],
            },
            {
                "label": "100셀 중 빈 셀",
                "path": ["coverage", "equal_area_cells", "cells_100", "empty_cells"],
                "fmt": "int",
                "better": "low",
                "uniform": ["equal_area_cells", "cells_100", "empty_cells_mean"],
            },
            {
                "label": "400셀 중 빈 셀",
                "path": ["coverage", "equal_area_cells", "cells_400", "empty_cells"],
                "fmt": "int",
                "better": "low",
                "uniform": ["equal_area_cells", "cells_400", "empty_cells_mean"],
            },
            {
                "label": "최대 공백 캡 (rad)",
                "path": ["coverage", "covering_radius_radians"],
                "fmt": "rad",
                "better": "low",
                "uniform": ["covering_radius_radians_mean"],
            },
            {
                "label": "최근접이웃 각 중앙값 (rad)",
                "path": ["coverage", "nearest_neighbour_angle", "median"],
                "fmt": "rad",
                "better": "uniform",
                "uniform": ["nearest_neighbour_angle_mean", "median"],
            },
        ],
    },
    {
        "title": "클러스터 분리 (k-medoids k=7, seed 20260806)",
        "compare": "none",
        "rows": [
            {
                "label": "분리비 (inter/intra)",
                "path": ["cluster", "separation_ratio"],
                "fmt": "num",
                "better": "high",
            },
            {
                "label": "실루엣 평균",
                "path": ["cluster", "silhouette_mean"],
                "fmt": "ratio",
                "better": "high",
            },
            {
                "label": "실루엣 중앙값",
                "path": ["cluster", "silhouette_median"],
                "fmt": "ratio",
                "better": "high",
            },
            {
                "label": "클러스터 중심 간 최소각 (rad)",
                "path": ["cluster", "centroid_angle_min"],
                "fmt": "rad",
                "better": "high",
            },
            {
                "label": "클러스터 내부 평균각 (rad)",
                "path": ["cluster", "mean_intra_cluster_angle"],
                "fmt": "rad",
                "better": "low",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_coordinates_csv(path: Path) -> dict[str, list[float]]:
    """Read ``cocktail_id,x,y,z`` into ``{node_id_string: [x, y, z]}``."""

    coords: dict[str, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"cocktail_id", "x", "y", "z"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            node_id = str(int(row["cocktail_id"]))
            coords[node_id] = [float(row["x"]), float(row["y"]), float(row["z"])]
    return coords


def load_public_json_coordinates(path: Path) -> dict[str, list[float]]:
    """Read ``graph.nodes[]`` coordinates, normalising string node ids to int."""

    document = load_json(path)
    coords: dict[str, list[float]] = {}
    for node in document["graph"]["nodes"]:
        node_id = str(int(node["node_id"]))
        coords[node_id] = [float(node["x"]), float(node["y"]), float(node["z"])]
    return coords


def load_clusters(path: Path, node_ids: Sequence[str]) -> dict[str, object] | None:
    """Read the k-medoids assignment that ships inside a public graph JSON.

    Never re-runs clustering: if the artifact has no component/cluster section
    the caller is expected to drop the cluster colour mode.
    """

    if not path.exists():
        return None
    graph = load_json(path).get("graph", {})
    nodes = graph.get("nodes") or []
    components = graph.get("components") or []
    assignment = {
        str(int(node["node_id"])): node.get("component_id")
        for node in nodes
        if node.get("component_id")
    }
    if not assignment or not components:
        return None
    order = [component["component_id"] for component in components]
    position = {component_id: i for i, component_id in enumerate(order)}
    if any(node_id not in assignment for node_id in node_ids):
        return None
    return {
        "policy": graph.get("clusterer", ""),
        "ids": [position[assignment[node_id]] for node_id in node_ids],
        "labels": [
            f"C{i} · n={component.get('member_count', 0)}"
            for i, component in enumerate(components)
        ],
        "medoids": [str(component.get("medoid_id") or "") for component in components],
    }


def load_acceptance_thresholds(
    layout_metrics: dict, baseline_graph: Path
) -> dict[str, float]:
    """Read the acceptance thresholds, never hardcode them."""

    acceptance = layout_metrics.get("acceptance")
    if isinstance(acceptance, dict) and isinstance(acceptance.get("thresholds"), dict):
        return dict(acceptance["thresholds"])
    document = load_json(baseline_graph)
    return dict(document["graph"]["layout_report"]["acceptance_thresholds"])


def load_directed_top5(path: Path) -> dict[str, list[tuple[int, str, float]]]:
    """Read the 48D ground-truth top-5 as ``{source: [(rank, target, cos)]}``."""

    table: dict[str, list[tuple[int, str, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source = str(int(row["source_id"]))
            table.setdefault(source, []).append(
                (int(row["rank"]), str(int(row["target_id"])), float(row["cosine"]))
            )
    for source in table:
        table[source].sort()
    return table


def load_union_edge_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pairs.append((str(int(row["a_id"])), str(int(row["b_id"]))))
    return pairs


def load_names(
    prep_dir: Path, node_ids: Sequence[str]
) -> tuple[list[str], list[str], str]:
    """Resolve cocktail display names from the read-only prep artifacts.

    ``frozen-cocktails.csv`` is checked first as instructed; it carries no name
    column, so ``cohort-source.csv`` (same read-only directory, no database) is
    the fallback. If neither has names the ids are used verbatim.
    """

    for filename in ("frozen-cocktails.csv", "cohort-source.csv"):
        path = prep_dir / filename
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        name_columns = [column for column in header if "name" in column.lower()]
        if name_columns:
            source = f"{filename}:{','.join(name_columns)}"
            return _read_name_columns(path, node_ids, name_columns, source)
    return list(node_ids), ["" for _ in node_ids], "none (ID만 표시)"


def _read_name_columns(
    path: Path, node_ids: Sequence[str], name_columns: Sequence[str], source: str
) -> tuple[list[str], list[str], str]:
    ko_column = next((c for c in name_columns if c.endswith("_ko")), name_columns[0])
    en_column = next((c for c in name_columns if c.endswith("_en")), "")
    primary: dict[str, str] = {}
    secondary: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            node_id = str(int(row["cocktail_id"]))
            primary[node_id] = (row.get(ko_column) or "").strip()
            secondary[node_id] = (row.get(en_column) or "").strip() if en_column else ""
    names = [primary.get(node_id) or node_id for node_id in node_ids]
    subtitles = [secondary.get(node_id, "") for node_id in node_ids]
    return names, subtitles, source


# ---------------------------------------------------------------------------
# metric block resolution
# ---------------------------------------------------------------------------


def iter_metric_sources(
    convex_metrics: dict, ratio_metrics: dict
) -> Iterator[tuple[str, dict]]:
    """Yield ``(provenance, block)`` for every grouped metric block on disk.

    A grouped block is one that carries ``topk``/``coverage``/``cluster``
    sub-dicts, which is the shape both the convex comparison table and the
    ratio candidate evaluations use.
    """

    for name, block in (convex_metrics.get("comparison") or {}).items():
        if isinstance(block, dict) and "topk" in block:
            yield f"convex/layout-metrics.json:comparison.{name}", block
    for name, entry in (convex_metrics.get("finalists") or {}).items():
        if not isinstance(entry, dict):
            continue
        for phase, block in entry.items():
            if isinstance(block, dict) and "topk" in block:
                yield f"convex/layout-metrics.json:finalists.{name}.{phase}", block
    for name, entry in (ratio_metrics.get("candidates") or {}).items():
        block = (entry or {}).get("evaluation")
        if isinstance(block, dict) and "topk" in block:
            yield f"ratio/layout-metrics.json:candidates.{name}.evaluation", block


def match_metric_block(
    sources: Sequence[tuple[str, dict]], observed: dict[str, float]
) -> tuple[str, dict] | tuple[None, None]:
    """Pick the metric block whose top-k statistics match the drawn coordinates."""

    for provenance, block in sources:
        topk = block.get("topk") or {}
        if all(
            isinstance(topk.get(key), (int, float))
            and abs(float(topk[key]) - value) < 1e-9
            for key, value in observed.items()
        ):
            return provenance, block
    return None, None


def iter_frontier_blocks(frontier: dict) -> Iterator[tuple[str, str, dict]]:
    """Yield ``(label, phase, flat_block)`` for every frontier measurement."""

    for label, entry in (frontier.get("frontier") or {}).items():
        if not isinstance(entry, dict):
            continue
        for phase in ("after_refinement", "before_refinement"):
            block = entry.get(phase)
            if isinstance(block, dict) and "mean_recall_at_5" in block:
                yield label, phase, block


def match_frontier_block(
    frontier: dict, observed: dict[str, float]
) -> tuple[str, str, dict] | tuple[None, None, None]:
    """Find the frontier measurement taken on the coordinates being drawn.

    Matching (never naming) is what keeps a candidate's numbers from ending up
    beside another candidate's points: several frontier labels differ only in a
    refinement phase, and the reference layouts on disk are the *pre*-refinement
    coordinates of their frontier entry.
    """

    hits = [
        (label, phase, block)
        for label, phase, block in iter_frontier_blocks(frontier)
        if all(
            isinstance(block.get(key), (int, float))
            and abs(float(block[key]) - value) < 1e-9
            for key, value in observed.items()
        )
    ]
    if len(hits) != 1:
        return None, None, None
    return hits[0]


def set_missing(root: dict, path: Sequence[str], value: float) -> float | None:
    """Insert ``value`` at ``path``; return the existing value if there was one."""

    node = root
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    existing = node.get(path[-1])
    if isinstance(existing, (int, float)):
        return float(existing)
    node[path[-1]] = value
    return None


def merge_frontier_into_grouped(
    grouped: dict, flat: dict
) -> tuple[dict, list[tuple[str, float, float]]]:
    """Fill the grouped metric structure from the flat frontier block.

    Returns the merged copy plus any disagreement between the two sources, so a
    silent divergence can never masquerade as a fresh number.
    """

    merged = json.loads(json.dumps(grouped)) if grouped else {}
    conflicts: list[tuple[str, float, float]] = []
    for name, path in FRONTIER_TO_GROUPED.items():
        value = flat.get(name)
        if not isinstance(value, (int, float)):
            continue
        existing = set_missing(merged, path, float(value))
        if existing is not None and abs(existing - float(value)) > 1e-9:
            conflicts.append((name, existing, float(value)))
    return merged, conflicts


def merge_uniform_references(primary: dict, extra: dict) -> dict:
    """Union the two uniform-sphere reference blocks (they share a seed)."""

    merged = json.loads(json.dumps(primary)) if primary else {}
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_uniform_references(merged[key], value)
        elif key not in merged:
            merged[key] = value
    return merged


def build_frontier_points(frontier: dict) -> list[dict[str, object]]:
    """Flatten the 27 frontier candidates into a compact plotting payload."""

    points: list[dict[str, object]] = []
    for label, entry in (frontier.get("frontier") or {}).items():
        points.append(
            {
                "label": label,
                "family": entry.get("family", ""),
                "a": entry.get("a"),
                "gamma": entry.get("gamma"),
                "scale": entry.get("scale"),
                "centering_weight": entry.get("centering_weight"),
                "clamped_pair_fraction": entry.get("clamped_pair_fraction"),
                "after": entry.get("after_refinement") or {},
                "before": entry.get("before_refinement") or {},
            }
        )
    return points


# ---------------------------------------------------------------------------
# per-node inspection data
# ---------------------------------------------------------------------------


def build_layout_payload(
    coords: np.ndarray,
    node_ids: Sequence[str],
    cosine_ranks: np.ndarray,
    true_top5: np.ndarray,
) -> dict[str, object]:
    """Compute the per-node top-5 inspection tables for one layout."""

    _, coord_ranks, coord_order = coordinate_angles_and_ranks(coords, node_ids)
    coord_top5 = coord_order[:, :TOP_K]
    node_count = len(node_ids)

    recall = np.zeros(node_count, dtype=np.float64)
    true_rank = np.zeros((node_count, TOP_K), dtype=np.int64)
    intruder_rank = np.zeros((node_count, TOP_K), dtype=np.int64)
    for i in range(node_count):
        truth = set(int(j) for j in true_top5[i])
        found = set(int(j) for j in coord_top5[i])
        recall[i] = len(truth & found) / float(TOP_K)
        for slot, target in enumerate(true_top5[i]):
            true_rank[i, slot] = int(coord_ranks[i, int(target)])
        for slot, candidate in enumerate(coord_top5[i]):
            index = int(candidate)
            intruder_rank[i, slot] = (
                0 if index in truth else int(cosine_ranks[i, index])
            )

    hit_rate = float(np.mean(recall > 0.0))
    full_recovery = float(np.mean(recall >= 1.0))
    return {
        "coords": [[round(float(v), 6) for v in row] for row in coords],
        "coord_top5": coord_top5.astype(int).tolist(),
        "recall5": [round(float(v), 4) for v in recall],
        "true_rank": true_rank.astype(int).tolist(),
        "intruder_rank": intruder_rank.astype(int).tolist(),
        "observed": {
            "mean_recall_at_5": float(recall.mean()),
            "hit_rate_at_5": hit_rate,
            "full_recovery_rate": full_recovery,
        },
    }


def build_payload(
    baseline_dir: Path,
    rankscaled_dir: Path,
    ratio_dir: Path,
    convex_dir: Path,
    prep_dir: Path,
    convex_v2_dir: Path,
) -> dict[str, object]:
    raw_ids, vectors = load_graph48(baseline_dir / "graph48.csv")
    order = sorted(range(len(raw_ids)), key=lambda j: node_id_sort_key(raw_ids[j]))
    node_ids = [str(int(raw_ids[j])) for j in order]
    vectors = vectors[np.asarray(order, dtype=np.int64)]
    index_of = {node_id: i for i, node_id in enumerate(node_ids)}
    node_count = len(node_ids)

    cosines = exact_cosine_matrix(vectors)
    cosine_ranks, cosine_order = similarity_rank_matrix(cosines, node_ids)

    directed = load_directed_top5(baseline_dir / "graph48-directed-top5.csv")
    true_top5 = np.zeros((node_count, TOP_K), dtype=np.int64)
    true_cos = np.zeros((node_count, TOP_K), dtype=np.float64)
    agreements = 0
    for i, node_id in enumerate(node_ids):
        rows = directed[node_id]
        if len(rows) != TOP_K:
            raise ValueError(f"node {node_id} has {len(rows)} top-5 rows")
        for slot, (_, target, cosine) in enumerate(rows):
            true_top5[i, slot] = index_of[target]
            true_cos[i, slot] = cosine
        if set(true_top5[i].tolist()) == set(cosine_order[i, :TOP_K].tolist()):
            agreements += 1

    edge_pairs = load_union_edge_pairs(baseline_dir / "graph48-union-edges.csv")
    edges = [[index_of[a], index_of[b]] for a, b in edge_pairs]

    rankscaled_metrics = load_json(rankscaled_dir / "layout-metrics.json")
    ratio_metrics = load_json(ratio_dir / "layout-metrics.json")
    convex_metrics = load_json(convex_dir / "layout-metrics.json")
    frontier_metrics = load_json(convex_v2_dir / "frontier-metrics.json")
    sources = list(iter_metric_sources(convex_metrics, ratio_metrics))

    convex_a = convex_metrics.get("selected_a")
    convex_gamma = convex_metrics.get("selected_gamma")
    convex_b = convex_metrics.get("selected_b")

    coordinate_maps = {
        "baseline": load_public_json_coordinates(
            baseline_dir / "spherical-graph-public.json"
        ),
        "rankscaled": load_coordinates_csv(rankscaled_dir / "coordinates.csv"),
        "linear": load_coordinates_csv(ratio_dir / "coordinates.csv"),
        "convex": load_coordinates_csv(convex_dir / "coordinates.csv"),
    }
    for spec in V2_LAYOUTS:
        coordinate_maps[spec["key"]] = load_coordinates_csv(
            convex_v2_dir / spec["file"]
        )

    labels = {
        "baseline": "① baseline · acos(cosine)",
        "rankscaled": "② rank-scaled · "
        + str(rankscaled_metrics.get("selected_rescaling", "")),
        "linear": "③ linear-ratio · 단일 배율",
        "convex": "④ convex v1 · a="
        + f"{convex_a}, γ={convex_gamma}"
        + (f", b={round(float(convex_b), 6)}" if convex_b is not None else "")
        + " + 정련",
    }
    short_labels = {
        "baseline": "① base",
        "rankscaled": "② rank",
        "linear": "③ linear",
        "convex": "④ v1",
    }
    for spec in V2_LAYOUTS:
        labels[spec["key"]] = f"{spec['index']} {spec['title']}"
        short_labels[spec["key"]] = spec["short"]

    layouts: dict[str, object] = {}
    provenance: dict[str, str] = {}
    unmatched: list[str] = []
    conflicts: list[str] = []
    for key in LAYOUT_ORDER:
        mapping = coordinate_maps[key]
        missing = set(node_ids) - set(mapping)
        if missing:
            raise ValueError(f"{key} layout is missing {len(missing)} node ids")
        coords = np.asarray([mapping[n] for n in node_ids], dtype=np.float64)
        payload = build_layout_payload(coords, node_ids, cosine_ranks, true_top5)
        source_name, block = match_metric_block(sources, payload["observed"])
        sources_used: list[str] = []
        if block is None:
            block = {}
        else:
            sources_used.append(source_name or "")
        frontier_label, frontier_phase, flat = match_frontier_block(
            frontier_metrics, payload["observed"]
        )
        if flat is None:
            flat = {}
            if not sources_used:
                unmatched.append(key)
        else:
            merged, found = merge_frontier_into_grouped(block, flat)
            block = merged
            conflicts.extend(
                f"{key}.{name}: grouped={old!r} frontier={new!r}"
                for name, old, new in found
            )
            sources_used.append(
                f"s2-convex-v2/frontier-metrics.json:frontier."
                f"{frontier_label}.{frontier_phase}"
            )
        payload["metrics"] = block
        payload["metrics_source"] = (
            " + ".join(sources_used)
            if sources_used
            else "미매칭 (좌표와 일치하는 지표 블록 없음)"
        )
        payload["frontier_label"] = frontier_label
        payload["frontier_phase"] = frontier_phase
        payload["label"] = labels[key]
        payload["short"] = short_labels[key]
        if key == "linear":
            ratio_k = _ratio_k_for(ratio_metrics, source_name or "")
            if ratio_k:
                payload["label"] = f"③ linear-ratio · k={ratio_k}"
        if key in {spec["key"] for spec in V2_LAYOUTS} and flat:
            entry = (frontier_metrics.get("frontier") or {}).get(frontier_label) or {}
            payload["label"] = labels[key] + _v2_parameter_suffix(entry)
        provenance[key] = payload["metrics_source"]
        layouts[key] = payload

    thresholds = load_acceptance_thresholds(
        rankscaled_metrics, baseline_dir / "spherical-graph-public.json"
    )
    names, subtitles, name_source = load_names(prep_dir, node_ids)
    clusters = load_clusters(convex_dir / "spherical-graph-public.json", node_ids)
    uniform = merge_uniform_references(
        convex_metrics.get("uniform_sphere_reference") or {},
        frontier_metrics.get("uniform_sphere_reference") or {},
    )
    frontier_thresholds = frontier_metrics.get("thresholds") or {}

    return {
        "frontier_points": build_frontier_points(frontier_metrics),
        "frontier_gates": {
            "revised": frontier_thresholds.get("revised_acceptance_gates") or {},
            "hard": frontier_thresholds.get("hard_gates") or {},
        },
        "gate_metric_paths": GATE_METRIC_PATHS,
        "gate_labels": GATE_LABELS,
        "gate_formats": GATE_FORMATS,
        "metric_conflicts": conflicts,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node_ids": [int(n) for n in node_ids],
        "names": names,
        "subtitles": subtitles,
        "name_source": name_source,
        "edges": edges,
        "true_top5": true_top5.astype(int).tolist(),
        "true_cos": [[round(float(v), 4) for v in row] for row in true_cos],
        "directed_top5_agreement": agreements,
        "layouts": layouts,
        "layout_order": LAYOUT_ORDER,
        "default_layout": DEFAULT_LAYOUT,
        "metric_provenance": provenance,
        "unmatched_layouts": unmatched,
        "thresholds": thresholds,
        "metric_groups": METRIC_GROUPS,
        "uniform": uniform,
        "clusters": clusters,
        "cluster_colors": CLUSTER_COLORS,
        "top5_cap_radians": rankscaled_metrics.get("top5_cap_radians"),
    }


def _v2_parameter_suffix(entry: dict) -> str:
    """Render ``a/γ/s/λ`` for a frontier entry, straight from the file."""

    parts = []
    for symbol, key in (("a", "a"), ("γ", "gamma"), ("s", "scale")):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{symbol}={float(value):.6g}")
    weight = entry.get("centering_weight")
    if isinstance(weight, (int, float)) and weight:
        parts.append(f"λ={float(weight):.6g}")
    return f" · {', '.join(parts)}" if parts else ""


def _ratio_k_for(ratio_metrics: dict, source_name: str) -> str:
    """Pull the ``k`` of the ratio candidate that a provenance string names."""

    marker = "candidates."
    if marker not in source_name:
        return ""
    candidate = source_name.split(marker, 1)[1].split(".", 1)[0]
    entry = (ratio_metrics.get("candidates") or {}).get(candidate) or {}
    value = entry.get("k")
    return f"{float(value):.6g}" if isinstance(value, (int, float)) else ""


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S² 구면 레이아웃 top-5 검수 — 602 칵테일</title>
<style>
  :root {
    --bg: #080b10;
    --panel: rgba(16, 21, 29, 0.94);
    --line: #22303f;
    --fg: #dde6f0;
    --muted: #8ea1b6;
    --accent: #6fb4ff;
    --pass: #35d08a;
    --fail: #ff6b6b;
    --win: #ffd93d;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg);
    color: var(--fg);
    font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
      "Malgun Gothic", "Noto Sans KR", "Segoe UI", sans-serif;
    overflow: hidden;
  }
  #view { position: fixed; inset: 0; display: block; cursor: grab; }
  #view.dragging { cursor: grabbing; }
  .panel {
    position: fixed;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 12px 14px;
    backdrop-filter: blur(6px);
  }
  #left { top: 14px; left: 14px; bottom: 14px; width: 646px; overflow-y: auto; }
  #right { top: 14px; right: 14px; max-height: calc(100vh - 28px); width: 392px; overflow-y: auto; }
  #bottom {
    position: fixed; left: 674px; right: 420px; bottom: 14px;
    display: flex; flex-direction: column; gap: 10px;
  }
  #bottom .panel { position: static; }
  #frontier { padding: 10px 12px 8px; }
  #fcanvas { display: block; width: 100%; cursor: crosshair; }
  h1 { font-size: 15px; margin: 0 0 2px; letter-spacing: -0.01em; }
  h2 { font-size: 12px; margin: 13px 0 5px; color: var(--muted); font-weight: 600; }
  h2.lead { color: var(--accent); }
  .sub { color: var(--muted); font-size: 11px; margin-bottom: 10px; }
  .row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 7px; }
  button {
    font: inherit; font-size: 12px; color: var(--fg); cursor: pointer;
    background: #16202c; border: 1px solid var(--line); border-radius: 6px;
    padding: 5px 9px;
  }
  button:hover { border-color: var(--accent); }
  button.on { background: #1b3a5c; border-color: var(--accent); color: #fff; }
  table { width: 100%; border-collapse: collapse; }
  #metrics table { font-size: 10px; table-layout: fixed; }
  #metrics th, #metrics td { padding: 2px 2px; }
  #gates table { font-size: 11px; }
  #gates th, #gates td { padding: 2px 4px; }
  #inspect table { font-size: 11.5px; }
  #inspect th, #inspect td { padding: 3px 4px; }
  th, td { text-align: right; border-bottom: 1px solid #18222e; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  td.num, th.num { font-variant-numeric: tabular-nums; }
  td.win { color: var(--win); font-weight: 700; }
  td.cur { background: rgba(111, 180, 255, 0.10); }
  th.cur { background: rgba(111, 180, 255, 0.16); color: #fff; }
  .pass { color: var(--pass); font-weight: 700; }
  .fail { color: var(--fail); font-weight: 700; }
  .dim { color: var(--muted); }
  .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px; vertical-align: -1px; margin-right: 5px; }
  .legend-item { display: inline-flex; align-items: center; margin: 0 10px 3px 0; font-size: 11.5px; }
  .ring { display: inline-block; width: 11px; height: 11px; border-radius: 50%; vertical-align: -1px; margin-right: 5px; }
  #tip {
    position: fixed; pointer-events: none; display: none; z-index: 5;
    background: rgba(8, 12, 18, 0.95); border: 1px solid var(--line);
    border-radius: 6px; padding: 4px 8px; font-size: 12px; white-space: nowrap;
  }
  .kpi { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .hint { color: var(--muted); font-size: 10.5px; margin-top: 8px; }
  .empty { color: var(--muted); font-size: 12px; padding: 18px 0; }
  .banner {
    border: 1px solid #5c2530; background: rgba(255, 107, 107, 0.10);
    border-radius: 7px; padding: 6px 9px; font-size: 11.5px; margin: 6px 0 4px;
  }
  .banner strong { color: var(--fail); }
  /* fixed height: the note must never resize the panel under the pointer,
     or the scatter canvas slides away between hover and click */
  .fnote {
    font-size: 11px; margin-top: 5px; height: 52px; overflow-y: auto;
    line-height: 1.45;
  }
  .ftitle { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .ftitle h2 { margin: 0 0 4px; }
  .ftitle button { padding: 2px 7px; font-size: 11px; }
</style>
</head>
<body>
<canvas id="view"></canvas>

<div class="panel" id="left">
  <h1>S² 구면 레이아웃 top-5 검수</h1>
  <div class="sub" id="head-sub"></div>
  <div class="row" id="layout-buttons"></div>
  <div class="row" id="color-buttons"></div>
  <div class="row">
    <button id="btn-face">선택 정면으로</button>
    <button id="btn-reset">회전 리셋</button>
    <button id="btn-spin">자동 회전</button>
    <button id="btn-edges" class="on">엣지 표시</button>
  </div>
  <div id="gates"></div>
  <div id="metrics"></div>
  <div class="hint" id="foot-hint"></div>
</div>

<div class="panel" id="right">
  <h1>노드 top-5 검수</h1>
  <div id="inspect"><div class="empty">구 위의 노드에 마우스를 올리면 미리보기,
  클릭하면 고정됩니다. 배경을 클릭하면 해제됩니다.</div></div>
</div>

<div id="bottom">
  <div class="panel" id="legend"></div>
  <div class="panel" id="frontier">
    <div class="ftitle">
      <h2 class="lead">몰림-recall 프런티어 · <span id="fcount"></span>개 후보</h2>
      <button id="btn-before">정련 전 겹쳐보기</button>
      <span class="dim" id="fspan"></span>
    </div>
    <canvas id="fcanvas"></canvas>
    <div class="fnote" id="fnote"></div>
  </div>
</div>
<div id="tip"></div>

<script id="payload" type="application/json">__PAYLOAD_JSON__</script>
<script>
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("payload").textContent);
  var N = DATA.node_ids.length;
  var EDGES = DATA.edges;
  var TOPK = 5;

  var RECALL_COLORS = ["#e5484d", "#f2740c", "#ffc53d", "#b0e64c", "#46c17b", "#00d8b0"];
  var RECALL_LABELS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"];
  var CLUSTER_COLORS = DATA.cluster_colors;
  var UNIFORM_COLOR = "#7fb8ff";
  var TRUE_COLOR = "#35d08a";
  var INTRUDER_COLOR = "#ff6b6b";
  var SELECT_COLOR = "#ffffff";

  var canvas = document.getElementById("view");
  var ctx = canvas.getContext("2d");
  var tip = document.getElementById("tip");

  // ---------------------------------------------------------------- state
  var layoutKey = DATA.default_layout;
  var colorMode = "recall";
  var showEdges = true;
  var autoSpin = false;
  var pinned = -1;
  var hovered = -1;
  var zoom = 1.0;
  var rot = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  var dirty = true;

  var W = 0, H = 0, DPR = 1;
  var CAM = 3.6;

  var sx = new Float64Array(N);
  var sy = new Float64Array(N);
  var sz = new Float64Array(N);
  var spp = new Float64Array(N);
  var drawOrder = new Int32Array(N);

  // --------------------------------------------------------------- geometry
  var layouts = {};
  DATA.layout_order.forEach(function (key) {
    var meta = DATA.layouts[key];
    var pos = new Float64Array(N * 3);
    for (var i = 0; i < N; i++) {
      pos[i * 3] = meta.coords[i][0];
      pos[i * 3 + 1] = meta.coords[i][1];
      pos[i * 3 + 2] = meta.coords[i][2];
    }
    layouts[key] = { meta: meta, pos: pos, edges: null };
  });

  function slerpInto(out, off, ax, ay, az, bx, by, bz, t, ang, sinAng) {
    var w0, w1;
    if (sinAng < 1e-8) { w0 = 1 - t; w1 = t; }
    else { w0 = Math.sin((1 - t) * ang) / sinAng; w1 = Math.sin(t * ang) / sinAng; }
    out[off] = ax * w0 + bx * w1;
    out[off + 1] = ay * w0 + by * w1;
    out[off + 2] = az * w0 + bz * w1;
  }

  function buildEdgeGeometry(pos) {
    var starts = new Int32Array(EDGES.length);
    var counts = new Int32Array(EDGES.length);
    var total = 0;
    var angles = new Float64Array(EDGES.length);
    var e, a, b, d, n;
    for (e = 0; e < EDGES.length; e++) {
      a = EDGES[e][0] * 3; b = EDGES[e][1] * 3;
      d = pos[a] * pos[b] + pos[a + 1] * pos[b + 1] + pos[a + 2] * pos[b + 2];
      if (d > 1) d = 1; if (d < -1) d = -1;
      angles[e] = Math.acos(d);
      n = Math.ceil(angles[e] / 0.08) + 1;
      if (n < 2) n = 2; if (n > 40) n = 40;
      starts[e] = total; counts[e] = n + 1;
      total += n + 1;
    }
    var data = new Float32Array(total * 3);
    for (e = 0; e < EDGES.length; e++) {
      a = EDGES[e][0] * 3; b = EDGES[e][1] * 3;
      n = counts[e] - 1;
      var ang = angles[e], sn = Math.sin(ang);
      for (var k = 0; k <= n; k++) {
        slerpInto(data, (starts[e] + k) * 3, pos[a], pos[a + 1], pos[a + 2],
          pos[b], pos[b + 1], pos[b + 2], k / n, ang, sn);
      }
    }
    return { data: data, starts: starts, counts: counts };
  }

  function edgesFor(key) {
    if (!layouts[key].edges) layouts[key].edges = buildEdgeGeometry(layouts[key].pos);
    return layouts[key].edges;
  }

  function buildWireframe() {
    var lines = [], SEG = 90, k, t, pts;
    for (var li = 1; li <= 5; li++) {
      var lat = (li - 3) * (Math.PI / 6);
      var r = Math.cos(lat), z = Math.sin(lat);
      pts = new Float32Array((SEG + 1) * 3);
      for (k = 0; k <= SEG; k++) {
        t = (k / SEG) * 2 * Math.PI;
        pts[k * 3] = r * Math.cos(t); pts[k * 3 + 1] = r * Math.sin(t); pts[k * 3 + 2] = z;
      }
      lines.push(pts);
    }
    for (var mi = 0; mi < 6; mi++) {
      var ph = (mi * Math.PI) / 6;
      pts = new Float32Array((SEG + 1) * 3);
      for (k = 0; k <= SEG; k++) {
        t = (k / SEG) * 2 * Math.PI;
        pts[k * 3] = Math.cos(t) * Math.cos(ph);
        pts[k * 3 + 1] = Math.cos(t) * Math.sin(ph);
        pts[k * 3 + 2] = Math.sin(t);
      }
      lines.push(pts);
    }
    return lines;
  }
  var WIRE = buildWireframe();

  // ------------------------------------------------------------- projection
  function rotX(a) {
    var c = Math.cos(a), s = Math.sin(a);
    return new Float64Array([1, 0, 0, 0, c, -s, 0, s, c]);
  }
  function rotY(a) {
    var c = Math.cos(a), s = Math.sin(a);
    return new Float64Array([c, 0, s, 0, 1, 0, -s, 0, c]);
  }
  function mul(a, b) {
    var o = new Float64Array(9);
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
      }
    }
    return o;
  }

  /** Rotate so the selected node points straight at the camera (Rodrigues). */
  function faceSelection() {
    var sel = selection();
    if (sel < 0) return;
    var pos = layouts[layoutKey].pos;
    var x = pos[sel * 3], y = pos[sel * 3 + 1], z = pos[sel * 3 + 2];
    var len = Math.hypot(y, x);
    if (len < 1e-9) {
      rot = z >= 0 ? new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]) : rotX(Math.PI);
      dirty = true;
      return;
    }
    var ux = y / len, uy = -x / len;
    var th = Math.acos(Math.max(-1, Math.min(1, z)));
    var c = Math.cos(th), s = Math.sin(th), t = 1 - c;
    rot = new Float64Array([
      c + t * ux * ux, t * ux * uy, s * uy,
      t * ux * uy, c + t * uy * uy, -s * ux,
      -s * uy, s * ux, c
    ]);
    dirty = true;
  }

  var scale = 300, cx = 0, cy = 0;

  function projectAll() {
    // keep the sphere inside the band between the two side panels, above the
    // bottom stack (legend + frontier scatter)
    var leftEdge = document.getElementById("left").getBoundingClientRect().right;
    var rightEdge = document.getElementById("right").getBoundingClientRect().left;
    var bottomEdge = document.getElementById("bottom").getBoundingClientRect().top;
    var band = Math.max(240, rightEdge - leftEdge - 28);
    var tall = Math.max(220, bottomEdge - 28);
    scale = zoom * Math.min(band, tall) * 0.46;
    cx = (leftEdge + rightEdge) / 2;
    cy = 14 + tall / 2;
    var pos = layouts[layoutKey].pos;
    for (var i = 0; i < N; i++) {
      var x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      var vx = rot[0] * x + rot[1] * y + rot[2] * z;
      var vy = rot[3] * x + rot[4] * y + rot[5] * z;
      var vz = rot[6] * x + rot[7] * y + rot[8] * z;
      var pp = CAM / (CAM - vz);
      sx[i] = cx + vx * scale * pp;
      sy[i] = cy - vy * scale * pp;
      sz[i] = vz;
      spp[i] = pp;
      drawOrder[i] = i;
    }
    var arr = Array.prototype.slice.call(drawOrder);
    arr.sort(function (p, q) { return sz[p] - sz[q]; });
    for (var j = 0; j < N; j++) drawOrder[j] = arr[j];
  }

  function px(x, y, z) {
    var vz = rot[6] * x + rot[7] * y + rot[8] * z;
    var pp = CAM / (CAM - vz);
    return [
      cx + (rot[0] * x + rot[1] * y + rot[2] * z) * scale * pp,
      cy - (rot[3] * x + rot[4] * y + rot[5] * z) * scale * pp,
      vz
    ];
  }

  // ---------------------------------------------------------------- drawing
  function nodeColor(i) {
    if (colorMode === "recall") {
      return RECALL_COLORS[Math.round(layouts[layoutKey].meta.recall5[i] * 5)];
    }
    if (colorMode === "cluster" && DATA.clusters) {
      return CLUSTER_COLORS[DATA.clusters.ids[i] % CLUSTER_COLORS.length];
    }
    return UNIFORM_COLOR;
  }

  function selection() { return pinned >= 0 ? pinned : hovered; }

  function drawPolyline(pts, count, front) {
    var started = false, i, p;
    for (i = 0; i < count; i++) {
      p = px(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
      if ((p[2] >= 0) !== front) { started = false; continue; }
      if (!started) { ctx.moveTo(p[0], p[1]); started = true; }
      else ctx.lineTo(p[0], p[1]);
    }
  }

  function drawWireframe(front) {
    ctx.beginPath();
    for (var i = 0; i < WIRE.length; i++) {
      drawPolyline(WIRE[i], WIRE[i].length / 3, front);
    }
    ctx.strokeStyle = front ? "rgba(130, 175, 220, 0.22)" : "rgba(130, 175, 220, 0.08)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawEdges(front) {
    var geo = edgesFor(layoutKey);
    var d = geo.data, starts = geo.starts, counts = geo.counts;
    var sel = selection();
    ctx.beginPath();
    for (var e = 0; e < EDGES.length; e++) {
      var s = starts[e], c = counts[e], started = false;
      for (var k = 0; k < c; k++) {
        var o = (s + k) * 3;
        var p = px(d[o], d[o + 1], d[o + 2]);
        if ((p[2] >= 0) !== front) { started = false; continue; }
        if (!started) { ctx.moveTo(p[0], p[1]); started = true; }
        else ctx.lineTo(p[0], p[1]);
      }
    }
    if (front) ctx.strokeStyle = sel >= 0 ? "rgba(125, 160, 195, 0.13)" : "rgba(125, 175, 220, 0.30)";
    else ctx.strokeStyle = sel >= 0 ? "rgba(125, 160, 195, 0.05)" : "rgba(125, 175, 220, 0.10)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawArc(ai, bi, color, width, dash) {
    var pos = layouts[layoutKey].pos;
    var ax = pos[ai * 3], ay = pos[ai * 3 + 1], az = pos[ai * 3 + 2];
    var bx = pos[bi * 3], by = pos[bi * 3 + 1], bz = pos[bi * 3 + 2];
    var d = ax * bx + ay * by + az * bz;
    if (d > 1) d = 1; if (d < -1) d = -1;
    var ang = Math.acos(d), sn = Math.sin(ang);
    var n = Math.max(8, Math.min(64, Math.ceil(ang / 0.03)));
    var buf = new Float64Array(3);
    var pts = [];
    for (var k = 0; k <= n; k++) {
      slerpInto(buf, 0, ax, ay, az, bx, by, bz, k / n, ang, sn);
      pts.push(px(buf[0], buf[1], buf[2]));
    }
    // two passes so the far-side half of the arc stays visible but recedes
    for (var pass = 0; pass < 2; pass++) {
      var front = pass === 1;
      ctx.save();
      ctx.setLineDash(dash || []);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = front ? 1 : 0.28;
      ctx.beginPath();
      var started = false;
      for (var j = 0; j < pts.length; j++) {
        var p = pts[j];
        if ((p[2] >= 0) !== front) { started = false; continue; }
        if (!started) { ctx.moveTo(p[0], p[1]); started = true; }
        else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  function render() {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#080b10";
    ctx.fillRect(0, 0, W, H);
    projectAll();

    var sel = selection();
    var meta = layouts[layoutKey].meta;
    var truth = null, coordTop = null, truthSet = null, coordSet = null;
    if (sel >= 0) {
      truth = DATA.true_top5[sel];
      coordTop = meta.coord_top5[sel];
      truthSet = {}; coordSet = {};
      truth.forEach(function (v) { truthSet[v] = true; });
      coordTop.forEach(function (v) { coordSet[v] = true; });
    }

    drawWireframe(false);
    if (showEdges) drawEdges(false);

    ctx.beginPath();
    ctx.arc(cx, cy, scale * CAM / Math.sqrt(CAM * CAM - 1), 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(140, 180, 220, 0.35)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    drawWireframe(true);
    if (showEdges) drawEdges(true);

    if (sel >= 0) {
      for (var t = 0; t < truth.length; t++) {
        drawArc(sel, truth[t], "rgba(53, 208, 138, 0.95)", 2.0, null);
      }
      for (var c = 0; c < coordTop.length; c++) {
        if (!truthSet[coordTop[c]]) {
          drawArc(sel, coordTop[c], "rgba(255, 107, 107, 0.85)", 1.4, [4, 3]);
        }
      }
    }

    for (var oi = 0; oi < N; oi++) {
      var i = drawOrder[oi];
      var depth = (sz[i] + 1) / 2;
      var r = (2.0 + 2.0 * depth) * spp[i] * Math.min(1.6, zoom * 0.9 + 0.35);
      var alpha = 0.20 + 0.80 * depth;
      var fill = nodeColor(i);
      var ring = null, ringWidth = 0, ringDash = null, ringColor = null;
      // highlighted nodes stay legible on the far side, but stay translucent
      // there so the reader can still tell which hemisphere they sit on
      var hi = sz[i] >= 0 ? 1 : 0.45;

      if (sel >= 0) {
        if (i === sel) {
          fill = SELECT_COLOR; r *= 2.0; alpha = hi;
          ringColor = "#ffffff"; ringWidth = 2; ring = true;
        } else if (truthSet[i] && coordSet[i]) {
          fill = TRUE_COLOR; r *= 1.7; alpha = hi;
          ringColor = "#ffffff"; ringWidth = 1.8; ring = true;
        } else if (truthSet[i]) {
          fill = TRUE_COLOR; r *= 1.6; alpha = hi;
          ringColor = TRUE_COLOR; ringWidth = 1.4; ringDash = [3, 2]; ring = true;
        } else if (coordSet[i]) {
          fill = "#1a232e"; r *= 1.6; alpha = hi;
          ringColor = INTRUDER_COLOR; ringWidth = 2.2; ring = true;
        } else {
          alpha *= 0.30;
        }
      } else if (i === hovered) {
        r *= 1.8; alpha = 1; ringColor = "#ffffff"; ringWidth = 1.5; ring = true;
      }

      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(sx[i], sy[i], r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      if (ring) {
        ctx.setLineDash(ringDash || []);
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = ringWidth;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.globalAlpha = 1;
  }

  function loop() {
    if (autoSpin) { rot = mul(rotY(0.0045), rot); dirty = true; }
    if (dirty) { dirty = false; render(); }
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------ interaction
  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = Math.round(W * DPR);
    canvas.height = Math.round(H * DPR);
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    FW = Math.max(240, Math.round(fcanvas.parentNode.clientWidth - 24));
    FH = Math.max(140, Math.min(210, Math.round(H * 0.21)));
    fcanvas.width = Math.round(FW * DPR);
    fcanvas.height = Math.round(FH * DPR);
    fcanvas.style.height = FH + "px";
    drawFrontier();
    dirty = true;
  }
  window.addEventListener("resize", resize);

  var dragging = false, lastX = 0, lastY = 0, moved = 0;
  canvas.addEventListener("mousedown", function (ev) {
    dragging = true; moved = 0; lastX = ev.clientX; lastY = ev.clientY;
    canvas.classList.add("dragging");
  });
  window.addEventListener("mouseup", function (ev) {
    if (!dragging) return;
    dragging = false;
    canvas.classList.remove("dragging");
    if (moved < 5 && ev.target === canvas) {
      var hit = pick(ev.clientX, ev.clientY);
      pinned = hit >= 0 && hit !== pinned ? hit : -1;
      updateInspector();
      dirty = true;
    }
  });
  window.addEventListener("mousemove", function (ev) {
    if (dragging) {
      var dx = ev.clientX - lastX, dy = ev.clientY - lastY;
      moved += Math.abs(dx) + Math.abs(dy);
      lastX = ev.clientX; lastY = ev.clientY;
      rot = mul(mul(rotX(-dy * 0.006), rotY(dx * 0.006)), rot);
      dirty = true;
      return;
    }
    if (ev.target === fcanvas) return;  // the scatter owns its own tooltip
    // only the sphere canvas hovers nodes; the panels own their own pointers
    var hit = ev.target === canvas ? pick(ev.clientX, ev.clientY) : -1;
    if (hit !== hovered) {
      hovered = hit;
      updateInspector();
      dirty = true;
    }
    if (hit >= 0) {
      tip.style.display = "block";
      tip.style.left = (ev.clientX + 14) + "px";
      tip.style.top = (ev.clientY + 14) + "px";
      tip.textContent = label(hit) + " · Recall@5 " +
        layouts[layoutKey].meta.recall5[hit].toFixed(1);
    } else {
      tip.style.display = "none";
    }
  });
  canvas.addEventListener("wheel", function (ev) {
    ev.preventDefault();
    zoom *= Math.exp(-ev.deltaY * 0.0014);
    zoom = Math.max(0.5, Math.min(6, zoom));
    dirty = true;
  }, { passive: false });

  function pick(mx, my) {
    var best = -1, bestD = 16 * 16;
    for (var i = 0; i < N; i++) {
      if (sz[i] < -0.05) continue;
      var dx = sx[i] - mx, dy = sy[i] - my;
      var d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function label(i) {
    var name = DATA.names[i];
    return name === String(DATA.node_ids[i]) ? "#" + DATA.node_ids[i] : name;
  }
  function labelFull(i) {
    return label(i) + " (#" + DATA.node_ids[i] + ")";
  }

  // ----------------------------------------------------------------- panels
  function dig(root, path) {
    var node = root;
    for (var i = 0; i < path.length; i++) {
      if (node === null || node === undefined || typeof node !== "object") return null;
      node = node[path[i]];
    }
    return typeof node === "number" ? node : null;
  }

  function fmt(kind, value) {
    if (value === null || value === undefined) return "—";
    if (kind === "ratio" || kind === "rad") return Number(value).toFixed(4);
    if (kind === "pct") return (Number(value) * 100).toFixed(1) + "%";
    if (kind === "int") return String(value);
    if (kind === "sci") return Number(value).toExponential(2);
    return Number(value).toFixed(2);
  }

  function bestIndex(values, better, uniformValue) {
    var best = -1, bestScore = Infinity;
    for (var i = 0; i < values.length; i++) {
      if (values[i] === null) continue;
      var score;
      if (better === "high") score = -values[i];
      else if (better === "low") score = values[i];
      else if (better === "uniform" && uniformValue !== null) {
        score = Math.abs(values[i] - uniformValue);
      } else continue;
      if (score < bestScore) { bestScore = score; best = i; }
    }
    return best;
  }

  function verdict(row, value) {
    if (!row.threshold || value === null) return "";
    var limit = DATA.thresholds[row.threshold];
    if (limit === undefined) return "";
    var ok = row.cmp === "ge" ? value >= limit
      : row.cmp === "le" ? value <= limit
      : value === limit;
    return ok ? "pass" : "fail";
  }

  function thresholdText(row) {
    if (!row.threshold) return "—";
    var limit = DATA.thresholds[row.threshold];
    if (limit === undefined) return "—";
    return (row.cmp === "ge" ? "≥ " : row.cmp === "le" ? "≤ " : "= ") + limit;
  }

  function updateMetrics() {
    var order = DATA.layout_order;
    var html = "";
    DATA.metric_groups.forEach(function (group) {
      html += '<h2 class="' + (group.lead ? "lead" : "") + '">' + group.title +
        "</h2><table><tr><th>항목</th>";
      order.forEach(function (key) {
        html += '<th class="num' + (key === layoutKey ? " cur" : "") + '">' +
          DATA.layouts[key].short + "</th>";
      });
      html += '<th class="num">' + (group.compare === "uniform" ? "균일" : "임계값") +
        "</th></tr>";

      group.rows.forEach(function (row) {
        var values = order.map(function (key) {
          return dig(DATA.layouts[key].metrics, row.path);
        });
        var uniformValue = row.uniform ? dig(DATA.uniform, row.uniform) : null;
        var win = bestIndex(values, row.better, uniformValue);
        html += "<tr><td>" + row.label + "</td>";
        values.forEach(function (value, i) {
          var cls = "num";
          if (i === win) cls += " win";
          if (order[i] === layoutKey) cls += " cur";
          var mark = verdict(row, value);
          var text = fmt(row.fmt, value);
          if (mark) text += mark === "pass" ? ' <span class="pass">✓</span>'
            : ' <span class="fail">✗</span>';
          html += '<td class="' + cls + '">' + text + "</td>";
        });
        html += '<td class="num dim">' +
          (group.compare === "uniform" ? fmt(row.fmt, uniformValue) : thresholdText(row)) +
          "</td></tr>";
      });
      html += "</table>";
    });
    html += '<div class="hint">노란색 = 그 행의 최고값. ✓/✗ = 수락 임계값 통과 여부. ' +
      "지표는 전부 layout-metrics.json / frontier-metrics.json에서 읽습니다. " +
      "빈 칸(—)은 그 레이아웃 지표 파일에 해당 항목이 없다는 뜻입니다.</div>";
    document.getElementById("metrics").innerHTML = html;
  }

  // ------------------------------------------------------------------ gates
  function gateValue(name) {
    var path = DATA.gate_metric_paths[name];
    if (!path) return null;
    return dig(layouts[layoutKey].meta.metrics, path);
  }

  function gatePassed(spec, value) {
    if (value === null) return null;
    return spec[0] === ">=" ? value >= spec[1]
      : spec[0] === "<=" ? value <= spec[1]
      : value === spec[1];
  }

  function gateTable(title, gates) {
    var names = Object.keys(gates).sort();
    var failed = [];
    var html = "<h2>" + title + "</h2><table>" +
      "<tr><th>항목</th><th>값</th><th>게이트</th><th>판정</th></tr>";
    names.forEach(function (name) {
      var spec = gates[name];
      var value = gateValue(name);
      var ok = gatePassed(spec, value);
      if (ok === false) failed.push(name);
      var kind = DATA.gate_formats[name] || "num";
      html += "<tr><td>" + (DATA.gate_labels[name] || name) + '</td><td class="num">' +
        fmt(kind, value) + '</td><td class="num dim">' + spec[0] + " " +
        fmt(kind, spec[1]) + "</td><td>" +
        (ok === null ? '<span class="dim">—</span>'
          : ok ? '<span class="pass">PASS</span>' : '<span class="fail">FAIL</span>') +
        "</td></tr>";
    });
    html += "</table>";
    return { html: html, failed: failed, total: names.length };
  }

  function updateGates() {
    var revised = gateTable("수락 게이트 (revised_acceptance_gates)",
      DATA.frontier_gates.revised);
    var hard = gateTable("하드 게이트 (hard_gates)", DATA.frontier_gates.hard);
    var head = "<h2>게이트 판정 — " + DATA.layouts[layoutKey].short + "</h2>";
    var lines = [];
    [["수락", revised], ["하드", hard]].forEach(function (pair) {
      var g = pair[1];
      if (g.failed.length) {
        lines.push('<div><strong>' + pair[0] + " 게이트 " +
          (g.total - g.failed.length) + "/" + g.total + " 통과 · 실패: " +
          g.failed.map(function (n) {
            return (DATA.gate_labels[n] || n) + " = " +
              fmt(DATA.gate_formats[n] || "num", gateValue(n)) + " (" +
              DATA.frontier_gates[pair[0] === "수락" ? "revised" : "hard"][n][0] + " " +
              fmt(DATA.gate_formats[n] || "num",
                DATA.frontier_gates[pair[0] === "수락" ? "revised" : "hard"][n][1]) +
              ")";
          }).join(", ") + "</strong></div>");
      } else {
        lines.push("<div>" + pair[0] + " 게이트 " + g.total + "/" + g.total +
          ' <span class="pass">전부 PASS</span></div>');
      }
    });
    var banner = '<div class="banner">' + lines.join("") + hardGateSummary() + "</div>";
    document.getElementById("gates").innerHTML =
      head + banner + revised.html + hard.html;
  }

  /** The hard gate that no candidate on the frontier clears -- stated plainly. */
  function hardGateSummary() {
    var name = "nearest_5pct_fraction_compressed";
    var spec = DATA.frontier_gates.hard[name];
    if (!spec) return "";
    var best = null, bestLabel = "", passing = 0, total = 0;
    DATA.frontier_points.forEach(function (p) {
      var v = p.after[name];
      if (typeof v !== "number") return;
      total += 1;
      if (gatePassed(spec, v)) passing += 1;
      if (best === null || v < best) { best = v; bestLabel = p.label; }
    });
    if (!total) return "";
    return "<div>프런티어 " + total + "개 후보 중 <strong>" + passing +
      "개만</strong> " + (DATA.gate_labels[name] || name) + " " + spec[0] + " " +
      fmt("ratio", spec[1]) + " 통과 (정련 후 최소값 " + fmt("ratio", best) +
      " — " + bestLabel + ") — 이 하드 게이트는 아직 미해결입니다.</div>";
  }

  // -------------------------------------------------------- frontier scatter
  var fcanvas = document.getElementById("fcanvas");
  var fctx = fcanvas.getContext("2d");
  var FW = 0, FH = 0;
  var showBefore = false;
  var fHover = -1;
  var fPicked = -1;
  var fPickedNote = "";
  var FPAD = { l: 44, r: 10, t: 10, b: 24 };

  var FPTS = [];
  DATA.frontier_points.forEach(function (p, pi) {
    [["after", p.after], ["before", p.before]].forEach(function (pair) {
      var block = pair[1];
      if (!block || typeof block.mean_vector_norm !== "number") return;
      var owner = null;
      DATA.layout_order.forEach(function (key) {
        var meta = DATA.layouts[key];
        if (meta.frontier_label === p.label &&
            meta.frontier_phase === pair[0] + "_refinement") {
          owner = key;
        }
      });
      FPTS.push({
        entry: p, index: pi, fi: 0, phase: pair[0], block: block, owner: owner,
        x: block.mean_vector_norm, y: block.mean_recall_at_5, px: 0, py: 0
      });
    });
  });
  FPTS.forEach(function (p, i) { p.fi = i; });

  var FDOM = (function () {
    var xmax = 0, ymax = 0;
    FPTS.forEach(function (p) {
      if (p.x > xmax) xmax = p.x;
      if (p.y > ymax) ymax = p.y;
    });
    return { x: Math.ceil(xmax * 20) / 20, y: Math.ceil(ymax * 20) / 20 };
  })();

  function fx(v) { return FPAD.l + (v / FDOM.x) * (FW - FPAD.l - FPAD.r); }
  function fy(v) { return FH - FPAD.b - (v / FDOM.y) * (FH - FPAD.t - FPAD.b); }

  function fVisible(p) {
    return p.phase === "after" || showBefore ||
      p.owner === layoutKey || p.fi === fPicked;
  }

  function paramText(entry) {
    var parts = [];
    if (typeof entry.a === "number") parts.push("a=" + entry.a);
    if (typeof entry.gamma === "number") parts.push("γ=" + entry.gamma);
    if (typeof entry.scale === "number") parts.push("s=" + entry.scale);
    if (typeof entry.centering_weight === "number") {
      parts.push("λ=" + entry.centering_weight);
    }
    return parts.length ? parts.join(" · ") : entry.family;
  }

  function pointText(p) {
    var b = p.block;
    return "<strong>" + p.entry.label + "</strong> · " +
      (p.phase === "after" ? "정련 후" : "정련 전") + "<br>" + paramText(p.entry) +
      "<br>‖평균벡터‖ " + fmt("ratio", b.mean_vector_norm) +
      " · Recall@5 " + fmt("ratio", b.mean_recall_at_5) +
      " · hit@5 " + fmt("ratio", b.hit_rate_at_5) +
      "<br>최적 반구 " + fmt("pct", b.best_hemisphere_fraction) +
      " · 공백 캡 " + fmt("rad", b.covering_radius_radians) +
      " · false-close " + fmt("int", b.bottom_decile_false_close_count) +
      "<br>최근접 5% 압축 " + fmt("ratio", b.nearest_5pct_fraction_compressed) +
      " · KS " + fmt("ratio", b.pair_angle_ks_vs_uniform_sphere);
  }

  function drawFrontier() {
    if (!FW || !FH) return;
    fctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    fctx.clearRect(0, 0, FW, FH);
    fctx.font = "9px -apple-system, sans-serif";

    // grid + axes
    fctx.strokeStyle = "rgba(130, 175, 220, 0.16)";
    fctx.fillStyle = "#8ea1b6";
    fctx.lineWidth = 1;
    var step = FDOM.x > 0.5 ? 0.2 : 0.1, v;
    for (v = 0; v <= FDOM.x + 1e-9; v += step) {
      fctx.beginPath();
      fctx.moveTo(fx(v), FPAD.t); fctx.lineTo(fx(v), FH - FPAD.b); fctx.stroke();
      fctx.textAlign = "center";
      fctx.fillText(v.toFixed(1), fx(v), FH - FPAD.b + 11);
    }
    for (v = 0; v <= FDOM.y + 1e-9; v += 0.1) {
      fctx.beginPath();
      fctx.moveTo(FPAD.l, fy(v)); fctx.lineTo(FW - FPAD.r, fy(v)); fctx.stroke();
      fctx.textAlign = "right";
      fctx.fillText(v.toFixed(1), FPAD.l - 5, fy(v) + 3);
    }
    fctx.textAlign = "left";
    fctx.fillText("‖평균벡터‖ →  (왼쪽일수록 고르게 퍼짐)", FPAD.l + 2, FH - 3);
    fctx.save();
    fctx.translate(9, FPAD.t + 4);
    fctx.rotate(-Math.PI / 2);
    fctx.textAlign = "right";
    fctx.fillText("mean Recall@5", 0, 0);
    fctx.restore();

    // uniform-sphere reference (x) and the recall acceptance gate (y)
    var uni = dig(DATA.uniform, ["mean_vector_norm_mean"]);
    if (uni !== null) {
      fctx.setLineDash([3, 3]);
      fctx.strokeStyle = "rgba(111, 180, 255, 0.55)";
      fctx.beginPath();
      fctx.moveTo(fx(uni), FPAD.t); fctx.lineTo(fx(uni), FH - FPAD.b); fctx.stroke();
      fctx.setLineDash([]);
      fctx.fillStyle = "#6fb4ff";
      fctx.textAlign = "left";
      fctx.fillText("균일 " + fmt("ratio", uni), fx(uni) + 3, FPAD.t + 8);
    }
    var gate = DATA.frontier_gates.revised.mean_recall_at_5;
    if (gate) {
      fctx.setLineDash([5, 3]);
      fctx.strokeStyle = "rgba(255, 217, 61, 0.55)";
      fctx.beginPath();
      fctx.moveTo(FPAD.l, fy(gate[1])); fctx.lineTo(FW - FPAD.r, fy(gate[1]));
      fctx.stroke();
      fctx.setLineDash([]);
      fctx.fillStyle = "#ffd93d";
      fctx.textAlign = "right";
      fctx.fillText("게이트 " + gate[0] + " " + fmt("ratio", gate[1]),
        FW - FPAD.r - 2, fy(gate[1]) - 3);
    }

    FPTS.forEach(function (p) {
      p.px = fx(p.x); p.py = fy(p.y);
    });
    FPTS.forEach(function (p) {
      if (!fVisible(p)) return;
      var isCurrent = p.owner === layoutKey;
      fctx.beginPath();
      fctx.arc(p.px, p.py, p.phase === "after" ? 3.6 : 2.6, 0, Math.PI * 2);
      if (p.phase === "after") {
        fctx.fillStyle = p.owner ? "#35d08a" : "#6fb4ff";
        fctx.globalAlpha = isCurrent ? 1 : 0.85;
        fctx.fill();
      } else {
        fctx.strokeStyle = "rgba(142, 161, 182, 0.75)";
        fctx.lineWidth = 1.2;
        fctx.globalAlpha = 1;
        fctx.stroke();
      }
      fctx.globalAlpha = 1;
      if (p.owner) {
        fctx.beginPath();
        fctx.arc(p.px, p.py, 6, 0, Math.PI * 2);
        fctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
        fctx.lineWidth = 1;
        fctx.stroke();
      }
      if (isCurrent) {
        fctx.beginPath();
        fctx.arc(p.px, p.py, 9, 0, Math.PI * 2);
        fctx.strokeStyle = "#ffffff";
        fctx.lineWidth = 2;
        fctx.stroke();
        fctx.fillStyle = "#ffffff";
        fctx.textAlign = "left";
        fctx.fillText(DATA.layouts[layoutKey].short, p.px + 11, p.py + 3);
      } else if (p.fi === fPicked) {
        fctx.beginPath();
        fctx.arc(p.px, p.py, 8, 0, Math.PI * 2);
        fctx.setLineDash([3, 2]);
        fctx.strokeStyle = "#ffd93d";
        fctx.lineWidth = 1.5;
        fctx.stroke();
        fctx.setLineDash([]);
      }
    });
  }

  function fpick(mx, my) {
    var best = -1, bestD = 11 * 11;
    for (var i = 0; i < FPTS.length; i++) {
      if (!fVisible(FPTS[i])) continue;
      var dx = FPTS[i].px - mx, dy = FPTS[i].py - my;
      var d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function frontierSummary() {
    var xs = [], ys = [];
    FPTS.forEach(function (p) {
      if (p.phase !== "after") return;
      xs.push(p.x); ys.push(p.y);
    });
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    return "정련 후 " + xs.length + "점: ‖평균벡터‖ " + fmt("ratio", xmin) + " → " +
      fmt("ratio", xmax) + " (" + (xmax / xmin).toFixed(1) +
      "배 차이) 인데 mean Recall@5는 " + fmt("ratio", ymin) + " ~ " +
      fmt("ratio", ymax) + " 뿐 — 거의 수평선입니다.";
  }

  function updateFrontierNote() {
    var note = document.getElementById("fnote");
    var idx = fHover >= 0 ? fHover : -1;
    if (idx < 0) {
      var cur = null;
      FPTS.forEach(function (p) { if (p.owner === layoutKey) cur = p; });
      var html = '<span class="dim">' + frontierSummary() + "</span>";
      if (cur) {
        html = "현재 표시 중: " + pointText(cur).replace(/<br>/g, " · ") + "<br>" + html;
      }
      if (fPickedNote) html = fPickedNote + "<br>" + html;
      note.innerHTML = html;
      return;
    }
    var p = FPTS[idx];
    note.innerHTML = pointText(p).replace(/<br>/g, " · ") + "<br>" +
      '<span class="dim">' + (p.owner
        ? "클릭하면 이 좌표로 전환합니다 (" + DATA.layouts[p.owner].short + ")"
        : "좌표 미저장 — 지표만 있고 이 페이지에 좌표 파일이 없습니다") + "</span>";
  }

  fcanvas.addEventListener("mousemove", function (ev) {
    var rect = fcanvas.getBoundingClientRect();
    var hit = fpick(ev.clientX - rect.left, ev.clientY - rect.top);
    if (hit !== fHover) {
      fHover = hit;
      updateFrontierNote();
    }
    if (hit >= 0) {
      tip.style.display = "block";
      tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 300) + "px";
      tip.style.top = (ev.clientY - 96) + "px";
      tip.innerHTML = pointText(FPTS[hit]);
    } else {
      tip.style.display = "none";
    }
  });
  fcanvas.addEventListener("mouseleave", function () {
    fHover = -1;
    tip.style.display = "none";
    updateFrontierNote();
  });
  fcanvas.addEventListener("click", function (ev) {
    var rect = fcanvas.getBoundingClientRect();
    var hit = fpick(ev.clientX - rect.left, ev.clientY - rect.top);
    if (hit < 0) return;
    var p = FPTS[hit];
    if (p.owner) {
      fPicked = -1;
      fPickedNote = "";
      setLayout(p.owner);
      return;
    }
    fPicked = p.fi;
    fPickedNote = '<span class="fail">좌표 미저장</span> · ' +
      pointText(p).replace(/<br>/g, " · ");
    fHover = -1;
    updateFrontierNote();
    drawFrontier();
  });

  function updateInspector() {
    var sel = selection();
    var box = document.getElementById("inspect");
    if (sel < 0) {
      box.innerHTML = '<div class="empty">구 위의 노드에 마우스를 올리면 미리보기, ' +
        "클릭하면 고정됩니다. 배경을 클릭하면 해제됩니다.</div>";
      return;
    }
    var meta = layouts[layoutKey].meta;
    var truth = DATA.true_top5[sel];
    var trueRank = meta.true_rank[sel];
    var coordTop = meta.coord_top5[sel];
    var intruder = meta.intruder_rank[sel];
    var truthSet = {};
    truth.forEach(function (v) { truthSet[v] = true; });
    var recall = meta.recall5[sel];

    var html = "<div><strong>" + labelFull(sel) + "</strong>";
    if (DATA.subtitles[sel]) html += ' <span class="dim">' + DATA.subtitles[sel] + "</span>";
    html += "</div>";
    if (DATA.clusters) {
      html += '<div class="dim" style="font-size:11px">클러스터 ' +
        DATA.clusters.labels[DATA.clusters.ids[sel]] + "</div>";
    }
    html += '<div class="kpi" style="color:' + RECALL_COLORS[Math.round(recall * 5)] + '">Recall@5 ' +
      recall.toFixed(1) + '</div><div class="dim" style="margin-bottom:8px">' +
      (recall * TOPK) + " / 5 개의 48D 정답 top-5가 좌표상 top-5 안에 있습니다 · " +
      DATA.layouts[layoutKey].short + " · " + (pinned >= 0 ? "고정됨" : "미리보기") + "</div>";

    html += "<h2>48D 정답 top-5 (초록)</h2><table>";
    html += "<tr><th>48D 순위</th><th>칵테일</th><th>cos</th><th>좌표순위</th><th>판정</th></tr>";
    for (var i = 0; i < truth.length; i++) {
      var hit = trueRank[i] <= TOPK;
      html += "<tr><td>" + (i + 1) + "</td><td>" + labelFull(truth[i]) +
        '</td><td class="num">' + DATA.true_cos[sel][i].toFixed(4) +
        '</td><td class="num">' + trueRank[i] + "</td><td>" +
        (hit ? '<span class="pass">적중</span>' : '<span class="fail">이탈</span>') +
        "</td></tr>";
    }
    html += "</table>";

    html += "<h2>좌표상 실제 최근접 5개 (빨강 테두리)</h2><table>";
    html += "<tr><th>좌표순위</th><th>칵테일</th><th>48D 순위</th><th>판정</th></tr>";
    for (var j = 0; j < coordTop.length; j++) {
      var isHit = !!truthSet[coordTop[j]];
      html += "<tr><td>" + (j + 1) + "</td><td>" + labelFull(coordTop[j]) +
        '</td><td class="num">' + (isHit ? "≤5" : intruder[j]) + "</td><td>" +
        (isHit ? '<span class="pass">정답</span>' : '<span class="fail">침입자</span>') +
        "</td></tr>";
    }
    html += "</table>";
    box.innerHTML = html;
  }

  function updateLegend() {
    var html = "";
    if (colorMode === "recall") {
      html += "<div><strong>노드 색 = 그 노드의 개별 Recall@5</strong>&nbsp;&nbsp;";
      for (var i = 0; i < RECALL_COLORS.length; i++) {
        html += '<span class="legend-item"><span class="swatch" style="background:' +
          RECALL_COLORS[i] + '"></span>' + RECALL_LABELS[i] + "</span>";
      }
      html += "</div>";
    } else if (colorMode === "cluster" && DATA.clusters) {
      html += "<div><strong>노드 색 = 클러스터</strong> <span class='dim'>" +
        DATA.clusters.policy + "</span>&nbsp;&nbsp;";
      for (var c = 0; c < DATA.clusters.labels.length; c++) {
        html += '<span class="legend-item"><span class="swatch" style="background:' +
          CLUSTER_COLORS[c % CLUSTER_COLORS.length] + '"></span>' +
          DATA.clusters.labels[c] + "</span>";
      }
      html += "</div>";
    } else {
      html += "<div><strong>노드 색 = 단색 (구조만 보기)</strong></div>";
    }
    html += "<div style='margin-top:6px'><strong>선택 시</strong>&nbsp;&nbsp;" +
      '<span class="legend-item"><span class="ring" style="background:#fff"></span>선택 노드</span>' +
      '<span class="legend-item"><span class="ring" style="background:' + TRUE_COLOR +
      ';border:2px solid #fff"></span>정답 top-5 · 적중</span>' +
      '<span class="legend-item"><span class="ring" style="background:' + TRUE_COLOR +
      ';border:1px dashed ' + TRUE_COLOR + '"></span>정답 top-5 · 이탈</span>' +
      '<span class="legend-item"><span class="ring" style="background:#1a232e;border:2px solid ' +
      INTRUDER_COLOR + '"></span>침입자</span></div>';
    document.getElementById("legend").innerHTML = html;
  }

  function setLayout(key) {
    if (!DATA.layouts[key]) return;
    layoutKey = key;
    var lb = document.getElementById("layout-buttons");
    Array.prototype.forEach.call(lb.children, function (c, idx) {
      c.className = DATA.layout_order[idx] === layoutKey ? "on" : "";
    });
    updateMetrics();
    updateGates();
    updateInspector();
    drawFrontier();
    updateFrontierNote();
    dirty = true;
  }

  function buildButtons() {
    var lb = document.getElementById("layout-buttons");
    DATA.layout_order.forEach(function (key) {
      var b = document.createElement("button");
      b.textContent = DATA.layouts[key].label;
      b.className = key === layoutKey ? "on" : "";
      b.onclick = function () { setLayout(key); };
      lb.appendChild(b);
    });

    var modes = [["recall", "색: Recall@5"]];
    if (DATA.clusters) modes.push(["cluster", "색: 클러스터"]);
    modes.push(["uniform", "색: 단색"]);
    var cb = document.getElementById("color-buttons");
    modes.forEach(function (pair) {
      var b = document.createElement("button");
      b.textContent = pair[1];
      b.className = pair[0] === colorMode ? "on" : "";
      b.onclick = function () {
        colorMode = pair[0];
        Array.prototype.forEach.call(cb.children, function (c) { c.className = ""; });
        b.className = "on";
        updateLegend(); dirty = true;
      };
      cb.appendChild(b);
    });

    document.getElementById("btn-face").onclick = faceSelection;
    document.getElementById("btn-reset").onclick = function () {
      rot = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      zoom = 1.0; dirty = true;
    };
    var spin = document.getElementById("btn-spin");
    spin.onclick = function () {
      autoSpin = !autoSpin;
      spin.className = autoSpin ? "on" : "";
      dirty = true;
    };
    var eb = document.getElementById("btn-edges");
    eb.onclick = function () {
      showEdges = !showEdges;
      eb.className = showEdges ? "on" : "";
      dirty = true;
    };
    var bb = document.getElementById("btn-before");
    bb.onclick = function () {
      showBefore = !showBefore;
      bb.className = showBefore ? "on" : "";
      drawFrontier();
    };
  }

  document.getElementById("head-sub").textContent =
    N + "개 노드 · " + EDGES.length + "개 union edge (대원호) · 48D exact-cosine top-5 기준 · 레이아웃 " +
    DATA.layout_order.length + "종";
  var provenance = DATA.layout_order.map(function (key) {
    return DATA.layouts[key].short + " ← " + DATA.layouts[key].metrics_source;
  }).join("<br>");
  document.getElementById("foot-hint").innerHTML =
    "드래그 회전 · 휠 줌 · 노드 클릭 고정<br>이름 출처: " + DATA.name_source +
    "<br>지표 출처 (좌표와 top-k가 일치하는 블록을 자동 선택):<br>" + provenance +
    "<br>생성: " + DATA.generated_at;
  document.getElementById("fcount").textContent = DATA.frontier_points.length;
  document.getElementById("fspan").textContent =
    "x=‖평균벡터‖, y=mean Recall@5 · 흰 테두리 = 좌표가 저장된 점(클릭하면 전환)";

  buildButtons();
  updateMetrics();
  updateGates();
  updateLegend();
  updateInspector();
  resize();
  updateFrontierNote();
  loop();
})();
</script>
</body>
</html>
"""


def render_html(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD_JSON__", blob)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--rankscaled-dir", type=Path, default=DEFAULT_RANKSCALED_DIR)
    parser.add_argument("--ratio-dir", type=Path, default=DEFAULT_RATIO_DIR)
    parser.add_argument("--convex-dir", type=Path, default=DEFAULT_CONVEX_DIR)
    parser.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP_DIR)
    parser.add_argument("--convex-v2-dir", type=Path, default=DEFAULT_CONVEX_V2_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        help="output HTML path; repeatable, defaults to the three shipped paths",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        args.baseline_dir,
        args.rankscaled_dir,
        args.ratio_dir,
        args.convex_dir,
        args.prep_dir,
        args.convex_v2_dir,
    )
    html = render_html(payload)

    outputs = args.output or DEFAULT_OUTPUTS
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")

    for key in payload["layout_order"]:
        layout = payload["layouts"][key]
        observed = layout["observed"]
        print(
            f"{key:11s} R@5={observed['mean_recall_at_5']:.6f} "
            f"hit={observed['hit_rate_at_5']:.6f} "
            f"full={observed['full_recovery_rate']:.6f} "
            f"<- {layout['frontier_label']}/{layout['frontier_phase']}"
        )
    if payload["unmatched_layouts"]:
        print(f"WARNING unmatched metric blocks: {payload['unmatched_layouts']}")
    if payload["metric_conflicts"]:
        print(f"WARNING metric source conflicts: {payload['metric_conflicts']}")
    print(f"frontier points: {len(payload['frontier_points'])}")
    clusters = payload["clusters"]
    print(f"clusters: {clusters['policy'] if clusters else 'NOT AVAILABLE'}")
    print(
        f"directed-top5 agreement: {payload['directed_top5_agreement']}"
        f"/{len(payload['node_ids'])}"
    )
    print(f"names: {payload['name_source']}")
    for output in outputs:
        print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
