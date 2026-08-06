"""Build a deterministic spherical graph JSON from an existing vector artifact.

The command is deliberately offline: it reads one local artifact, never imports
DB code, and performs no network calls. Vectors are used only to precompute the
canonical topology and cluster/hub metadata; the S² force layout is graph-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.embedding_pipeline.core import (
    VectorArtifact,
    file_sha256,
    load_vector_artifact,
    write_json,
)
from app.spherical_graph import (
    CosineKMedoidsClusterer,
    SphericalGraphConfig,
    UnionGraphComponentClusterer,
    VectorRecord,
)
from app.spherical_graph.adapters import build_spherical_graph_from_canonical
from app.vector_similarity import DenseVector, build_canonical_neighbor_artifact

CSV_COLUMNS = ("cocktail_id", "cocktail_name_ko", "embedding")
DEFAULT_INPUT = Path("embedding-artifacts/embeddings-32.csv")
DEFAULT_OUTPUT = Path("embedding-artifacts/spherical-graph.json")
InputFormat = Literal["csv", "npz"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline directed-kNN cocktail graph and deterministic "
            "unit-sphere layout from an existing vector CSV or NPZ."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--clusters",
        type=int,
        default=7,
        help=(
            "Cosine k-medoids cluster count; use 0 for union-graph connected "
            "components."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--iterations", type=int, default=450)
    parser.add_argument(
        "--multistarts",
        type=int,
        default=16,
        help="Deterministic force-layout starts (production default: 16).",
    )
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument(
        "--report-only",
        dest="report_only",
        action="store_true",
        default=True,
        help="Write diagnostics when a gate fails (offline CLI default).",
    )
    quality.add_argument(
        "--enforce-quality",
        dest="report_only",
        action="store_false",
        help="Abort unless every evaluated production acceptance gate passes.",
    )
    return parser


def load_csv_vector_artifact(path: Path) -> VectorArtifact:
    """Strictly load the exported 32D-style CSV without changing the source."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            columns = tuple(reader.fieldnames or ())
            if columns != CSV_COLUMNS:
                raise ValueError(
                    f"{path}: expected columns {CSV_COLUMNS}, got {columns}"
                )
            rows = list(reader)
    except FileNotFoundError as error:
        raise ValueError(f"vector CSV does not exist: {path}") from error
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid vector CSV {path}: {error}") from error

    parsed: list[tuple[int, str, tuple[float, ...]]] = []
    seen_ids: set[int] = set()
    dimension: int | None = None
    for line_number, row in enumerate(rows, start=2):
        if None in row or any(row.get(column) is None for column in CSV_COLUMNS):
            raise ValueError(f"{path}:{line_number}: malformed CSV column count")

        raw_id = row["cocktail_id"]
        raw_name = row["cocktail_name_ko"]
        raw_vector = row["embedding"]
        try:
            cocktail_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: cocktail_id must be an integer"
            ) from error
        if cocktail_id <= 0:
            raise ValueError(f"{path}:{line_number}: cocktail_id must be positive")
        if cocktail_id in seen_ids:
            raise ValueError(
                f"{path}:{line_number}: duplicate cocktail_id {cocktail_id}"
            )

        name = raw_name.strip()
        if not name:
            raise ValueError(
                f"{path}:{line_number}: cocktail_name_ko must be non-empty"
            )

        try:
            decoded = json.loads(raw_vector)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{path}:{line_number}: embedding must be a JSON float array"
            ) from error
        if not isinstance(decoded, list) or not decoded:
            raise ValueError(
                f"{path}:{line_number}: embedding must be a non-empty JSON array"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in decoded
        ):
            raise ValueError(
                f"{path}:{line_number}: embedding values must be finite numbers"
            )
        vector = tuple(float(value) for value in decoded)
        if not any(value != 0.0 for value in vector):
            raise ValueError(f"{path}:{line_number}: embedding vector must be non-zero")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError(f"{path}:{line_number}: embedding dimensions must match")

        seen_ids.add(cocktail_id)
        parsed.append((cocktail_id, name, vector))

    parsed.sort(key=lambda item: item[0])
    vectors = np.asarray([item[2] for item in parsed], dtype=np.float64)
    if vectors.ndim != 2 or not np.all(np.isfinite(vectors)):
        raise ValueError(f"{path}: embeddings must form a finite 2D matrix")
    return VectorArtifact(
        cocktail_ids=np.asarray([item[0] for item in parsed], dtype=np.int64),
        cocktail_names=tuple(item[1] for item in parsed),
        vectors=vectors,
        metadata={},
    )


def load_input_artifact(path: Path) -> tuple[VectorArtifact, InputFormat]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_vector_artifact(path), "csv"
    if suffix == ".npz":
        return load_vector_artifact(path), "npz"
    raise ValueError("input must use the .csv or .npz extension")


def build_graph_payload(
    input_path: Path,
    *,
    k: int,
    clusters: int,
    seed: int,
    iterations: int,
    multistarts: int = 16,
    report_only: bool = True,
) -> dict[str, Any]:
    """Load, hash-pin, and transform an artifact without modifying it."""

    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if type(clusters) is not int or clusters < 0:
        raise ValueError("clusters must be a non-negative integer")

    initial_sha256 = file_sha256(input_path)
    artifact, input_format = load_input_artifact(input_path)
    if len(artifact) < k + 1:
        raise ValueError("input must contain at least k + 1 cocktail rows")
    if clusters > len(artifact):
        raise ValueError("clusters cannot exceed the artifact row count")

    numeric_records = tuple(
        DenseVector.from_values(
            int(cocktail_id),
            tuple(float(value) for value in vector),
        )
        for cocktail_id, vector in zip(
            artifact.cocktail_ids,
            artifact.vectors,
            strict=True,
        )
    )
    canonical_neighbors = build_canonical_neighbor_artifact(
        numeric_records,
        k=k,
    )
    graph_records = tuple(
        VectorRecord(
            node_id=str(record.id),
            vector=record.values,
        )
        for record in numeric_records
    )
    clusterer = (
        UnionGraphComponentClusterer()
        if clusters == 0
        else CosineKMedoidsClusterer(cluster_count=clusters, seed=seed)
    )
    graph = build_spherical_graph_from_canonical(
        graph_records,
        canonical_neighbors,
        clusterer=clusterer,
        config=SphericalGraphConfig(
            k=k,
            seed=seed,
            layout_iterations=iterations,
            multistart_count=multistarts,
            report_only=report_only,
        ),
    )

    final_sha256 = file_sha256(input_path)
    if final_sha256 != initial_sha256:
        raise RuntimeError("input artifact changed during graph construction")

    row_count = len(artifact)
    vector_dimension = (
        int(artifact.vectors.shape[1]) if artifact.vectors.ndim == 2 else 0
    )
    return {
        "schema_version": 1,
        "canonical_neighbor_provenance": {
            "schema_version": canonical_neighbors.schema_version,
            "provider_id": canonical_neighbors.provider_id,
            "k": canonical_neighbors.k,
            "vector_set_sha256": canonical_neighbors.vector_set_sha256,
        },
        "canonical_neighbors": canonical_neighbors.to_dict(),
        "source": {
            "path": str(input_path.resolve()),
            "format": input_format,
            "sha256": initial_sha256,
            "row_count": row_count,
            "vector_dimension": vector_dimension,
            "metadata": artifact.metadata,
        },
        "cocktails": [
            {
                "node_id": str(int(cocktail_id)),
                "cocktail_id": int(cocktail_id),
                "cocktail_name": name,
            }
            for cocktail_id, name in zip(
                artifact.cocktail_ids,
                artifact.cocktail_names,
                strict=True,
            )
        ],
        # Private hub IDs and edges are force-layout implementation details.
        "graph": graph.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path: Path = args.input
    output_path: Path = args.output
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")

    payload = build_graph_payload(
        input_path,
        k=args.k,
        clusters=args.clusters,
        seed=args.seed,
        iterations=args.iterations,
        multistarts=args.multistarts,
        report_only=args.report_only,
    )
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
