"""Build local sensory embeddings, exact neighbors, and a public S² graph.

The command is intentionally offline. It accepts only projection-ready teacher
JSONL produced by ``scripts/sensory_vertex_batch.py project`` and never imports
database application code or a cloud SDK.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.sensory_embedding import (
    RAW240_COORDINATES,
    SENSORY_V2_LEVELS,
    SENSORY_V2_REGISTRY,
    AxisSoftLabels,
    TEACHER_SOURCE_SCHEMA,
    TeacherEmbeddingBundle,
    project_teacher_soft_labels,
)
from app.sensory_embedding.contracts import canonical_sha256
from app.sensory_embedding.vertex_batch import (
    COHORT_ID_SET_SHA256,
    CORPUS_ROWS,
    RESULT_SCHEMA_VERSION,
    id_set_sha256,
)
from app.spherical_graph import (
    CosineKMedoidsClusterer,
    DirectedNeighbor as SphericalDirectedNeighbor,
    GraphEdge,
    SphericalGraph,
    SphericalGraphConfig,
    UnionGraphComponentClusterer,
    VectorRecord as SphericalVectorRecord,
    build_spherical_graph_from_topology,
    prepare_spherical_graph_topology,
    similarity_to_target_distance,
)
from app.vector_similarity import (
    CanonicalGraph48Artifact,
    build_graph48_artifact,
    graph48_ids_sha256,
    write_canonical_graph48_csv,
)

ARTIFACT_SCHEMA_VERSION = 1
TOPOLOGY_PROVIDER_ID = "vector_similarity_graph48_exact_cosine_v1"
PUBLIC_HUB_PREFIX = "__spherical_graph_hub__:"

RAW240_FILENAME = "raw240.csv"
GRAPH48_FILENAME = "graph48.csv"
PREFERENCE48_FILENAME = "preference48.csv"
CANONICAL_RUN_FILENAME = "canonical-run.csv"
DIRECTED_FILENAME = "graph48-directed-top5.csv"
UNION_FILENAME = "graph48-union-edges.csv"
SPHERICAL_FILENAME = "spherical-graph-public.json"
MANIFEST_FILENAME = "manifest.json"


class SensoryArtifactError(ValueError):
    """Projection-ready input or artifact output violates the local contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SensoryArtifactError(f"cannot read input file {path}: {error}") from error
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _strict_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SensoryArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_positive_id(value: object, *, line_number: int) -> int:
    if type(value) is not int or value <= 0:
        raise SensoryArtifactError(
            f"input line {line_number}: cocktail_id must be a positive integer"
        )
    return value


def _load_axis(
    value: object,
    *,
    expected_order: int,
    line_number: int,
) -> tuple[AxisSoftLabels, str, str]:
    if not isinstance(value, Mapping):
        raise SensoryArtifactError(
            f"input line {line_number}: axis {expected_order} must be an object"
        )
    expected = SENSORY_V2_REGISTRY.axes[expected_order]
    if (
        value.get("axis_order") != expected_order
        or value.get("axis_id") != expected.axis_id
    ):
        raise SensoryArtifactError(
            f"input line {line_number}: axes must follow the exact registry order"
        )
    probabilities = value.get("probabilities")
    if isinstance(probabilities, (str, bytes)) or not isinstance(
        probabilities, Sequence
    ):
        raise SensoryArtifactError(
            f"input line {line_number}: axis probabilities must be an array"
        )
    try:
        labels = AxisSoftLabels.from_values(expected.axis_id, probabilities)
    except (TypeError, ValueError) as error:
        raise SensoryArtifactError(
            f"input line {line_number}: invalid {expected.axis_id} probabilities"
        ) from error
    response_sha256 = _strict_sha256(
        value.get("response_sha256"),
        field=f"input line {line_number} response_sha256",
    )
    raw_response_sha256 = _strict_sha256(
        value.get("raw_response_sha256"),
        field=f"input line {line_number} raw_response_sha256",
    )
    return labels, response_sha256, raw_response_sha256


def _teacher_lineage_sha256(
    *,
    cocktail_id: int,
    record_sha256: str,
    axes: Sequence[Mapping[str, object]],
) -> str:
    return canonical_sha256(
        {
            "axes": [
                {
                    "axis_id": axis["axis_id"],
                    "axis_order": axis["axis_order"],
                    "raw_response_sha256": axis["raw_response_sha256"],
                    "response_sha256": axis["response_sha256"],
                }
                for axis in axes
            ],
            "cocktail_id": cocktail_id,
            "projection_ready_record_sha256": record_sha256,
            "schema": "sensory-teacher-lineage-v1",
        }
    )


def load_projection_ready_jsonl(
    path: Path,
    *,
    allow_partial: bool,
) -> tuple[
    tuple[TeacherEmbeddingBundle, ...],
    dict[int, tuple[str, str]],
    str,
    str,
]:
    """Strictly load, validate, and project complete 48×A-E teacher records."""

    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise SensoryArtifactError(
            f"cannot read projection JSONL {path}: {error}"
        ) from error
    if not lines:
        raise SensoryArtifactError("projection-ready input must not be empty")

    bundles: list[TeacherEmbeddingBundle] = []
    lineage: dict[int, tuple[str, str]] = {}
    seen_ids: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise SensoryArtifactError(
                f"input line {line_number}: blank records are forbidden"
            )
        try:
            raw = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SensoryArtifactError(
                f"input line {line_number}: invalid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise SensoryArtifactError(
                f"input line {line_number}: record must be an object"
            )
        cocktail_id = _canonical_positive_id(
            raw.get("cocktail_id"),
            line_number=line_number,
        )
        if cocktail_id in seen_ids:
            raise SensoryArtifactError(f"duplicate cocktail_id: {cocktail_id}")
        seen_ids.add(cocktail_id)
        if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise SensoryArtifactError(
                f"input line {line_number}: unsupported schema_version"
            )
        if raw.get("registry_sha256") != SENSORY_V2_REGISTRY.registry_sha256:
            raise SensoryArtifactError(
                f"input line {line_number}: registry_sha256 mismatch"
            )
        if raw.get("labels") != list(SENSORY_V2_LEVELS):
            raise SensoryArtifactError(
                f"input line {line_number}: labels must be exactly A-E"
            )
        raw_axes = raw.get("axes")
        if not isinstance(raw_axes, list) or len(raw_axes) != len(
            SENSORY_V2_REGISTRY.axes
        ):
            raise SensoryArtifactError(
                f"input line {line_number}: exactly 48 axes are required"
            )
        loaded_axes = tuple(
            _load_axis(
                axis,
                expected_order=axis_order,
                line_number=line_number,
            )
            for axis_order, axis in enumerate(raw_axes)
        )
        axes = tuple(item[0] for item in loaded_axes)
        flattened = tuple(
            probability for axis in axes for probability in axis.probabilities
        )
        raw_probabilities = raw.get("raw_probabilities")
        if (
            not isinstance(raw_probabilities, list)
            or tuple(raw_probabilities) != flattened
        ):
            raise SensoryArtifactError(
                f"input line {line_number}: raw_probabilities do not match axes"
            )
        try:
            bundle = project_teacher_soft_labels(cocktail_id, axes)
        except ValueError as error:
            raise SensoryArtifactError(
                f"input line {line_number}: projection failed: {error}"
            ) from error
        if raw.get("source_sha256") != bundle.source_sha256:
            raise SensoryArtifactError(
                f"input line {line_number}: source_sha256 mismatch"
            )
        record_sha256 = _canonical_json_sha256(raw)
        lineage_axes = tuple(
            {
                "axis_order": axis_order,
                "axis_id": axes[axis_order].axis_id,
                "response_sha256": loaded_axes[axis_order][1],
                "raw_response_sha256": loaded_axes[axis_order][2],
            }
            for axis_order in range(len(axes))
        )
        lineage[cocktail_id] = (
            record_sha256,
            _teacher_lineage_sha256(
                cocktail_id=cocktail_id,
                record_sha256=record_sha256,
                axes=lineage_axes,
            ),
        )
        bundles.append(bundle)

    bundles.sort(key=lambda item: item.cocktail_id)
    ids = tuple(bundle.cocktail_id for bundle in bundles)
    cohort_ids_sha256 = id_set_sha256(ids)
    graph_ids_sha256 = graph48_ids_sha256(ids)
    if not allow_partial and (
        len(bundles) != CORPUS_ROWS or cohort_ids_sha256 != COHORT_ID_SET_SHA256
    ):
        raise SensoryArtifactError(
            "production mode requires exactly the current 602-cocktail cohort "
            f"(ids_sha256={COHORT_ID_SET_SHA256})"
        )
    if allow_partial and len(bundles) <= 5:
        raise SensoryArtifactError("partial mode requires at least six cocktails")
    return tuple(bundles), lineage, cohort_ids_sha256, graph_ids_sha256


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _projection_csv_payloads(
    bundles: Sequence[TeacherEmbeddingBundle],
    lineage: Mapping[int, tuple[str, str]],
) -> dict[str, bytes]:
    axis_ids = tuple(axis.axis_id for axis in SENSORY_V2_REGISTRY.axes)
    raw_columns = tuple(
        f"{axis_id}__p_{label}" for axis_id, label in RAW240_COORDINATES
    )
    common = (
        "cocktail_id",
        "registry_version",
        "registry_sha256",
        "source_sha256",
        "projection_ready_record_sha256",
        "teacher_lineage_sha256",
        "projection_provenance_sha256",
    )
    vector_common = common + (
        "contract_version",
        "contract_sha256",
        "vector_sha256",
    )
    raw_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    preference_rows: list[dict[str, object]] = []
    for bundle in bundles:
        record_sha256, lineage_sha256 = lineage[bundle.cocktail_id]
        base: dict[str, object] = {
            "cocktail_id": bundle.cocktail_id,
            "registry_version": bundle.raw240.registry_version,
            "registry_sha256": bundle.registry_sha256,
            "source_sha256": bundle.source_sha256,
            "projection_ready_record_sha256": record_sha256,
            "teacher_lineage_sha256": lineage_sha256,
            "projection_provenance_sha256": bundle.provenance_sha256,
        }
        raw_rows.append(
            {
                **base,
                **dict(zip(raw_columns, map(repr, bundle.raw240.values), strict=True)),
            }
        )
        graph_rows.append(
            {
                **base,
                "contract_version": bundle.graph48.contract.version,
                "contract_sha256": bundle.graph48.contract.contract_sha256,
                "vector_sha256": bundle.graph48.vector_sha256,
                **dict(zip(axis_ids, map(repr, bundle.graph48.values), strict=True)),
            }
        )
        preference_rows.append(
            {
                **base,
                "contract_version": bundle.preference48.contract.version,
                "contract_sha256": bundle.preference48.contract.contract_sha256,
                "vector_sha256": bundle.preference48.vector_sha256,
                **dict(
                    zip(axis_ids, map(repr, bundle.preference48.values), strict=True)
                ),
            }
        )
    return {
        RAW240_FILENAME: _csv_bytes(common + raw_columns, raw_rows),
        GRAPH48_FILENAME: _csv_bytes(vector_common + axis_ids, graph_rows),
        PREFERENCE48_FILENAME: _csv_bytes(
            vector_common + axis_ids,
            preference_rows,
        ),
    }


def _spherical_graph(
    bundles: Sequence[TeacherEmbeddingBundle],
    artifact: CanonicalGraph48Artifact,
    *,
    clusters: int,
    seed: int,
    iterations: int,
    multistarts: int,
    report_only: bool,
) -> SphericalGraph:
    records = tuple(
        SphericalVectorRecord(
            node_id=str(bundle.cocktail_id),
            vector=bundle.graph48.values,
        )
        for bundle in bundles
    )
    directed = tuple(
        SphericalDirectedNeighbor(
            source_id=row.source_id,
            target_id=row.target_id,
            rank=row.rank,
            similarity=row.cosine,
            target_distance=similarity_to_target_distance(row.cosine),
        )
        for row in artifact.directed_neighbors
    )
    edges = tuple(
        GraphEdge(
            source_id=edge.a_id,
            target_id=edge.b_id,
            edge_kind="cocktail_knn",
            similarity=edge.cosine,
            target_distance=similarity_to_target_distance(edge.cosine),
            source_rank=edge.a_rank,
            target_rank=edge.b_rank,
            is_mutual=edge.mutual,
            is_bridge=False,
            visible=True,
            recommendable=True,
        )
        for edge in artifact.union_edges
    )
    clusterer = (
        UnionGraphComponentClusterer()
        if clusters == 0
        else CosineKMedoidsClusterer(cluster_count=clusters, seed=seed)
    )
    (
        node_ids,
        components,
        private_hub_edges,
        clustering_policy,
        audit_similarities,
    ) = prepare_spherical_graph_topology(
        records,
        directed_neighbors=directed,
        cocktail_edges=edges,
        clusterer=clusterer,
    )
    return build_spherical_graph_from_topology(
        node_ids,
        directed_neighbors=directed,
        cocktail_edges=edges,
        components=components,
        private_hub_edges=private_hub_edges,
        topology_provider_id=TOPOLOGY_PROVIDER_ID,
        clustering_policy=clustering_policy,
        audit_similarities=audit_similarities,
        audit_node_ids=node_ids,
        config=SphericalGraphConfig(
            k=5,
            seed=seed,
            layout_iterations=iterations,
            multistart_count=multistarts,
            report_only=report_only,
        ),
    )


def _assert_public_graph(payload: Mapping[str, object], *, row_count: int) -> None:
    graph = payload.get("graph")
    if not isinstance(graph, Mapping):
        raise SensoryArtifactError("public S² payload is missing graph")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or len(nodes) != row_count:
        raise SensoryArtifactError("public S² payload must contain every cocktail node")
    if not isinstance(edges, list):
        raise SensoryArtifactError("public S² payload edges are invalid")
    if any(
        not isinstance(node, Mapping)
        or node.get("node_kind") != "cocktail"
        or node.get("visible") is not True
        or node.get("recommendable") is not True
        for node in nodes
    ):
        raise SensoryArtifactError("public S² payload leaked a hidden node")
    if any(
        not isinstance(edge, Mapping)
        or edge.get("edge_kind") != "cocktail_knn"
        or edge.get("visible") is not True
        or edge.get("recommendable") is not True
        for edge in edges
    ):
        raise SensoryArtifactError("public S² payload leaked a hidden edge")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if PUBLIC_HUB_PREFIX in encoded:
        raise SensoryArtifactError("public S² payload leaked a private hub identifier")


def _atomic_create(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_sensory_artifacts(
    input_path: Path,
    output_dir: Path,
    *,
    run_id: str,
    allow_partial: bool = False,
    clusters: int = 7,
    seed: int = 20260806,
    iterations: int = 450,
    multistarts: int = 16,
    report_only: bool = True,
) -> dict[str, object]:
    """Build a create-only local artifact directory and return its manifest."""

    if not run_id:
        raise SensoryArtifactError("run_id must be non-empty")
    if type(clusters) is not int or clusters < 0:
        raise SensoryArtifactError("clusters must be a non-negative integer")
    if type(seed) is not int:
        raise SensoryArtifactError("seed must be an integer")
    if type(iterations) is not int or iterations <= 0:
        raise SensoryArtifactError("iterations must be a positive integer")
    if type(multistarts) is not int or multistarts <= 0:
        raise SensoryArtifactError("multistarts must be a positive integer")
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")

    initial_input_sha256 = _file_sha256(input_path)
    bundles, lineage, cohort_ids_sha256, graph_ids_sha256 = load_projection_ready_jsonl(
        input_path,
        allow_partial=allow_partial,
    )
    if clusters > len(bundles):
        raise SensoryArtifactError("clusters cannot exceed cocktail count")
    csv_payloads = _projection_csv_payloads(bundles, lineage)
    graph_artifact = build_graph48_artifact(
        tuple(bundle.graph48.embedding for bundle in bundles),
        run_id=run_id,
        source_artifact_sha256=initial_input_sha256,
    )
    graph = _spherical_graph(
        bundles,
        graph_artifact,
        clusters=clusters,
        seed=seed,
        iterations=iterations,
        multistarts=multistarts,
        report_only=report_only,
    )
    graph_payload: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "provenance": {
            "run_id": run_id,
            "input_file_sha256": initial_input_sha256,
            "cohort_ids_sha256": cohort_ids_sha256,
            "graph48_ids_sha256": graph_ids_sha256,
            "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
            "graph48_contract_sha256": bundles[0].graph48.contract.contract_sha256,
            "graph48_vector_set_sha256": graph_artifact.vector_sha256,
            "topology_provider": TOPOLOGY_PROVIDER_ID,
            "layout_method": "graph-only; no high-dimensional coordinate projection",
        },
        "public_hub_node_count": 0,
        "public_hub_edge_count": 0,
        "graph": graph.to_dict(),
    }
    _assert_public_graph(graph_payload, row_count=len(bundles))
    graph_json = _json_bytes(graph_payload)

    final_input_sha256 = _file_sha256(input_path)
    if final_input_sha256 != initial_input_sha256:
        raise SensoryArtifactError("input file changed during artifact construction")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.create.lock"
    lock_descriptor: int | None = None
    staging: Path | None = None
    try:
        try:
            lock_descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise SensoryArtifactError(
                f"another create is active for {output_dir}"
            ) from error
        if output_dir.exists():
            raise FileExistsError(f"refusing to replace existing output: {output_dir}")
        staging = Path(
            tempfile.mkdtemp(
                dir=output_dir.parent,
                prefix=f".{output_dir.name}.",
            )
        )
        for filename, payload in csv_payloads.items():
            _atomic_create(staging / filename, payload)
        write_canonical_graph48_csv(
            graph_artifact,
            run_path=staging / CANONICAL_RUN_FILENAME,
            directed_path=staging / DIRECTED_FILENAME,
            union_path=staging / UNION_FILENAME,
        )
        _atomic_create(staging / SPHERICAL_FILENAME, graph_json)

        data_filenames = (
            RAW240_FILENAME,
            GRAPH48_FILENAME,
            PREFERENCE48_FILENAME,
            CANONICAL_RUN_FILENAME,
            DIRECTED_FILENAME,
            UNION_FILENAME,
            SPHERICAL_FILENAME,
        )
        files = {
            filename: {
                "sha256": _file_sha256(staging / filename),
                "bytes": (staging / filename).stat().st_size,
            }
            for filename in data_filenames
        }
        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "mode": "partial-test" if allow_partial else "production-current-602",
            "network_calls": 0,
            "database_reads": 0,
            "database_writes": 0,
            "input": {
                "path": str(input_path),
                "sha256": initial_input_sha256,
                "schema_version": RESULT_SCHEMA_VERSION,
            },
            "cohort": {
                "row_count": len(bundles),
                "ids_sha256": cohort_ids_sha256,
                "graph48_ids_sha256": graph_ids_sha256,
                "production_expected_row_count": CORPUS_ROWS,
                "production_expected_ids_sha256": COHORT_ID_SET_SHA256,
            },
            "registry": {
                "version": SENSORY_V2_REGISTRY.version,
                "sha256": SENSORY_V2_REGISTRY.registry_sha256,
                "axis_count": len(SENSORY_V2_REGISTRY.axes),
                "level_order": list(SENSORY_V2_LEVELS),
            },
            "spaces": {
                "raw240": {
                    "dimension": 240,
                    "source_schema": TEACHER_SOURCE_SCHEMA,
                },
                "graph48": {
                    "dimension": 48,
                    "metric": "exact_cosine",
                    "contract_sha256": bundles[0].graph48.contract.contract_sha256,
                    "vector_set_sha256": graph_artifact.vector_sha256,
                    "k": 5,
                },
                "preference48": {
                    "dimension": 48,
                    "metric": "maximum_inner_product",
                    "contract_sha256": (
                        bundles[0].preference48.contract.contract_sha256
                    ),
                },
            },
            "spherical_layout": {
                "method": "graph-only; no high-dimensional coordinate projection",
                "clusters": clusters,
                "seed": seed,
                "iterations": iterations,
                "multistarts": multistarts,
                "report_only": report_only,
                "public_hub_node_count": 0,
                "public_hub_edge_count": 0,
                "quality": graph.layout_report,
            },
            "files": files,
        }
        _atomic_create(staging / MANIFEST_FILENAME, _json_bytes(manifest))
        if output_dir.exists():
            raise FileExistsError(f"refusing to replace existing output: {output_dir}")
        os.rename(staging, output_dir)
        staging = None
        return manifest
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build local raw240/graph48/preference48 CSVs, exact top-5 graph "
            "CSVs, and a public graph-only S² JSON."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Test-only: permit an arbitrary cohort of at least six cocktails.",
    )
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--iterations", type=int, default=450)
    parser.add_argument("--multistarts", type=int, default=16)
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument(
        "--report-only",
        dest="report_only",
        action="store_true",
        default=True,
        help="Create artifacts with complete diagnostics even if a gate fails.",
    )
    quality.add_argument(
        "--enforce-quality",
        dest="report_only",
        action="store_false",
        help="Abort unless all S² acceptance checks pass.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = build_sensory_artifacts(
            arguments.input,
            arguments.output_dir,
            run_id=arguments.run_id,
            allow_partial=arguments.allow_partial,
            clusters=arguments.clusters,
            seed=arguments.seed,
            iterations=arguments.iterations,
            multistarts=arguments.multistarts,
            report_only=arguments.report_only,
        )
    except (FileExistsError, OSError, SensoryArtifactError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "output_dir": str(arguments.output_dir),
                "manifest_sha256": _file_sha256(
                    arguments.output_dir / MANIFEST_FILENAME
                ),
                "row_count": manifest["cohort"]["row_count"],  # type: ignore[index]
                "network_calls": 0,
                "database_writes": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
