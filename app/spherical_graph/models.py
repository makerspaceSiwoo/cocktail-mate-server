"""Serializable contracts for deterministic cocktail graphs on S²."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


NodeKind = Literal["cocktail", "hub"]
EdgeKind = Literal["cocktail_knn", "hub_anchor", "hub_mst"]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One immutable high-dimensional source vector."""

    node_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DirectedNeighbor:
    source_id: str
    target_id: str
    rank: int
    similarity: float
    target_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rank": self.rank,
            "similarity": self.similarity,
            "target_distance": self.target_distance,
        }


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_kind: NodeKind
    component_id: str
    x: float
    y: float
    z: float
    visible: bool
    recommendable: bool
    medoid_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "component_id": self.component_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visible": self.visible,
            "recommendable": self.recommendable,
            "medoid_id": self.medoid_id,
        }


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    edge_kind: EdgeKind
    similarity: float
    target_distance: float
    source_rank: int | None
    target_rank: int | None
    is_mutual: bool
    is_bridge: bool
    visible: bool
    recommendable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_kind": self.edge_kind,
            "similarity": self.similarity,
            "target_distance": self.target_distance,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "is_mutual": self.is_mutual,
            "is_bridge": self.is_bridge,
            "visible": self.visible,
            "recommendable": self.recommendable,
        }


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    component_id: str
    hub_id: str
    member_ids: tuple[str, ...]
    medoid_id: str
    centroid: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "member_ids": list(self.member_ids),
            "member_count": len(self.member_ids),
            "medoid_id": self.medoid_id,
        }


@dataclass(frozen=True, slots=True)
class SphericalGraph:
    """Public cocktail graph; private force-layout hubs are never retained."""

    k: int
    seed: int
    similarity_provider: str
    clusterer: str
    directed_neighbors: tuple[DirectedNeighbor, ...]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    components: tuple[ComponentSummary, ...]
    layout_report: dict[str, Any]

    def node_rows(self, *, include_hidden: bool = False) -> list[dict[str, Any]]:
        """Return public cocktail nodes.

        ``include_hidden`` is retained as a compatibility keyword, but private
        layout hubs are never stored in or serialized from this artifact.
        """
        del include_hidden
        return [node.to_dict() for node in self.nodes if node.visible]

    def edge_rows(self, *, include_hidden: bool = False) -> list[dict[str, Any]]:
        """Return public cocktail union edges only."""
        del include_hidden
        return [edge.to_dict() for edge in self.edges if edge.visible]

    def to_dict(self, *, include_hidden: bool = False) -> dict[str, Any]:
        del include_hidden
        return {
            "schema_version": 2,
            "k": self.k,
            "seed": self.seed,
            "similarity_provider": self.similarity_provider,
            "clusterer": self.clusterer,
            "directed_neighbors": [
                neighbor.to_dict() for neighbor in self.directed_neighbors
            ],
            "nodes": self.node_rows(),
            "edges": self.edge_rows(),
            "components": [component.to_dict() for component in self.components],
            "layout_report": self.layout_report,
        }
