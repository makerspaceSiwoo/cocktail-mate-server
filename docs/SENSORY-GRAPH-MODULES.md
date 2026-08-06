# Sensory graph module boundaries

This implementation is offline. It does not connect to PostgreSQL, mutate
production data, or call Gemini or another network API.

## The three modules

### 1. `app.sensory_embedding`

Owns versioned cocktail/query vectors, validation, and content hashes.

- `legacy-cocktail-32` adapts the current normalized 32D vectors.
- `sensory-preference-48` defines the future 48-axis sensory-v2 space.
- The future teacher boundary preserves the ordered
  `48 axes x 5 levels = 240D` soft-label probabilities and their content hash,
  then deterministically projects one affinity value per axis.
- Positive-only selections become a category-balanced query with L1 norm one.
- A 48D query is ranked against raw 48D cocktail vectors by inner product.

This module does not call a teacher or load a model, file, or database.

### 2. `app.vector_similarity`

Owns exact vector comparisons and the canonical recommendation topology.

- Exact cosine is used for cocktail-to-cocktail graph neighbors.
- Inner-product ranking has pgvector-compatible negative-distance semantics.
- `CanonicalNeighborArtifact` computes numeric-ID-tied exact-cosine directed
  top-k rows and their either-direction undirected union once.
- The artifact binds ordered vectors with a vector-set SHA-256 and validates
  ranks, scores, asymmetry, mutual flags, and union completeness.
- ANN is an explicit future adapter, not a silent fallback.

This module accepts structural vector records and has no dependency on module 1.

### 3. `app.spherical_graph`

Owns the public cocktail graph artifact and deterministic layout on the unit
sphere.

- The production adapter consumes module 2's directed rows and visible union
  edges without independently deriving top-k, ranks, or the union.
- Cluster assignments and a private hub graph are fixed before layout.
- Fibonacci-initialized hubs, cluster-local cocktail initialization,
  hub-to-every-member cosine anchors, and the hub MST participate in every
  force iteration.
- The S² solver receives only cocktail IDs, weighted union edges, the fixed
  cluster partition, and fixed private hub edges.
- Returned 3D coordinates are visualization coordinates, not recommendation
  vectors.

Optional imports of modules 1 and 2 remain isolated in
`app.spherical_graph.adapters`.

## The S² layout is not dimensionality reduction

The S² stage does not read source embeddings or a full similarity matrix. It
does not use PCA, UMAP, t-SNE, MDS, or random projection.

`prepare_spherical_graph_topology` is an explicit pre-layout adapter. It may
validate canonical rows and precompute cluster/hub metadata. It then calls
`build_spherical_graph_from_topology`, whose force boundary is graph-only. The
standalone `build_spherical_graph` compatibility helper follows the same rule:
it derives topology first and passes only the fixed graph into the solver.

An optional full matrix may be supplied as `audit_similarities`. It is read only
after coordinates and multistart selection are final. Changing audit scores
cannot change coordinates, the selected start, or the coordinate hash.

## Recommendation and graph single source

The recommendation source is only
`canonical_neighbors.directed_neighbors`. For cocktail `source_id`, outgoing
rows have ranks 1 through 5 and numeric `target_id` breaks score ties.

Graph-visible edges are exactly `canonical_neighbors.undirected_edges`: an edge
exists when either endpoint selected the other. Visible undirected degree may
therefore exceed five, one endpoint rank may be `null`, and an edge need not be
mutual. Visible edges must not be read as another recommendation list.

Private hubs never enter `SphericalGraph.nodes`, `SphericalGraph.edges`,
`node_rows`, `edge_rows`, or JSON. The public graph contains cocktail nodes and
canonical cocktail union edges, with zero hub IDs and zero hub edges. Only
numeric private-hub counts appear in layout diagnostics.

## Deterministic graph-only layout

Production runs 16 deterministic starts. Each start keeps hub initialization on
the Fibonacci sphere, initializes cocktail nodes locally around their cluster
hub, and optimizes these graph-only constraints:

- a union edge targets exactly `acos(clamp(edge cosine, -1, 1))` radians;
- sampled graph nonedges receive topology-only repulsion;
- graph neighbors rank ahead of sampled graph nonedges;
- every private hub-to-member cosine anchor and every hub-MST edge remains
  active;
- every update projects coordinates back to unit S².

Multistart selection uses only graph objective terms. The seed sequence,
per-start objectives, selected start, and SHA-256 of canonical hexadecimal
coordinates are recorded. Tests may explicitly lower `multistart_count`; the
production default remains 16.

## Exhaustive quality report and gates

After selection, every cocktail node and every union edge is evaluated:

- mean coordinate Recall@5 against canonical directed neighbors must be at
  least `0.60`;
- the fraction of nodes with at least one true top-5 neighbor in their
  coordinate top five must be at least `0.90`;
- for every source, cosine-bottom-decile graph nonneighbors that are closer
  than that source's farthest true top-5 coordinate neighbor must number `0`
  when optional audit similarities are supplied;
- union-edge angular target RMSE must be at most `0.40` radians;
- maximum unit-norm error must be at most `1e-12`;
- a deterministic coordinate SHA-256 is always reported.

By default, a failed evaluated gate raises `SphericalLayoutQualityError`.
`SphericalGraphConfig(report_only=True)` returns the complete report for
diagnosis. The offline artifact CLI is diagnostic by default and records failed
gates without changing thresholds; pass `--enforce-quality` for production
promotion. `--report-only` is accepted explicitly for clarity.

## Offline CLI

Build from an existing local CSV or NPZ:

```bash
python -m scripts.build_spherical_graph \
  --input embedding-artifacts/embeddings-32.csv \
  --output embedding-artifacts/spherical-graph.json \
  --k 5 \
  --clusters 7 \
  --seed 20260806 \
  --iterations 450 \
  --multistarts 16
```

CSV input must have exactly
`cocktail_id,cocktail_name_ko,embedding`; `embedding` is a JSON numeric array.
The loader rejects missing or extra columns, non-positive or duplicate IDs,
empty names, non-finite or zero vectors, dimension mismatches, and fewer than
`k + 1` rows. Existing `.npz` vector artifacts remain supported.

`--clusters 7` is the default deterministic cosine k-medoids preprocessing
policy. Set `--clusters 0` to use connected components of the union graph.

The command hashes the input before and after construction and aborts if it
changes. It writes JSON atomically with source format, SHA-256, row count,
vector dimension, cocktail ID/name mapping, canonical neighbor provenance,
canonical recommendation rows, the canonical union, and the public S² graph.
The graph JSON contains no private hub identifier or edge. Diagnostic CLI runs
write failed metrics visibly; production promotion must add
`--enforce-quality`. Input and output paths must differ.

## Deployment status

Modules 1 and 2 are not connected to production recommendation or database
code. They are local, tested replacement boundaries only. No schema migration,
database write, or live query behavior changes in this work.

The future sensory path becomes production-eligible only after sensory data
generation and evaluation gates are complete. Weighted Wasserstein graph
distance and a soft-label reranker are not implemented here.
