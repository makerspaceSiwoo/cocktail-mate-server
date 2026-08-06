# Sensory embedding boundary

This package is a pure, local-only boundary. It validates supplied teacher
probabilities and deterministically builds cocktail/query vectors. It does not
import database or startup code, load a model or file, or make a network call.

## Immutable v2 registry and raw240

`SENSORY_V2_REGISTRY` pins 48 axes, their order and categories, registry
version, source SHA-256, and the exact level order `A, B, C, D, E`. Its registry
SHA-256 commits to that complete contract.

`Raw240` preserves each teacher distribution without renormalizing it. The only
valid flattening is:

```text
registry axis 0: p_A, p_B, p_C, p_D, p_E
registry axis 1: p_A, p_B, p_C, p_D, p_E
...
registry axis 47: p_A, p_B, p_C, p_D, p_E
```

Every group must contain five finite probabilities in `[0, 1]` whose sum is
one. `source_sha256` hashes all 240 canonical values together with the registry
SHA, A–E order, and raw schema.

## Two distinct 48D projections

`project_teacher_soft_labels(cocktail_id, axes)` returns an immutable
`TeacherEmbeddingBundle` containing the retained `Raw240`, a `Graph48`, and a
`Preference48`. These spaces must not be interchanged.

`Graph48` is for cocktail-to-cocktail cosine topology:

```text
x_j = E[p_j; (0, .25, .5, .75, 1)]
b_j = x_j * sqrt(1 / size(category(j)))
graph48 = b / ||b||_2
```

It carries the `sensory-graph-48` cosine contract and always has unit L2 norm.
An all-zero expected-intensity result raises `ZeroGraph48VectorError` and is
quarantined instead of emitting an invalid graph vector.

`Preference48` is for user-to-cocktail maximum inner-product search (MIPS).
Only these five axes use `(0, .2, .65, 1, .8)`:

- `sweetness`
- `saltiness`
- `sourness`
- `bitterness`
- `umami`

The other 43 axes use the monotonic `(0, .25, .5, .75, 1)` scale. In
particular, `pungency`, `astringency`, and `fattiness` are monotonic.
Preference48 is deliberately **not** L2-normalized and carries the separate
`sensory-preference-48` inner-product contract.

Each derived vector has a contract-bound content hash. The bundle retains the
raw source hash on both projections and adds a `provenance_sha256` binding the
cocktail ID, registry, raw source, projection contracts, and both vector hashes.

## User query and MIPS

`build_user_query(selected_axis_weights)` accepts a mapping such as:

```python
query = build_user_query(
    {
        "sweetness": 2.0,
        "saltiness": 1.0,
        "citrus_fruit": 1.0,
    }
)
```

For active category set `C`, it computes:

```text
q_j = (1 / |C|) * w_j / sum(weights in category(j))  for selected j
q_j = 0                                                 otherwise
```

Thus each active category receives equal mass, weights are L1-normalized within
the category, all unselected axes remain exactly zero, and `||q||_1 = 1`.
Empty input, unknown or duplicate axes, and non-positive/non-finite weights are
rejected.

`score_preference_mips(query, cocktails, k)` requires exact preference48
contract equality, returns exact inner-product scores sorted by descending
score then numeric cocktail ID, and exposes the equivalent pgvector
negative-inner-product distance. A graph48 vector is rejected at this boundary.

`TeacherSoftLabelProjector`, `sensory_48_contract`, and
`SensoryPositiveQueryEncoder` remain as backward-compatible preference48
adapters. They follow the corrected five-axis nonmonotonic policy.
