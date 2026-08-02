"""Train the taste-list virtual cocktail graph decoder from local CSV data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sqlalchemy import select

from app.embedding_pipeline.core import load_vector_artifact, read_taste_dataset
from app.taste_query.training import (
    GraphDecoderConfig,
    save_runtime_artifact,
    train_graph_decoder,
)
from app.taste_query.model import load_taste_query_model
from app.taste_query.vocabulary import DESCRIPTOR_CODES, extract_descriptor_codes

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "taste_query"
    / "artifacts"
    / "taste-query-gnn.npz"
)
DEFAULT_METRICS = DEFAULT_MODEL.with_suffix(".metrics.json")


def _ingredient_weight(amount: float | None, unit: str | None) -> float:
    value = float(amount) if amount is not None else 1.0
    normalized_unit = (unit or "").strip().lower()
    factor = {"ml": 1.0, "oz": 29.5735, "tsp": 4.92892, "dash": 0.92}.get(
        normalized_unit,
        1.0,
    )
    return float(np.log1p(max(value * factor, 0.1)))


def _load_graph_data(taste_csv: Path, embedding_path: Path):
    from cocktail_mate_db.models import CocktailIngredient, Ingredient  # noqa: PLC0415

    from app.core.database import SessionLocal  # noqa: PLC0415

    dataset = read_taste_dataset(taste_csv)
    targets = load_vector_artifact(embedding_path)
    if not np.array_equal(dataset.cocktail_ids, targets.cocktail_ids):
        raise ValueError("taste CSV and 32D artifact cocktail IDs do not match")
    code_index = {code: index for index, code in enumerate(DESCRIPTOR_CODES)}
    descriptor_matrix = np.zeros(
        (len(dataset.cocktail_ids), len(DESCRIPTOR_CODES)),
        dtype=np.float32,
    )
    edge_rows: list[dict[str, object]] = []
    for row, (cocktail_id, text) in enumerate(
        zip(dataset.cocktail_ids, dataset.embedding_texts, strict=True)
    ):
        codes = extract_descriptor_codes(text)
        if not codes:
            raise ValueError(f"cocktail {cocktail_id} has no controlled taste terms")
        descriptor_matrix[row, [code_index[code] for code in codes]] = 1.0
        edge_rows.append(
            {"cocktail_id": int(cocktail_id), "descriptor_codes": list(codes)}
        )

    with SessionLocal() as session:
        rows = session.execute(
            select(
                CocktailIngredient.cocktail_id,
                CocktailIngredient.ingredient_id,
                CocktailIngredient.amount,
                CocktailIngredient.unit,
                Ingredient.name,
            ).join(Ingredient, Ingredient.id == CocktailIngredient.ingredient_id)
        ).all()
    ingredient_ids = tuple(sorted({int(row.ingredient_id) for row in rows}))
    ingredient_index = {
        ingredient_id: index for index, ingredient_id in enumerate(ingredient_ids)
    }
    cocktail_index = {
        int(cocktail_id): index
        for index, cocktail_id in enumerate(dataset.cocktail_ids)
    }
    ingredient_matrix = np.zeros(
        (len(dataset.cocktail_ids), len(ingredient_ids)),
        dtype=np.float32,
    )
    for relation in rows:
        cocktail_id = int(relation.cocktail_id)
        if cocktail_id not in cocktail_index:
            continue
        ingredient_matrix[
            cocktail_index[cocktail_id],
            ingredient_index[int(relation.ingredient_id)],
        ] = _ingredient_weight(relation.amount, relation.unit)
    return descriptor_matrix, ingredient_matrix, targets.vectors, edge_rows


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train(arguments: argparse.Namespace) -> dict:
    descriptor_matrix, ingredient_matrix, targets, edge_rows = _load_graph_data(
        arguments.taste_csv,
        arguments.embedding_32,
    )
    result = train_graph_decoder(
        descriptor_matrix,
        ingredient_matrix,
        targets,
        GraphDecoderConfig(
            hidden_dim=arguments.hidden_dim,
            max_epochs=arguments.epochs,
            patience=arguments.patience,
            restarts=arguments.restarts,
            seed=arguments.seed,
            device=arguments.device,
        ),
    )
    save_runtime_artifact(
        arguments.output,
        descriptor_codes=DESCRIPTOR_CODES,
        descriptor_support=(descriptor_matrix > 0).sum(axis=0),
        result=result,
    )
    model = load_taste_query_model(arguments.output)
    target_unit = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    edge_sets = [set(row["descriptor_codes"]) for row in edge_rows]
    precisions: list[float] = []
    baselines: list[float] = []
    for code in sorted(model.supported_codes):
        query = model.encode([code])
        nearest = np.argsort(-(target_unit @ query))[:5]
        precisions.append(sum(code in edge_sets[index] for index in nearest) / 5)
        baselines.append(sum(code in values for values in edge_sets) / len(edge_sets))
    descriptor_evaluation = {
        "minimum_support": 5,
        "evaluated_descriptors": len(precisions),
        "macro_top5_descriptor_precision": float(np.mean(precisions)),
        "macro_catalog_prevalence": float(np.mean(baselines)),
        "precision_lift": float(np.mean(precisions) / np.mean(baselines)),
    }
    report = result.report | {"single_descriptor_queries": descriptor_evaluation}
    _write_json(arguments.metrics, report)
    return report | {
        "model_output": str(arguments.output),
        "metrics_output": str(arguments.metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument(
        "--taste-csv",
        type=Path,
        default=Path("taste-data/cocktail-taste-descriptions.csv"),
    )
    training.add_argument(
        "--embedding-32",
        type=Path,
        default=Path("embedding-artifacts/embeddings-32.npz"),
    )
    training.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    training.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    training.add_argument("--hidden-dim", type=int, default=96)
    training.add_argument("--epochs", type=int, default=1600)
    training.add_argument("--patience", type=int, default=250)
    training.add_argument("--restarts", type=int, default=3)
    training.add_argument("--seed", type=int, default=20260801)
    training.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    result = train(arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
