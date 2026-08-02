"""Offline heterogeneous graph training for taste-list query embeddings."""

# TODO: 재료 임베딩 학습 필요. 현재 재료 경로는 보조 손실에만 사용하며,
# DB에서 사용할 재료별 32D 임베딩은 학습 산출물로 내보내거나 동기화하지 않는다.

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GraphDecoderConfig:
    hidden_dim: int = 96
    teacher_k: int = 15
    teacher_temperature: float = 0.03
    student_temperature: float = 0.05
    descriptor_dropout: float = 0.25
    learning_rate: float = 0.02
    max_epochs: int = 1600
    patience: int = 250
    restarts: int = 3
    validation_fraction: float = 0.2
    seed: int = 20260801
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class GraphDecoderResult:
    state: dict[str, np.ndarray]
    report: dict[str, Any]


def _load_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for taste GNN training") from error
    return torch


def _row_normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        missing = np.flatnonzero(totals[:, 0] <= 0).tolist()
        raise ValueError(f"graph rows without edges: {missing}")
    return matrix / totals


def _query_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    rows: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    target = np.asarray(targets, dtype=np.float32)
    predicted = np.asarray(predictions, dtype=np.float32)
    target_similarity = target @ target.T
    query_similarity = predicted @ target.T
    cosine_values: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    for row in rows.tolist():
        target_similarity[row, row] = -np.inf
        query_similarity[row, row] = -np.inf
        cosine_values.append(float(np.dot(predicted[row], target[row])))
        for k in ks:
            expected = np.argpartition(-target_similarity[row], k)[:k]
            actual = np.argpartition(-query_similarity[row], k)[:k]
            recalls[k].append(len(set(expected).intersection(actual)) / k)
    return {
        "mean_target_cosine": float(np.mean(cosine_values)),
        **{f"recall@{k}": float(np.mean(values)) for k, values in recalls.items()},
    }


def train_graph_decoder(
    descriptor_matrix: np.ndarray,
    ingredient_matrix: np.ndarray,
    target_vectors: np.ndarray,
    config: GraphDecoderConfig = GraphDecoderConfig(),
) -> GraphDecoderResult:
    """Train taste/ingredient node messages against catalog 32D coordinates."""

    torch = _load_torch()
    descriptors = _row_normalize(descriptor_matrix)
    ingredients = _row_normalize(ingredient_matrix)
    targets = _row_normalize_l2(target_vectors)
    row_count = len(targets)
    if len(descriptors) != row_count or len(ingredients) != row_count:
        raise ValueError("descriptor, ingredient, and target rows must match")
    if targets.shape[1] != 32:
        raise ValueError("taste graph target vectors must be 32D")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")

    generator = np.random.default_rng(config.seed)
    shuffled = generator.permutation(row_count)
    validation_count = max(1, round(row_count * config.validation_fraction))
    validation_rows = np.sort(shuffled[:validation_count])
    training_rows = np.sort(shuffled[validation_count:])
    device = torch.device(config.device)
    descriptor_tensor = torch.as_tensor(descriptors, device=device)
    ingredient_tensor = torch.as_tensor(ingredients, device=device)
    target_tensor = torch.as_tensor(targets, device=device)
    training_index = torch.as_tensor(training_rows, dtype=torch.long, device=device)
    diagonal = torch.eye(row_count, dtype=torch.bool, device=device)
    target_similarity = target_tensor @ target_tensor.T
    target_similarity = target_similarity.masked_fill(diagonal, -torch.inf)
    teacher_values, teacher_indices = torch.topk(
        target_similarity,
        k=config.teacher_k,
        dim=1,
    )
    teacher_weights = torch.softmax(
        teacher_values / config.teacher_temperature,
        dim=1,
    )

    class HeterogeneousTasteGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.taste_nodes = torch.nn.Parameter(
                torch.randn(descriptors.shape[1], config.hidden_dim) * 0.1
            )
            self.ingredient_nodes = torch.nn.Parameter(
                torch.randn(ingredients.shape[1], config.hidden_dim) * 0.1
            )
            self.taste_hidden = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
            self.ingredient_hidden = torch.nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            )
            self.output = torch.nn.Linear(config.hidden_dim, 32)

        def forward(self, taste_edges, ingredient_edges):
            taste_message = taste_edges @ self.taste_nodes
            ingredient_message = ingredient_edges @ self.ingredient_nodes
            taste_hidden = torch.tanh(self.taste_hidden(taste_message))
            ingredient_hidden = torch.tanh(self.ingredient_hidden(ingredient_message))
            taste_output = torch.nn.functional.normalize(
                self.output(taste_hidden),
                dim=1,
            )
            ingredient_output = torch.nn.functional.normalize(
                self.output(ingredient_hidden),
                dim=1,
            )
            return taste_output, ingredient_output, taste_hidden, ingredient_hidden

    def distribution_loss(predictions, rows):
        logits = predictions @ target_tensor.T / config.student_temperature
        logits = logits.masked_fill(diagonal, -torch.inf)
        log_probabilities = torch.log_softmax(logits, dim=1)
        selected = log_probabilities.gather(1, teacher_indices)
        return -(teacher_weights[rows] * selected[rows]).sum(dim=1).mean()

    best_state: dict[str, np.ndarray] | None = None
    best_key = (-1.0, -1.0)
    best_epoch = 0
    restart_reports: list[dict[str, Any]] = []
    for restart in range(config.restarts):
        restart_seed = config.seed + restart
        random.seed(restart_seed)
        np.random.seed(restart_seed)
        torch.manual_seed(restart_seed)
        model = HeterogeneousTasteGraph().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * 0.05,
        )
        stale_epochs = 0
        restart_best_key = (-1.0, -1.0)
        restart_best_epoch = 0
        restart_best_state: dict[str, np.ndarray] | None = None
        final_loss = float("nan")
        for epoch in range(1, config.max_epochs + 1):
            dropped = descriptor_tensor.clone()
            keep = torch.rand_like(dropped) >= config.descriptor_dropout
            dropped = dropped * keep
            empty = dropped.sum(dim=1) <= 0
            if bool(empty.any()):
                dropped[empty] = descriptor_tensor[empty]
            dropped = dropped / dropped.sum(dim=1, keepdim=True)
            taste, ingredient, taste_hidden, ingredient_hidden = model(
                dropped,
                ingredient_tensor,
            )
            taste_cosine = (
                1.0
                - (taste[training_index] * target_tensor[training_index])
                .sum(dim=1)
                .mean()
            )
            ingredient_cosine = (
                1.0
                - (ingredient[training_index] * target_tensor[training_index])
                .sum(dim=1)
                .mean()
            )
            taste_neighbors = distribution_loss(taste, training_index)
            ingredient_neighbors = distribution_loss(ingredient, training_index)
            alignment = (
                1.0
                - torch.nn.functional.cosine_similarity(
                    taste_hidden[training_index],
                    ingredient_hidden[training_index],
                    dim=1,
                ).mean()
            )
            loss = (
                taste_cosine
                + taste_neighbors
                + 0.35 * ingredient_cosine
                + 0.35 * ingredient_neighbors
                + 0.1 * alignment
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.detach().cpu())

            if epoch != 1 and epoch % 25 != 0:
                continue
            with torch.no_grad():
                validation_taste = model(descriptor_tensor, ingredient_tensor)[0]
            validation_metrics = _query_metrics(
                targets,
                validation_taste.detach().cpu().numpy(),
                validation_rows,
            )
            score_key = (
                validation_metrics["recall@5"],
                validation_metrics["mean_target_cosine"],
            )
            if score_key > restart_best_key:
                restart_best_key = score_key
                restart_best_epoch = epoch
                restart_best_state = _export_state(model)
                stale_epochs = 0
            else:
                stale_epochs += 25
                if stale_epochs >= config.patience:
                    break

        assert restart_best_state is not None
        evaluation_model = HeterogeneousTasteGraph().to(device)
        _import_state(evaluation_model, restart_best_state, device)
        with torch.no_grad():
            taste_prediction, ingredient_prediction, _, _ = evaluation_model(
                descriptor_tensor,
                ingredient_tensor,
            )
        taste_array = taste_prediction.cpu().numpy()
        ingredient_array = ingredient_prediction.cpu().numpy()
        report = {
            "restart": restart,
            "seed": restart_seed,
            "best_epoch": restart_best_epoch,
            "final_loss": final_loss,
            "validation_taste": _query_metrics(
                targets,
                taste_array,
                validation_rows,
            ),
            "validation_ingredient": _query_metrics(
                targets,
                ingredient_array,
                validation_rows,
            ),
        }
        restart_reports.append(report)
        if restart_best_key > best_key:
            best_key = restart_best_key
            best_epoch = restart_best_epoch
            best_state = restart_best_state

    assert best_state is not None
    final_state = _fit_all_rows(
        descriptors,
        ingredients,
        targets,
        best_epoch,
        config,
    )
    final_model = HeterogeneousTasteGraph().to(device)
    _import_state(final_model, final_state, device)
    with torch.no_grad():
        final_taste, final_ingredient, _, _ = final_model(
            descriptor_tensor,
            ingredient_tensor,
        )
    report = {
        "algorithm": "heterogeneous_taste_ingredient_graph_distillation",
        "config": asdict(config),
        "rows": row_count,
        "taste_nodes": descriptors.shape[1],
        "ingredient_nodes": ingredients.shape[1],
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "selection_rule": "maximize held-out query recall@5, then target cosine",
        "selected_epoch": best_epoch,
        "restarts": restart_reports,
        "all_rows_final_taste": _query_metrics(
            targets,
            final_taste.cpu().numpy(),
            np.arange(row_count),
        ),
        "all_rows_final_ingredient": _query_metrics(
            targets,
            final_ingredient.cpu().numpy(),
            np.arange(row_count),
        ),
    }
    return GraphDecoderResult(final_state, report)


def _fit_all_rows(
    descriptors: np.ndarray,
    ingredients: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    config: GraphDecoderConfig,
) -> dict[str, np.ndarray]:
    """Retrain the selected architecture on the complete catalog."""

    torch = _load_torch()
    device = torch.device(config.device)
    torch.manual_seed(config.seed + 10_000)

    class FinalGraph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.taste_nodes = torch.nn.Parameter(
                torch.randn(descriptors.shape[1], config.hidden_dim) * 0.1
            )
            self.ingredient_nodes = torch.nn.Parameter(
                torch.randn(ingredients.shape[1], config.hidden_dim) * 0.1
            )
            self.taste_hidden = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
            self.ingredient_hidden = torch.nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            )
            self.output = torch.nn.Linear(config.hidden_dim, 32)

        def forward(self, taste_edges, ingredient_edges):
            taste_hidden = torch.tanh(self.taste_hidden(taste_edges @ self.taste_nodes))
            ingredient_hidden = torch.tanh(
                self.ingredient_hidden(ingredient_edges @ self.ingredient_nodes)
            )
            return (
                torch.nn.functional.normalize(self.output(taste_hidden), dim=1),
                torch.nn.functional.normalize(self.output(ingredient_hidden), dim=1),
                taste_hidden,
                ingredient_hidden,
            )

    descriptor_tensor = torch.as_tensor(descriptors, device=device)
    ingredient_tensor = torch.as_tensor(ingredients, device=device)
    target_tensor = torch.as_tensor(targets, device=device)
    diagonal = torch.eye(len(targets), dtype=torch.bool, device=device)
    teacher_similarity = (target_tensor @ target_tensor.T).masked_fill(
        diagonal,
        -torch.inf,
    )
    values, indices = torch.topk(
        teacher_similarity,
        k=config.teacher_k,
        dim=1,
    )
    weights = torch.softmax(values / config.teacher_temperature, dim=1)
    model = FinalGraph().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=config.learning_rate * 0.05,
    )
    for _ in range(max(epochs, 1)):
        dropped = descriptor_tensor.clone()
        keep = torch.rand_like(dropped) >= config.descriptor_dropout
        dropped *= keep
        empty = dropped.sum(dim=1) <= 0
        dropped[empty] = descriptor_tensor[empty]
        dropped /= dropped.sum(dim=1, keepdim=True)
        taste, ingredient, taste_hidden, ingredient_hidden = model(
            dropped,
            ingredient_tensor,
        )

        def neighbor_loss(prediction):
            logits = (
                prediction @ target_tensor.T / config.student_temperature
            ).masked_fill(
                diagonal,
                -torch.inf,
            )
            selected = torch.log_softmax(logits, dim=1).gather(1, indices)
            return -(weights * selected).sum(dim=1).mean()

        loss = (
            1.0
            - (taste * target_tensor).sum(dim=1).mean()
            + neighbor_loss(taste)
            + 0.35 * (1.0 - (ingredient * target_tensor).sum(dim=1).mean())
            + 0.35 * neighbor_loss(ingredient)
            + 0.1
            * (
                1.0
                - torch.nn.functional.cosine_similarity(
                    taste_hidden,
                    ingredient_hidden,
                    dim=1,
                ).mean()
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return _export_state(model)


def _export_state(model) -> dict[str, np.ndarray]:
    state = model.state_dict()
    return {name: value.detach().cpu().numpy().copy() for name, value in state.items()}


def _import_state(model, state: dict[str, np.ndarray], device) -> None:
    torch = _load_torch()
    model.load_state_dict(
        {name: torch.as_tensor(value, device=device) for name, value in state.items()}
    )


def _row_normalize_l2(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("target vectors contain zero rows")
    return matrix / norms


def save_runtime_artifact(
    path: Path,
    *,
    descriptor_codes: tuple[str, ...],
    descriptor_support: np.ndarray,
    result: GraphDecoderResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = result.state
    np.savez_compressed(
        path,
        descriptor_codes=np.asarray(descriptor_codes),
        descriptor_support=np.asarray(descriptor_support, dtype=np.int32),
        node_embeddings=state["taste_nodes"],
        hidden_weight=state["taste_hidden.weight"].T,
        hidden_bias=state["taste_hidden.bias"],
        output_weight=state["output.weight"].T,
        output_bias=state["output.bias"],
    )
