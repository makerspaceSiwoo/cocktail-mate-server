"""Learn 32D coordinates directly from the 512D cosine-neighbor graph."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.embedding_pipeline.core import l2_normalize, neighbor_metrics, write_json


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    input_dim: int = 512
    output_dim: int = 32
    teacher_k: int = 15
    teacher_temperature: float = 0.02
    student_temperature: float = 0.05
    learning_rate: float = 0.03
    max_epochs: int = 2000
    patience: int = 300
    restarts: int = 4
    seed: int = 20260801
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    vectors: np.ndarray
    report: dict[str, Any]


def _load_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install requirements-embedding.txt"
        ) from error
    return torch


def train_projection(
    source_vectors: np.ndarray,
    config: ProjectionConfig = ProjectionConfig(),
) -> ProjectionResult:
    """Optimize one normalized 32D coordinate per catalog item.

    This is intentionally transductive: its exact objective is preservation of the
    current catalog's cosine-neighbor graph. When cocktails change, rebuild all 32D
    coordinates together instead of projecting one new row independently.
    """
    torch = _load_torch()
    source = l2_normalize(source_vectors)
    if source.shape[1] != config.input_dim:
        raise ValueError(
            f"expected {config.input_dim} input dimensions, got {source.shape[1]}"
        )
    if len(source) < max(config.teacher_k + 2, 20):
        raise ValueError("at least teacher_k + 2 rows and 20 total rows are required")
    if config.restarts <= 0 or config.max_epochs <= 0 or config.patience <= 0:
        raise ValueError("restarts, max_epochs, and patience must be positive")
    if config.teacher_temperature <= 0 or config.student_temperature <= 0:
        raise ValueError("temperatures must be positive")

    device = torch.device(config.device)
    source_tensor = torch.as_tensor(source, dtype=torch.float32, device=device)
    similarities = source_tensor @ source_tensor.T
    similarities.fill_diagonal_(-torch.inf)
    values, indices = torch.topk(similarities, k=config.teacher_k, dim=1)
    weights = torch.softmax(values / config.teacher_temperature, dim=1)
    teacher = torch.zeros_like(similarities)
    teacher.scatter_(1, indices, weights)
    diagonal = torch.eye(len(source), dtype=torch.bool, device=device)

    best_vectors: np.ndarray | None = None
    best_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    restart_reports: list[dict[str, Any]] = []
    for restart in range(config.restarts):
        restart_seed = config.seed + restart
        _set_seed(restart_seed)
        parameters = torch.nn.Parameter(
            torch.randn(
                (len(source), config.output_dim),
                dtype=torch.float32,
                device=device,
            )
        )
        optimizer = torch.optim.Adam([parameters], lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * 0.05,
        )
        best_restart_vectors: np.ndarray | None = None
        best_restart_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
        best_restart_epoch = 0
        stale_epochs = 0
        final_loss = float("nan")

        for epoch in range(1, config.max_epochs + 1):
            coordinates = torch.nn.functional.normalize(parameters, dim=1)
            logits = coordinates @ coordinates.T / config.student_temperature
            logits = logits.masked_fill(diagonal, -torch.inf)
            log_probabilities = torch.log_softmax(logits, dim=1)
            log_probabilities = log_probabilities.masked_fill(diagonal, 0.0)
            loss = -(teacher * log_probabilities).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.detach().cpu())

            if epoch != 1 and epoch % 25 != 0:
                continue
            with torch.no_grad():
                candidate = (
                    torch.nn.functional.normalize(parameters, dim=1).cpu().numpy()
                )
            metrics = neighbor_metrics(source, candidate)
            score_key = (
                metrics["recall@5"],
                metrics["recall@10"],
                metrics["recall@15"],
            )
            if score_key > best_restart_key:
                best_restart_key = score_key
                best_restart_epoch = epoch
                best_restart_vectors = candidate.copy()
                stale_epochs = 0
            else:
                stale_epochs += 25
                if stale_epochs >= config.patience:
                    break

        assert best_restart_vectors is not None
        metrics = neighbor_metrics(source, best_restart_vectors)
        restart_reports.append(
            {
                "restart": restart,
                "seed": restart_seed,
                "best_epoch": best_restart_epoch,
                "final_loss": final_loss,
                "metrics": metrics,
            }
        )
        if best_restart_key > best_key:
            best_key = best_restart_key
            best_vectors = best_restart_vectors

    assert best_vectors is not None
    report: dict[str, Any] = {
        "algorithm": "transductive_cosine_knn_distribution_embedding",
        "config": asdict(config),
        "selection_rule": "maximize recall@5, then recall@10, then recall@15",
        "all_data_metrics": neighbor_metrics(source, best_vectors),
        "restarts": restart_reports,
        "new_item_policy": "retrain the full catalog",
    }
    return ProjectionResult(l2_normalize(best_vectors), report)


def save_projection_model(
    weights_path: Path,
    manifest_path: Path,
    *,
    vectors: np.ndarray,
    cocktail_ids: np.ndarray,
    config: ProjectionConfig,
    report: dict[str, Any],
    source_metadata: dict[str, Any],
) -> None:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError(
            "safetensors and PyTorch are required; install requirements-embedding.txt"
        ) from error
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "cocktail_ids": torch.as_tensor(cocktail_ids, dtype=torch.int64),
            "coordinates": torch.as_tensor(vectors, dtype=torch.float32),
        },
        str(weights_path),
    )
    write_json(
        manifest_path,
        {
            "architecture": "catalog-indexed normalized 32D coordinates",
            "config": asdict(config),
            "source": source_metadata,
            "training": report,
            "weights": weights_path.name,
            "new_item_policy": "retrain the full catalog",
        },
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _load_torch()
    torch.manual_seed(seed)
