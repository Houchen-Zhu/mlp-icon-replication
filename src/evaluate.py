import csv
import os

import numpy as np
import torch

from .config import ICONConfig
from .data_generation import generate_from_config
from .dataset import ICONDataset
from .model import MLPICON
from .train import select_device, set_seed


def load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(config: ICONConfig, device: torch.device) -> MLPICON:
    checkpoint = load_checkpoint(config.final_checkpoint_path, device)
    model = MLPICON(input_dim=config.input_dim, output_dim=config.output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def relative_l2_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.norm(y_pred - y_true) / (torch.norm(y_true) + 1e-8)


def evaluate(config: ICONConfig, num_test_tasks: int = 500) -> dict[str, float]:
    set_seed(config.seed)
    device = select_device()
    print(f"Using device: {device}")
    print("Generating unseen test data...")

    u_paths, y_paths, params, _ = generate_from_config(
        config=config,
        n_regimes=config.n_regimes_test,
        seed=config.seed + 10_000,
    )
    dataset = ICONDataset(
        u_paths=u_paths,
        y_paths=y_paths,
        params=params,
        context_size=config.context_size,
        n_tasks=num_test_tasks,
        seed=config.seed + 20_000,
    )

    model = load_model(config, device)
    errors = []

    with torch.no_grad():
        for _ in range(num_test_tasks):
            x_input, y_true = dataset.sample_task()

            # x_input shape [1100] -> model input shape [1, 1100].
            # x_input 形状 [1100] -> 模型输入形状 [1, 1100]。
            x_input = x_input.unsqueeze(0).to(device)
            y_true = y_true.to(device)

            # y_pred shape [100].
            # y_pred 形状 [100]。
            y_pred = model(x_input).squeeze(0)
            rel_l2 = relative_l2_error(y_pred, y_true)
            errors.append(float(rel_l2.cpu()))

    errors_array = np.asarray(errors, dtype=np.float64)
    metrics = {
        "mean_relative_l2": float(errors_array.mean()),
        "std_relative_l2": float(errors_array.std()),
        "median_relative_l2": float(np.median(errors_array)),
        "min_relative_l2": float(errors_array.min()),
        "max_relative_l2": float(errors_array.max()),
        "num_test_tasks": float(num_test_tasks),
    }

    os.makedirs(config.metrics_dir, exist_ok=True)
    metrics_path = os.path.join(config.metrics_dir, "evaluation_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print("Evaluation results:")
    print(f"Mean relative L2 error:   {metrics['mean_relative_l2']:.6f}")
    print(f"Std relative L2 error:    {metrics['std_relative_l2']:.6f}")
    print(f"Median relative L2 error: {metrics['median_relative_l2']:.6f}")
    print(f"Min relative L2 error:    {metrics['min_relative_l2']:.6f}")
    print(f"Max relative L2 error:    {metrics['max_relative_l2']:.6f}")
    print(f"Saved metrics to {metrics_path}")
    return metrics
