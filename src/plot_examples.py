import os

os.makedirs("outputs/.matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import ICONConfig
from .data_generation import generate_from_config
from .dataset import ICONDataset
from .evaluate import load_model, relative_l2_error
from .train import select_device, set_seed


def _predict_one(model, x_input: torch.Tensor, y_true: torch.Tensor, device: torch.device):
    with torch.no_grad():
        # x_input shape [1100] -> model input shape [1, 1100].
        # x_input 形状 [1100] -> 模型输入形状 [1, 1100]。
        y_pred = model(x_input.unsqueeze(0).to(device)).squeeze(0).cpu()
    rel_l2 = float(relative_l2_error(y_pred, y_true).cpu())
    return y_pred, rel_l2


def plot_examples(config: ICONConfig) -> None:
    set_seed(config.seed)
    device = select_device()
    print(f"Using device: {device}")
    print("Generating unseen plotting data...")

    u_paths, y_paths, params, times = generate_from_config(
        config=config,
        n_regimes=config.n_regimes_test,
        seed=config.seed + 30_000,
    )
    dataset = ICONDataset(
        u_paths=u_paths,
        y_paths=y_paths,
        params=params,
        context_size=config.context_size,
        n_tasks=10,
        seed=config.seed + 40_000,
        return_metadata=True,
    )

    model = load_model(config, device)
    os.makedirs(config.figures_dir, exist_ok=True)

    first_example = None
    for plot_index in range(5):
        x_input, y_true, metadata = dataset.sample_task()
        y_pred, rel_l2 = _predict_one(model, x_input, y_true, device)

        if first_example is None:
            first_example = (x_input, y_true, y_pred, rel_l2, metadata)

        plt.figure(figsize=(7, 4))
        plt.plot(times, y_true.numpy(), label="True Y", linewidth=2)
        plt.plot(times, y_pred.numpy(), label="Predicted Y", linewidth=2, linestyle="--")
        plt.title(
            f"lambda={metadata['lambda_value']:.3f}, "
            f"beta={metadata['beta_value']:.3f}, rel L2={rel_l2:.4f}"
        )
        plt.xlabel("time t")
        plt.ylabel("price impact")
        plt.legend()
        plt.tight_layout()
        figure_path = os.path.join(config.figures_dir, f"true_vs_pred_{plot_index}.png")
        plt.savefig(figure_path, dpi=160)
        plt.close()
        print(f"Saved {figure_path}")

    plot_context_question_prediction(
        times=times,
        u_paths=u_paths,
        y_paths=y_paths,
        example=first_example,
        figures_dir=config.figures_dir,
    )


def plot_context_question_prediction(
    times,
    u_paths,
    y_paths,
    example,
    figures_dir,
) -> None:
    _, y_true, y_pred, rel_l2, metadata = example
    regime_index = metadata["regime_index"]
    question_index = metadata["question_index"]
    context_indices = metadata["context_indices"]

    # context_u shape [M, N], context_y shape [M, N], question_u shape [N].
    # context_u 形状 [M, N]，context_y 形状 [M, N]，question_u 形状 [N]。
    context_u = u_paths[regime_index, context_indices]
    context_y = y_paths[regime_index, context_indices]
    question_u = u_paths[regime_index, question_index]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for i, u_path in enumerate(context_u):
        axes[0].plot(times, u_path, alpha=0.75, label=f"context u {i + 1}")
    axes[0].plot(times, question_u, color="black", linewidth=2.2, label="question u")
    axes[0].set_ylabel("selling rate")
    axes[0].legend(ncol=2, fontsize=8)

    for i, y_path in enumerate(context_y):
        axes[1].plot(times, y_path, alpha=0.85, label=f"context Y {i + 1}")
    axes[1].set_ylabel("price impact")
    axes[1].legend(ncol=2, fontsize=8)

    axes[2].plot(times, y_true.numpy(), label="True Y", linewidth=2)
    axes[2].plot(times, y_pred.numpy(), label="Predicted Y", linewidth=2, linestyle="--")
    axes[2].set_xlabel("time t")
    axes[2].set_ylabel("price impact")
    axes[2].legend()

    fig.suptitle(
        f"Context/question diagnostic | lambda={metadata['lambda_value']:.3f}, "
        f"beta={metadata['beta_value']:.3f}, rel L2={rel_l2:.4f}"
    )
    fig.tight_layout()
    figure_path = os.path.join(figures_dir, "context_question_prediction.png")
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"Saved {figure_path}")
