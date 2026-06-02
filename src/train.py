import os
import random
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from .config import ICONConfig
from .data_generation import generate_from_config
from .dataset import ICONDataset
from .model import MLPICON


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str,
    model: MLPICON,
    optimizer: torch.optim.Optimizer,
    config: ICONConfig,
    step: int,
    loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "loss": loss,
        },
        path,
    )


def train(config: ICONConfig) -> str:
    set_seed(config.seed)
    device = select_device()
    print(f"Using device: {device}")
    print("Generating training data...")

    u_paths, y_paths, params, _ = generate_from_config(
        config=config,
        n_regimes=config.n_regimes_train,
        seed=config.seed,
    )

    dataset = ICONDataset(
        u_paths=u_paths,
        y_paths=y_paths,
        params=params,
        context_size=config.context_size,
        n_tasks=config.num_iterations * config.batch_size,
        seed=config.seed + 1,
    )

    model = MLPICON(input_dim=config.input_dim, output_dim=config.output_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1000, 7000],
        gamma=0.3,
    )
    loss_fn = nn.MSELoss()

    model.train()
    last_loss = 0.0
    for step in range(1, config.num_iterations + 1):
        x_batch, y_batch = dataset.sample_batch(config.batch_size)

        # x_batch shape [B, 1100], y_batch shape [B, 100].
        # x_batch 形状 [B, 1100]，y_batch 形状 [B, 100]。
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        last_loss = float(loss.detach().cpu())
        if step % 200 == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:05d}/{config.num_iterations} | "
                f"train_mse={last_loss:.8f} | lr={current_lr:.2e}"
            )

        if step % 1000 == 0:
            checkpoint_path = os.path.join(config.checkpoints_dir, f"mlp_icon_step_{step}.pt")
            save_checkpoint(checkpoint_path, model, optimizer, config, step, last_loss)

    save_checkpoint(config.final_checkpoint_path, model, optimizer, config, config.num_iterations, last_loss)
    print(f"Saved final checkpoint to {config.final_checkpoint_path}")
    return config.final_checkpoint_path
