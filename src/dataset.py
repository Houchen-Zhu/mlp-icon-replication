from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ICONDataset(Dataset):
    """Randomly samples in-context operator learning tasks."""

    def __init__(
        self,
        u_paths: np.ndarray,
        y_paths: np.ndarray,
        params: np.ndarray,
        context_size: int = 5,
        n_tasks: int = 10000,
        seed: int = 0,
        return_metadata: bool = False,
    ) -> None:
        self.u_paths = u_paths
        self.y_paths = y_paths
        self.params = params
        self.context_size = context_size
        self.n_tasks = n_tasks
        self.return_metadata = return_metadata
        self.rng = np.random.default_rng(seed)

        self.n_regimes, self.pairs_per_regime, self.n_steps = u_paths.shape
        if self.pairs_per_regime <= self.context_size:
            raise ValueError("pairs_per_regime must be larger than context_size.")

    def __len__(self) -> int:
        return self.n_tasks

    def __getitem__(self, index: int) -> Any:
        return self.sample_task()

    def sample_task(self, regime_index: int | None = None) -> Any:
        if regime_index is None:
            regime_index = int(self.rng.integers(0, self.n_regimes))

        question_index = int(self.rng.integers(0, self.pairs_per_regime))
        available_indices = [i for i in range(self.pairs_per_regime) if i != question_index]
        context_indices = self.rng.choice(
            available_indices,
            size=self.context_size,
            replace=False,
        )

        pieces = []
        for context_index in context_indices:
            # Each context contributes u^m and Y^m, each shape [N].
            # 每个上下文样本贡献 u^m 和 Y^m，二者形状都是 [N]。
            pieces.append(self.u_paths[regime_index, context_index])
            pieces.append(self.y_paths[regime_index, context_index])

        # Query u^0 has shape [N]; flattened input shape is [M * 2N + N] = [1100].
        # 查询 u^0 的形状是 [N]；展平输入形状为 [M * 2N + N] = [1100]。
        pieces.append(self.u_paths[regime_index, question_index])
        x_input = np.concatenate(pieces, axis=0).astype(np.float32)

        # Target Y^0 has shape [N] = [100].
        # 目标 Y^0 的形状是 [N] = [100]。
        y_target = self.y_paths[regime_index, question_index].astype(np.float32)

        x_tensor = torch.from_numpy(x_input)
        y_tensor = torch.from_numpy(y_target)

        if not self.return_metadata:
            return x_tensor, y_tensor

        lambda_value, beta_value = self.params[regime_index]
        metadata = {
            "lambda_value": float(lambda_value),
            "beta_value": float(beta_value),
            "regime_index": int(regime_index),
            "question_index": int(question_index),
            "context_indices": [int(i) for i in context_indices],
        }
        return x_tensor, y_tensor, metadata

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            x_input, y_target = self.sample_task()
            x_batch.append(x_input)
            y_batch.append(y_target)

        # Batch input shape [B, 1100], target shape [B, 100].
        # 批量输入形状 [B, 1100]，目标形状 [B, 100]。
        return torch.stack(x_batch, dim=0), torch.stack(y_batch, dim=0)
