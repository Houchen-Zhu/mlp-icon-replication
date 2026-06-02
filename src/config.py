from dataclasses import dataclass
import os


@dataclass
class ICONConfig:
    n_regimes_train: int = 1000
    n_regimes_test: int = 200
    pairs_per_regime: int = 10
    n_steps: int = 100
    context_size: int = 5
    batch_size: int = 4
    num_iterations: int = 10000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    output_dir: str = "outputs"
    checkpoint_path: str | None = None
    beta_min: float = 0.462
    beta_max: float = 9.011
    lambda_min: float = 0.1
    lambda_max: float = 0.5
    t_final: float = 1.0

    @property
    def dt(self) -> float:
        return self.t_final / self.n_steps

    @property
    def input_dim(self) -> int:
        return self.context_size * (self.n_steps + self.n_steps) + self.n_steps

    @property
    def output_dim(self) -> int:
        return self.n_steps

    @property
    def final_checkpoint_path(self) -> str:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        return os.path.join(self.output_dir, "checkpoints", "mlp_icon_final.pt")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints")

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.output_dir, "metrics")

    @property
    def figures_dir(self) -> str:
        return os.path.join(self.output_dir, "figures")
