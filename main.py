import argparse

from src.config import ICONConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified MLP-ICON replication.")
    parser.add_argument("--mode", choices=["train", "evaluate", "plot"], required=True)
    parser.add_argument("--n_regimes_train", type=int, default=1000)
    parser.add_argument("--n_regimes_test", type=int, default=200)
    parser.add_argument("--pairs_per_regime", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--context_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ICONConfig:
    return ICONConfig(
        n_regimes_train=args.n_regimes_train,
        n_regimes_test=args.n_regimes_test,
        pairs_per_regime=args.pairs_per_regime,
        n_steps=args.n_steps,
        context_size=args.context_size,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if config.batch_size != 4:
        raise ValueError("This replication run must use batch_size = 4.")

    # Input shape is [B, 1100] and output shape is [B, 100] for the README defaults.
    # README 默认设置下，输入形状是 [B, 1100]，输出形状是 [B, 100]。
    if args.mode == "train":
        from src.train import train

        train(config)
    elif args.mode == "evaluate":
        from src.evaluate import evaluate

        evaluate(config)
    elif args.mode == "plot":
        from src.plot_examples import plot_examples

        plot_examples(config)


if __name__ == "__main__":
    main()
