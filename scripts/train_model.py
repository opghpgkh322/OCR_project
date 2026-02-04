import argparse
from pathlib import Path

from ocr_app.training.trainer import TrainingConfig, train_model


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset_root = repo_root / "dataset_external"

    parser = argparse.ArgumentParser(description="Train OCR model with style-stratified splits.")
    parser.add_argument("--data-root", default=str(default_dataset_root), help="Root folder with datasets.")
    parser.add_argument("--image-size", type=int, default=32, help="Square size for model inputs.")
    parser.add_argument("--epochs", type=int, default=24, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/validation split ratio.")
    parser.add_argument("--style-clusters", type=int, default=3, help="Style cluster count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    parser.add_argument("--label-smoothing", type=float, default=0.03, help="Label smoothing factor.")
    parser.add_argument("--output-dir", default="model", help="Directory to save model artifacts.")
    args = parser.parse_args()

    config = TrainingConfig(
        data_root=Path(args.data_root),
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        style_clusters=args.style_clusters,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        output_dir=Path(args.output_dir),
    )
    train_model(config)


if __name__ == "__main__":
    main()
