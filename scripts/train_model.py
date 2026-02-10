import argparse
from pathlib import Path
from ocr_app.training.trainer import TrainingConfig, train_model


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset_root = repo_root / "dataset_external"

    # Папка для дообучения (сюда будут падать ваши исправления)
    review_root = repo_root / "dataset_review"

    parser = argparse.ArgumentParser(description="Train Fast OCR Model on CPU.")
    parser.add_argument("--data-root", default=str(default_dataset_root))
    parser.add_argument("--review-root", default=str(review_root))

    # Оптимальные параметры для i7-7700
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)  # Должно хватить 20-25 эпох
    parser.add_argument("--batch-size", type=int, default=32)  # Маленький батч быстрее на CPU
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--output-dir", default="model")

    # Флаг для дообучения
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        default=True,  # <--- Добавили эту строку
        help="Дообучение существующей модели (по умолчанию True). Используйте --no-fine-tune для обучения с нуля"
    )

    # И добавьте новый аргумент для отключения:
    parser.add_argument(
        "--no-fine-tune",
        dest="fine_tune",
        action="store_false",
        help="Обучить модель с нуля (сбросить веса)"
    )

    args = parser.parse_args()

    config = TrainingConfig(
        data_root=Path(args.data_root),
        review_root=Path(args.review_root),
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        output_dir=Path(args.output_dir),
        is_fine_tuning=args.fine_tune  # Передаем флаг в конфиг
    )

    print(f"Запуск обучения на CPU (i7-7700)...")
    print(f"Режим: {'ДО-ОБУЧЕНИЕ (Fine-Tuning)' if args.fine_tune else 'ОБУЧЕНИЕ С НУЛЯ'}")

    train_model(config)


if __name__ == "__main__":
    main()
