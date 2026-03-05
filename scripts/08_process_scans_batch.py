import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def move_scan(scan_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / scan_path.name
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = destination_dir / f"{scan_path.stem}_{stamp}{scan_path.suffix}"
    shutil.move(str(scan_path), str(target))
    return target


def run_ocr(repo_root: Path, scans_dir: Path, output_csv: Path, no_correct: bool) -> None:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "04_scans_to_csv.py"),
        "--scans",
        str(scans_dir),
        "--output",
        str(output_csv),
    ]
    if no_correct:
        command.append("--no-correct")

    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "OCR pipeline failed")


def append_history(output_csv: Path, history_csv: Path, run_id: str) -> int:
    if not output_csv.exists():
        return 0

    with open(output_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return 0

    history_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "processed_at", "filename", "last_name", "first_name", "patronymic", "birth_date", "phone"]
    is_new = not history_csv.exists()

    with open(history_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        now = datetime.now().isoformat(timespec="seconds")
        for row in rows:
            writer.writerow({"run_id": run_id, "processed_at": now, **row})

    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Process only new scans from scans/inbox and archive them.")
    parser.add_argument("--scans-root", default=str(repo_root / "scans"))
    parser.add_argument("--output", default=str(repo_root / "output.csv"))
    parser.add_argument("--history", default=str(repo_root / "scans" / "recognized_history.csv"))
    parser.add_argument("--no-correct", action="store_true")
    args = parser.parse_args()

    scans_root = Path(args.scans_root)
    inbox = scans_root / "inbox"
    processed = scans_root / "processed"
    failed = scans_root / "failed"

    inbox.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    failed.mkdir(parents=True, exist_ok=True)

    scan_files = sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not scan_files:
        print("Нет новых сканов в scans/inbox.")
        return

    print(f"Найдено {len(scan_files)} новых сканов. Запускаю OCR...")
    output_csv = Path(args.output)

    try:
        run_ocr(repo_root=repo_root, scans_dir=inbox, output_csv=output_csv, no_correct=args.no_correct)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Ошибка OCR: {exc}")
        for scan in scan_files:
            new_path = move_scan(scan, failed)
            print(f"⚠️ Перемещен в failed: {new_path.name}")
        raise SystemExit(1) from exc

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_count = append_history(output_csv, Path(args.history), run_id)

    for scan in scan_files:
        new_path = move_scan(scan, processed)
        print(f"✅ Обработан: {new_path.name}")

    print("-" * 40)
    print(f"Готово. Строк в {output_csv.name}: {rows_count}")
    print(f"История распознавания: {args.history}")


if __name__ == "__main__":
    main()