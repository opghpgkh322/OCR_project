import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path


FIELDS = ["last_name", "first_name", "patronymic", "birth_date", "phone"]


def normalize(value: str) -> str:
    return (value or "").strip().upper().replace(" ", "")


def build_client_fingerprint(row: dict[str, str]) -> str:
    last_name = normalize(row.get("last_name", ""))
    first_name = normalize(row.get("first_name", ""))
    patronymic = normalize(row.get("patronymic", ""))
    birth_date = normalize(row.get("birth_date", ""))
    phone = "".join(ch for ch in row.get("phone", "") if ch.isdigit())

    if phone:
        key = f"PHONE:{phone}|BIRTH:{birth_date}|NAME:{last_name}|FIRST:{first_name}|PATR:{patronymic}"
    else:
        key = f"NO_PHONE|BIRTH:{birth_date}|NAME:{last_name}|FIRST:{first_name}|PATR:{patronymic}"

    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_registry(path: Path) -> set[str]:
    if not path.exists():
        return set()

    rows = read_csv(path)
    return {row.get("client_fingerprint", "") for row in rows if row.get("client_fingerprint")}


def append_registry(path: Path, new_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sent_at", "source_filename", *FIELDS, "client_fingerprint"]
    is_new = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        now = datetime.now().isoformat(timespec="seconds")
        for row in new_rows:
            record = {
                "sent_at": now,
                "source_filename": row.get("filename", ""),
                "last_name": row.get("last_name", ""),
                "first_name": row.get("first_name", ""),
                "patronymic": row.get("patronymic", ""),
                "birth_date": row.get("birth_date", ""),
                "phone": row.get("phone", ""),
                "client_fingerprint": row["client_fingerprint"],
            }
            writer.writerow(record)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export only new OCR clients for CRM sync.")
    parser.add_argument("--input", default=str(repo_root / "output.csv"))
    parser.add_argument("--registry", default=str(repo_root / "crm" / "crm_sent_registry.csv"))
    parser.add_argument("--export-dir", default=str(repo_root / "crm" / "exports"))
    args = parser.parse_args()

    input_path = Path(args.input)
    registry_path = Path(args.registry)
    export_dir = Path(args.export_dir)

    rows = read_csv(input_path)
    if not rows:
        print("Нет данных для экспорта в CRM. Сначала выполните OCR в output.csv.")
        return

    known_fingerprints = ensure_registry(registry_path)
    new_rows = []

    for row in rows:
        fingerprint = build_client_fingerprint(row)
        if fingerprint in known_fingerprints:
            continue
        normalized_row = dict(row)
        normalized_row["client_fingerprint"] = fingerprint
        new_rows.append(normalized_row)
        known_fingerprints.add(fingerprint)

    if not new_rows:
        print("Новых клиентов для CRM не найдено (дубликаты отфильтрованы).")
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = export_dir / f"crm_new_clients_{stamp}.csv"

    with open(export_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", *FIELDS, "client_fingerprint"])
        writer.writeheader()
        for row in new_rows:
            writer.writerow({
                "filename": row.get("filename", ""),
                "last_name": row.get("last_name", ""),
                "first_name": row.get("first_name", ""),
                "patronymic": row.get("patronymic", ""),
                "birth_date": row.get("birth_date", ""),
                "phone": row.get("phone", ""),
                "client_fingerprint": row["client_fingerprint"],
            })

    append_registry(registry_path, new_rows)

    print(f"✅ Новых клиентов: {len(new_rows)}")
    print(f"📤 Файл для CRM: {export_path}")
    print(f"🧾 Реестр отправок: {registry_path}")


if __name__ == "__main__":
    main()