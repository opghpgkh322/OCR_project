#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import zipfile
import unicodedata
import re
from pathlib import Path
from collections import Counter
from typing import Optional, Dict

# --- Маппинг русских букв -> имена папок как на скриншоте ---
CYR_TO_LAT: Dict[str, str] = {
    "А": "A_cyr",
    "Б": "B_cyr",
    "В": "V_cyr",
    "Г": "G_cyr",
    "Д": "D_cyr",
    "Е": "E_cyr",
    "Ё": "Yo_cyr",
    "Ж": "Zh_cyr",
    "З": "Z_cyr",
    "И": "I_cyr",
    "Й": "Y_cyr",
    "К": "K_cyr",
    "Л": "L_cyr",
    "М": "M_cyr",
    "Н": "N_cyr",
    "О": "O_cyr",
    "П": "P_cyr",
    "Р": "R_cyr",
    "С": "S_cyr",
    "Т": "T_cyr",
    "У": "U_cyr",
    "Ф": "F_cyr",
    "Х": "Kh_cyr",
    "Ц": "Ts_cyr",
    "Ч": "Ch_cyr",
    "Ш": "Sh_cyr",
    "Щ": "Shch_cyr",
    "Ъ": "Hard_cyr",
    "Ы": "Yery_cyr",
    "Ь": "Soft_cyr",
    "Э": "E_rev_cyr",
    "Ю": "Yu_cyr",
    "Я": "Ya_cyr",
}
CYR_TO_LAT.update({k.lower(): v for k, v in CYR_TO_LAT.items()})

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CYR_RE = re.compile(r"[А-Яа-яЁё]")


def fix_zip_name(name: str) -> str:
    """
    Пытаемся восстановить кириллицу в именах внутри ZIP.
    Если ZIP не помечен как UTF-8, Python часто декодирует как cp437,
    а реально там cp866/cp1251. Мы пробуем несколько вариантов и выбираем тот,
    где появляется кириллица.
    """
    # Единые разделители
    name = name.replace("\\", "/")

    # Если кириллица уже есть — оставляем
    if CYR_RE.search(name):
        return unicodedata.normalize("NFC", name)

    # Попытка "пере-декодировать" из cp437 в типичные windows-кодировки
    try:
        raw = name.encode("cp437", errors="replace")
    except Exception:
        return unicodedata.normalize("NFC", name)

    candidates = []
    for enc in ("utf-8", "cp866", "cp1251", "koi8-r"):
        try:
            s = raw.decode(enc)
            s = s.replace("\\", "/")
            s = unicodedata.normalize("NFC", s)
            candidates.append(s)
        except Exception:
            pass

    # Выбираем кандидата, где появилась кириллица
    for s in candidates:
        if CYR_RE.search(s):
            return s

    # Не нашли — вернём исходное
    return unicodedata.normalize("NFC", name)


def is_digit_folder(name: str) -> bool:
    s = name.strip()
    return len(s) == 1 and s.isdigit()


def find_class_folder(parts) -> Optional[str]:
    """
    Ищем компоненту пути (директорию), которая является:
      - кириллической буквой (А/а/.../Ё/ё/Ъ/ъ/Ь/ь/Ы/ы/Э/э и т.д.)
      - цифрой 0..9
    """
    for p in parts:
        p2 = p.strip()
        if p2 in CYR_TO_LAT:
            return p2
        if is_digit_folder(p2):
            return p2
    return None


def unique_path(dst_dir: Path, filename: str) -> Path:
    candidate = dst_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem}_{i:03d}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def normalize_target_folder(class_name: str, digits_suffix: Optional[str]) -> str:
    class_name = class_name.strip()
    if class_name in CYR_TO_LAT:
        return CYR_TO_LAT[class_name]
    if is_digit_folder(class_name):
        return f"{class_name}{digits_suffix}" if digits_suffix else class_name
    raise ValueError(f"Unknown class folder: {class_name!r}")


def extract_all_zips(src_dir: Path, dst_dir: Path, digits_suffix: Optional[str], verbose: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(src_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"В папке {src_dir} не найдено ни одного .zip")

    extracted_per_class = Counter()
    skipped = 0

    for zip_path in zips:
        if verbose:
            print(f"\n[ZIP] {zip_path.name}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                fixed_name = fix_zip_name(info.filename)
                parts = [p for p in fixed_name.split("/") if p and p not in ("__MACOSX",) and not p.startswith(".")]

                if not parts:
                    skipped += 1
                    continue

                ext = Path(parts[-1]).suffix.lower()
                if ext not in IMAGE_EXTS:
                    skipped += 1
                    continue

                class_folder = find_class_folder(parts[:-1])
                if class_folder is None:
                    # Если хочешь, можно в verbose показать пару примеров пропусков
                    if verbose and CYR_RE.search(fixed_name) is False:
                        pass
                    skipped += 1
                    continue

                out_folder_name = normalize_target_folder(class_folder, digits_suffix)
                out_folder = dst_dir / out_folder_name
                out_folder.mkdir(parents=True, exist_ok=True)

                original_name = Path(parts[-1]).name
                prefixed_name = f"{zip_path.stem}__{original_name}"
                out_path = unique_path(out_folder, prefixed_name)

                with zf.open(info, "r") as src_f, open(out_path, "wb") as dst_f:
                    dst_f.write(src_f.read())

                extracted_per_class[out_folder_name] += 1

    total = sum(extracted_per_class.values())
    print("\nГотово.")
    print(f"Извлечено изображений: {total}")
    print(f"Пропущено элементов (не картинки/не нашли класс/мусор): {skipped}")
    print(f"Результат: {dst_dir.resolve()}")

    # Сводка по папкам
    print("\nСводка по классам (папкам):")
    for k in sorted(extracted_per_class.keys()):
        print(f"  {k:10s}  {extracted_per_class[k]}")


def main():
    parser = argparse.ArgumentParser(
        description="Распаковать все zip из папки X в папку Y, переименовав папки-классы (кириллица -> латиница + _cyr)."
    )
    parser.add_argument("--src", required=True, type=Path, help="Папка X с .zip архивами")
    parser.add_argument("--dst", required=True, type=Path, help="Папка Y для результата")
    parser.add_argument(
        "--digits-suffix",
        default="",
        help='Суффикс для папок цифр. Например "_dig" даст 0_dig..9_dig. По умолчанию цифры будут 0..9.',
    )
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    if not args.src.exists() or not args.src.is_dir():
        raise NotADirectoryError(f"--src должен быть существующей папкой: {args.src}")

    digits_suffix = args.digits_suffix if args.digits_suffix else None
    extract_all_zips(args.src, args.dst, digits_suffix, verbose=args.verbose)


if __name__ == "__main__":
    main()
