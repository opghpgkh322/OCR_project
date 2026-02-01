import hashlib
import tarfile
import zipfile
from pathlib import Path

import requests

DATASETS = [
    {
        "name": "handwritten-cyrillic",
        "url": "https://github.com/egorsmkv/handwritten-cyrillic/archive/refs/heads/master.zip",
        "type": "zip",
    },
    {
        "name": "russian-handwriting",
        "url": "https://github.com/konstantinlevin/russian-handwriting-dataset/archive/refs/heads/master.zip",
        "type": "zip",
    },
    {
        "name": "printed-cyrillic-fonts",
        "url": "https://github.com/dimaba/fonts-dataset/archive/refs/heads/master.zip",
        "type": "zip",
    },
    {
        "name": "digits-mnist",
        "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        "type": "npz",
    },
]


def download_file(url: str, dest: Path) -> None:
    headers = {"User-Agent": "OCR-project-downloader"}
    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path) as archive:
            archive.extractall(target_dir)


def sha256sum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target_root = root / "external_database"
    target_root.mkdir(parents=True, exist_ok=True)

    report_lines = []
    for dataset in DATASETS:
        name = dataset["name"]
        url = dataset["url"]
        file_type = dataset["type"]
        dest = target_root / f"{name}.{file_type}"
        print(f"Downloading {name}...")
        try:
            download_file(url, dest)
        except Exception as exc:
            print(f"Failed to download {name}: {exc}")
            report_lines.append(f"{name}: download failed ({exc})")
            continue

        checksum = sha256sum(dest)
        report_lines.append(f"{name}: downloaded ({checksum})")

        if file_type in {"zip", "tar.gz"}:
            extract_dir = target_root / name
            extract_dir.mkdir(exist_ok=True)
            extract_archive(dest, extract_dir)
            report_lines.append(f"{name}: extracted to {extract_dir}")

    report_path = target_root / "download_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
