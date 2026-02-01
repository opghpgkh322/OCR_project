import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import requests

DEFAULT_DATASETS = [
    {
        "name": "google-fonts",
        "urls": ["https://codeload.github.com/google/fonts/zip/refs/heads/main"],
        "type": "zip",
        "enabled": True,
    },
    {
        "name": "emnist",
        "urls": ["https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"],
        "type": "zip",
        "enabled": True,
    },
    {
        "name": "kaggle-handwritten-russian-letters",
        "type": "kaggle",
        "enabled": True,
        "kaggle_dataset": "kapralok/handwritten-russian-letters",
    },
    {
        "name": "digits-mnist",
        "urls": ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"],
        "type": "npz",
        "enabled": True,
    },
]


def get_remote_size_mb(url: str) -> float | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=(10, 30))
        response.raise_for_status()
    except Exception:
        return None
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    return int(content_length) / (1024 * 1024)


def download_file(url: str, dest: Path) -> None:
    headers = {"User-Agent": "OCR-project-downloader"}
    with requests.get(url, stream=True, headers=headers, timeout=(10, 60)) as response:
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


def download_kaggle_dataset(dataset: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target_dir)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Kaggle download failed.")


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
    config_path = target_root / "datasets.json"
    if config_path.exists():
        datasets = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        datasets = DEFAULT_DATASETS
        config_path.write_text(json.dumps(DEFAULT_DATASETS, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = []
    for dataset in datasets:
        if dataset.get("enabled") is False:
            report_lines.append(f"{dataset.get('name', 'unknown')}: skipped (disabled)")
            continue
        name = dataset["name"]
        urls = dataset.get("urls") or [dataset.get("url")]
        file_type = dataset["type"]
        dest = target_root / f"{name}.{file_type}"
        print(f"Downloading {name}...")
        try:
            last_error: Exception | None = None
            if file_type == "kaggle":
                kaggle_dataset = dataset.get("kaggle_dataset")
                if not kaggle_dataset:
                    raise RuntimeError("Missing kaggle_dataset entry.")
                download_kaggle_dataset(kaggle_dataset, target_root / name)
                last_error = None
            else:
                for url in urls:
                    if not url:
                        continue
                    try:
                        size_mb = get_remote_size_mb(url)
                        if size_mb is not None:
                            print(f"Remote size for {name}: {size_mb:.1f} MB")
                        download_file(url, dest)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
            if last_error is not None:
                raise last_error
        except Exception as exc:
            print(f"Failed to download {name}: {exc}")
            report_lines.append(f"{name}: download failed ({exc})")
            continue

        if file_type == "kaggle":
            report_lines.append(f"{name}: downloaded via kaggle")
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
