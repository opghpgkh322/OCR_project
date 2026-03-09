import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


@dataclass
class AppPaths:
    root: Path
    scans_root: Path
    scans_inbox: Path
    scans_processed: Path
    scans_failed: Path
    exports: Path
    state_dir: Path
    scan_registry: Path
    crm_registry: Path


class OCRDesktopApp:
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, root_window: tk.Tk) -> None:
        self.window = root_window
        self.paths = self._resolve_paths()
        self._ensure_structure()

        self.window.title("OCR Forms Manager")
        self.window.geometry("940x700")

        self.epochs_var = tk.StringVar(value="12")
        self.no_correct_var = tk.BooleanVar(value=False)
        self.fine_tune_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_status()

    @staticmethod
    def _resolve_paths() -> AppPaths:
        root = Path(__file__).resolve().parent
        scans_root = root / "scans"
        return AppPaths(
            root=root,
            scans_root=scans_root,
            scans_inbox=scans_root / "inbox",
            scans_processed=scans_root / "processed",
            scans_failed=scans_root / "failed",
            exports=root / "exports",
            state_dir=root / "app_state",
            scan_registry=root / "app_state" / "scan_registry.json",
            crm_registry=root / "app_state" / "crm_sent_registry.json",
        )

    def _ensure_structure(self) -> None:
        for path in [
            self.paths.scans_root,
            self.paths.scans_inbox,
            self.paths.scans_processed,
            self.paths.scans_failed,
            self.paths.exports,
            self.paths.state_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        if not self.paths.scan_registry.exists():
            self.paths.scan_registry.write_text("{}", encoding="utf-8")
        if not self.paths.crm_registry.exists():
            self.paths.crm_registry.write_text("{}", encoding="utf-8")

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        workflow_tab = ttk.Frame(notebook)
        tools_tab = ttk.Frame(notebook)
        training_tab = ttk.Frame(notebook)
        notebook.add(workflow_tab, text="Основной процесс")
        notebook.add(tools_tab, text="Инструменты")
        notebook.add(training_tab, text="Дообучение")

        self._build_workflow_tab(workflow_tab)
        self._build_tools_tab(tools_tab)
        self._build_training_tab(training_tab)

        log_frame = ttk.LabelFrame(self.window, text="Журнал")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def _build_workflow_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Поток кассира")
        controls.pack(fill="x", padx=8, pady=8)

        ttk.Button(controls, text="1) Импортировать новые сканы в scans/inbox", command=self.import_scans).pack(fill="x", padx=6, pady=4)
        ttk.Button(controls, text="2) Обработать новые сканы в CSV", command=self.process_new_scans).pack(fill="x", padx=6, pady=4)
        ttk.Button(controls, text="3) Отправить в CRM только новых клиентов", command=self.export_new_clients_for_crm).pack(fill="x", padx=6, pady=4)

        ttk.Checkbutton(
            controls,
            text="Отключить словарную автокоррекцию (аналог --no-correct)",
            variable=self.no_correct_var,
        ).pack(anchor="w", padx=6, pady=(6, 2))

        status = ttk.LabelFrame(parent, text="Статус папок")
        status.pack(fill="x", padx=8, pady=8)

        self.status_label = ttk.Label(status, text="", justify="left")
        self.status_label.pack(anchor="w", padx=6, pady=6)

        ttk.Button(status, text="Обновить статус", command=self._refresh_status).pack(anchor="w", padx=6, pady=(0, 6))

    def _build_tools_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Управление OCR")
        frame.pack(fill="x", padx=8, pady=8)

        ttk.Button(frame, text="Открыть редактор координат ячеек", command=self.open_configurator).pack(fill="x", padx=6, pady=4)
        ttk.Button(frame, text="Запустить коррекцию распознанных бланков (dataset_review)", command=self.run_review).pack(fill="x", padx=6, pady=4)
        ttk.Button(frame, text="Инвертировать dataset_review", command=self.invert_review_dataset).pack(fill="x", padx=6, pady=4)

        folders = ttk.LabelFrame(parent, text="Быстрый доступ к папкам")
        folders.pack(fill="x", padx=8, pady=8)

        ttk.Button(folders, text="Открыть dataset_external", command=lambda: self.open_folder(self.paths.root / "dataset_external")).pack(fill="x", padx=6, pady=4)
        ttk.Button(folders, text="Открыть dataset_review", command=lambda: self.open_folder(self.paths.root / "dataset_review")).pack(fill="x", padx=6, pady=4)
        ttk.Button(folders, text="Открыть scans", command=lambda: self.open_folder(self.paths.scans_root)).pack(fill="x", padx=6, pady=4)

    def _build_training_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Дообучение модели")
        frame.pack(fill="x", padx=8, pady=8)

        ttk.Label(frame, text="EPOCHS:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(frame, textvariable=self.epochs_var, width=12).grid(row=0, column=1, sticky="w", padx=6, pady=6)

        ttk.Checkbutton(frame, text="Fine-tune (дообучать текущую модель)", variable=self.fine_tune_var).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            padx=6,
            pady=4,
        )

        ttk.Button(frame, text="Запустить обучение", command=self.train_model).grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=8)

    def _log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {message}\n")
        self.log_text.see("end")
        self.window.update_idletasks()

    def _run_command(self, args: list[str], title: str) -> tuple[int, str, str]:
        self._log(f"{title}: {' '.join(args)}")
        process = subprocess.run(args, cwd=self.paths.root, capture_output=True, text=True)
        if process.stdout.strip():
            self._log(process.stdout.strip())
        if process.stderr.strip():
            self._log(process.stderr.strip())
        return process.returncode, process.stdout, process.stderr

    def _refresh_status(self) -> None:
        inbox_count = len([p for p in self.paths.scans_inbox.glob("*") if p.suffix.lower() in self.IMAGE_EXTENSIONS])
        processed_count = len([p for p in self.paths.scans_processed.glob("*") if p.is_file()])
        failed_count = len([p for p in self.paths.scans_failed.glob("*") if p.is_file()])
        self.status_label.configure(
            text=(
                f"Inbox (необработанные): {inbox_count}\n"
                f"Processed (обработанные): {processed_count}\n"
                f"Failed (ошибки): {failed_count}\n"
                f"Экспорт CSV: {self.paths.exports}"
            )
        )

    def _read_json(self, path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _normalize_text(value: str) -> str:
        return "".join(ch for ch in (value or "").strip().upper() if ch.isalnum())

    def _client_fingerprint(self, row: dict) -> str:
        parts = [
            self._normalize_text(row.get("last_name", "")),
            self._normalize_text(row.get("first_name", "")),
            self._normalize_text(row.get("patronymic", "")),
            self._normalize_text(row.get("birth_date", "")),
            self._normalize_text(row.get("phone", "")),
        ]
        return "|".join(parts)

    def import_scans(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Выберите сканы",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")],
        )
        if not selected:
            return

        imported = 0
        for src in selected:
            src_path = Path(src)
            target = self.paths.scans_inbox / src_path.name
            stem = src_path.stem
            suffix = src_path.suffix
            counter = 1
            while target.exists():
                target = self.paths.scans_inbox / f"{stem}_{counter}{suffix}"
                counter += 1
            shutil.copy2(src_path, target)
            imported += 1

        self._log(f"Импортировано файлов: {imported}")
        self._refresh_status()

    def process_new_scans(self) -> None:
        scan_files = [p for p in sorted(self.paths.scans_inbox.glob("*")) if p.suffix.lower() in self.IMAGE_EXTENSIONS]
        if not scan_files:
            messagebox.showinfo("OCR", "В scans/inbox нет новых сканов.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = self.paths.exports / f"ocr_batch_{timestamp}.csv"

        cmd = [
            sys.executable,
            "scripts/04_scans_to_csv.py",
            "--scans",
            str(self.paths.scans_inbox),
            "--output",
            str(output_csv),
        ]
        if self.no_correct_var.get():
            cmd.append("--no-correct")

        code, _, _ = self._run_command(cmd, "OCR")
        registry = self._read_json(self.paths.scan_registry)

        for scan in scan_files:
            digest = self._file_sha256(scan)
            state = {
                "filename": scan.name,
                "sha256": digest,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "output_csv": str(output_csv),
                "status": "processed" if code == 0 else "failed",
            }
            registry[digest] = state

            destination_dir = self.paths.scans_processed if code == 0 else self.paths.scans_failed
            shutil.move(str(scan), str(destination_dir / scan.name))

        self._write_json(self.paths.scan_registry, registry)
        self._refresh_status()

        if code == 0:
            messagebox.showinfo("OCR", f"Обработка завершена. CSV: {output_csv}")
        else:
            messagebox.showerror("OCR", "Ошибка OCR. Файлы перемещены в scans/failed.")

    def export_new_clients_for_crm(self) -> None:
        batches = sorted(self.paths.exports.glob("ocr_batch_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not batches:
            messagebox.showinfo("CRM", "Нет CSV батчей для отправки.")
            return

        source_csv = batches[0]
        sent_registry = self._read_json(self.paths.crm_registry)

        new_rows = []
        with open(source_csv, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                fp = self._client_fingerprint(row)
                if not fp.strip("|"):
                    continue
                if fp in sent_registry:
                    continue
                sent_registry[fp] = {
                    "first_seen_at": datetime.now().isoformat(timespec="seconds"),
                    "source_csv": str(source_csv),
                    "client": row,
                }
                new_rows.append(row)

        export_path = self.paths.exports / f"crm_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(export_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["filename", "last_name", "first_name", "patronymic", "birth_date", "phone"],
            )
            writer.writeheader()
            writer.writerows(new_rows)

        self._write_json(self.paths.crm_registry, sent_registry)
        self._log(f"CRM export: новых клиентов {len(new_rows)} -> {export_path}")
        messagebox.showinfo("CRM", f"Сформирован файл для CRM: {export_path}\nНовых клиентов: {len(new_rows)}")

    def open_configurator(self) -> None:
        code, _, _ = self._run_command([sys.executable, "scripts/02_configurator.py"], "Конфигуратор")
        if code != 0:
            messagebox.showerror("Конфигуратор", "Не удалось запустить редактор координат.")

    def run_review(self) -> None:
        code, _, _ = self._run_command([sys.executable, "scripts/05_review_scans.py"], "Review")
        if code != 0:
            messagebox.showwarning("Review", "Review-скрипт завершился с ошибкой. Проверьте лог.")

    def invert_review_dataset(self) -> None:
        code, _, _ = self._run_command([sys.executable, "invert_dataset_review.py"], "Invert dataset_review")
        if code == 0:
            messagebox.showinfo("Invert", "Инвертирование dataset_review завершено.")
        else:
            messagebox.showwarning("Invert", "Ошибка при инвертировании. Проверьте лог.")

    def train_model(self) -> None:
        epochs = self.epochs_var.get().strip()
        if not epochs.isdigit() or int(epochs) <= 0:
            messagebox.showerror("Train", "EPOCHS должен быть положительным целым числом.")
            return

        cmd = [sys.executable, "scripts/train_model.py", "--epochs", epochs]
        if not self.fine_tune_var.get():
            cmd.append("--no-fine-tune")

        code, _, _ = self._run_command(cmd, "Training")
        if code == 0:
            messagebox.showinfo("Train", "Обучение завершено успешно.")
        else:
            messagebox.showwarning("Train", "Обучение завершилось с ошибкой. Проверьте лог.")

    @staticmethod
    def open_folder(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


def main() -> None:
    root = tk.Tk()
    app = OCRDesktopApp(root)
    app._log("Приложение готово. Рекомендуемая точка для сканера: scans/inbox")
    root.mainloop()


if __name__ == "__main__":
    main()
