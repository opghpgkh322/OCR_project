import argparse
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class OCRDesktopApp:
    def __init__(self, root: tk.Tk, repo_root: Path) -> None:
        self.root = root
        self.repo_root = repo_root
        self.python = sys.executable

        self.root.title("OCR оператор — рукописные бланки")
        self.root.geometry("980x700")

        self.epochs_var = tk.StringVar(value="12")
        self.fine_tune_var = tk.BooleanVar(value=True)
        self.correction_var = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scans_tab = ttk.Frame(notebook)
        quality_tab = ttk.Frame(notebook)
        train_tab = ttk.Frame(notebook)
        crm_tab = ttk.Frame(notebook)

        notebook.add(scans_tab, text="1. Сканирование и OCR")
        notebook.add(quality_tab, text="2. Коррекция и датасет")
        notebook.add(train_tab, text="3. Дообучение")
        notebook.add(crm_tab, text="4. CRM выгрузка")

        self._build_scans_tab(scans_tab)
        self._build_quality_tab(quality_tab)
        self._build_train_tab(train_tab)
        self._build_crm_tab(crm_tab)

        log_frame = ttk.LabelFrame(self.root, text="Лог выполнения")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _build_scans_tab(self, parent: ttk.Frame) -> None:
        scans_root = self.repo_root / "scans"
        inbox = scans_root / "inbox"
        processed = scans_root / "processed"
        failed = scans_root / "failed"

        for folder in (inbox, processed, failed):
            folder.mkdir(parents=True, exist_ok=True)

        ttk.Label(
            parent,
            text=(
                "Поток работы:\n"
                "1) Сканер сохраняет новые изображения в scans/inbox\n"
                "2) Нажмите 'Запустить OCR новых сканов'\n"
                "3) Обработанные файлы автоматически уйдут в scans/processed, ошибки — в scans/failed"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=10)

        ttk.Checkbutton(
            parent,
            text="Включить словарную автокоррекцию ФИО",
            variable=self.correction_var,
        ).pack(anchor=tk.W, padx=10, pady=(0, 10))

        button_row = ttk.Frame(parent)
        button_row.pack(anchor=tk.W, padx=10, pady=5)

        ttk.Button(button_row, text="Открыть scans/inbox", command=lambda: open_folder(inbox)).grid(row=0, column=0, padx=4)
        ttk.Button(button_row, text="Открыть scans/processed", command=lambda: open_folder(processed)).grid(row=0, column=1, padx=4)
        ttk.Button(button_row, text="Открыть scans/failed", command=lambda: open_folder(failed)).grid(row=0, column=2, padx=4)

        action_row = ttk.Frame(parent)
        action_row.pack(anchor=tk.W, padx=10, pady=8)

        ttk.Button(action_row, text="Запустить OCR новых сканов", command=self.run_ocr_batch).grid(row=0, column=0, padx=4)
        ttk.Button(action_row, text="Открыть output.csv", command=self.open_output_file).grid(row=0, column=1, padx=4)

    def _build_quality_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Используйте инструменты ниже для поддержки качества распознавания:\n"
                "• Корректировка координат ячеек\n"
                "• Ручная валидация полей и пополнение dataset_review\n"
                "• Нормализация (invert) изображений в dataset_review"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=10)

        row1 = ttk.Frame(parent)
        row1.pack(anchor=tk.W, padx=10, pady=5)
        ttk.Button(row1, text="1) Коррекция координат ячеек", command=self.run_configurator).grid(row=0, column=0, padx=4)
        ttk.Button(row1, text="2) Коррекция распознанных бланков", command=self.run_review).grid(row=0, column=1, padx=4)
        ttk.Button(row1, text="3) Запустить invert_dataset_review", command=self.run_invert_review).grid(row=0, column=2, padx=4)

        row2 = ttk.Frame(parent)
        row2.pack(anchor=tk.W, padx=10, pady=5)
        ttk.Button(row2, text="Открыть dataset_external", command=lambda: open_folder(self.repo_root / "dataset_external")).grid(row=0, column=0, padx=4)
        ttk.Button(row2, text="Открыть dataset_review", command=lambda: open_folder(self.repo_root / "dataset_review")).grid(row=0, column=1, padx=4)

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        params = ttk.Frame(parent)
        params.pack(anchor=tk.W, padx=10, pady=15)

        ttk.Label(params, text="EPOCHS:").grid(row=0, column=0, padx=4, sticky=tk.W)
        ttk.Entry(params, textvariable=self.epochs_var, width=8).grid(row=0, column=1, padx=4, sticky=tk.W)

        ttk.Checkbutton(params, text="Дообучение существующей модели (fine-tune)", variable=self.fine_tune_var).grid(
            row=1, column=0, columnspan=2, padx=4, pady=8, sticky=tk.W
        )

        ttk.Button(parent, text="Запустить обучение", command=self.run_training).pack(anchor=tk.W, padx=10, pady=5)

    def _build_crm_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Выгрузка в CRM отправляет только новых клиентов:\n"
                "• сравнение по отпечатку (ФИО + дата рождения + телефон)\n"
                "• уже отправленные записи хранятся в crm/crm_sent_registry.csv"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=10)

        actions = ttk.Frame(parent)
        actions.pack(anchor=tk.W, padx=10, pady=6)

        ttk.Button(actions, text="Сформировать CSV только новых клиентов", command=self.run_crm_export).grid(row=0, column=0, padx=4)
        ttk.Button(actions, text="Открыть crm/exports", command=lambda: open_folder(self.repo_root / "crm" / "exports")).grid(row=0, column=1, padx=4)
        ttk.Button(actions, text="Открыть реестр CRM", command=self.open_crm_registry).grid(row=0, column=2, padx=4)

    def log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def run_command(self, command: list[str], title: str) -> None:
        def _worker() -> None:
            self.log(f"\n=== {title} ===")
            self.log("$ " + " ".join(command))
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.root.after(0, self.log, line.rstrip())
            code = process.wait()
            if code == 0:
                self.root.after(0, self.log, f"✅ {title}: завершено успешно")
            else:
                self.root.after(0, self.log, f"❌ {title}: ошибка (код {code})")

        threading.Thread(target=_worker, daemon=True).start()

    def run_ocr_batch(self) -> None:
        command = [self.python, str(self.repo_root / "scripts" / "08_process_scans_batch.py")]
        if not self.correction_var.get():
            command.append("--no-correct")
        self.run_command(command, "OCR новых сканов")

    def run_configurator(self) -> None:
        command = [self.python, str(self.repo_root / "scripts" / "02_configurator.py")]
        self.run_command(command, "Редактор координат")

    def run_review(self) -> None:
        command = [self.python, str(self.repo_root / "scripts" / "interactive_trainer.py")]
        self.run_command(command, "Коррекция распознавания")

    def run_invert_review(self) -> None:
        command = [self.python, str(self.repo_root / "invert_dataset_review.py")]
        self.run_command(command, "Invert dataset_review")

    def run_training(self) -> None:
        try:
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Неверный параметр", "EPOCHS должен быть положительным целым числом")
            return

        command = [
            self.python,
            str(self.repo_root / "scripts" / "train_model.py"),
            "--epochs",
            str(epochs),
        ]
        if not self.fine_tune_var.get():
            command.append("--no-fine-tune")

        self.run_command(command, "Дообучение модели")

    def run_crm_export(self) -> None:
        command = [self.python, str(self.repo_root / "scripts" / "09_export_new_clients.py")]
        self.run_command(command, "Экспорт новых клиентов в CRM")

    def open_output_file(self) -> None:
        output = self.repo_root / "output.csv"
        if output.exists():
            if sys.platform.startswith("win"):
                os.startfile(output)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output)])
            else:
                subprocess.Popen(["xdg-open", str(output)])
        else:
            messagebox.showinfo("Файл отсутствует", "output.csv пока не создан")

    def open_crm_registry(self) -> None:
        registry = self.repo_root / "crm" / "crm_sent_registry.csv"
        if registry.exists():
            if sys.platform.startswith("win"):
                os.startfile(registry)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(registry)])
            else:
                subprocess.Popen(["xdg-open", str(registry)])
        else:
            messagebox.showinfo("Реестр отсутствует", "crm_sent_registry.csv пока не создан")


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop shell for OCR operations.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = tk.Tk()
    app = OCRDesktopApp(root=root, repo_root=Path(args.repo_root))
    app.log("Приложение запущено. Начните со вкладки 'Сканирование и OCR'.")
    root.mainloop()


if __name__ == "__main__":
    main()
