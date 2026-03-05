import argparse
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

BG_MAIN = "#F9FFF6"
BG_PANEL = "#FFFFFF"
COLOR_ACCENT = "#7ED957"
COLOR_ACCENT_DARK = "#5DA83F"
COLOR_TEXT = "#1A1A1A"
COLOR_MUTED = "#5D6D5A"


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

        self.root.title("SEQUOIA PARK • OCR рабочее место")
        self.root.geometry("1220x760")
        self.root.minsize(1120, 700)
        self.root.configure(bg=BG_MAIN)

        self.epochs_var = tk.StringVar(value="12")
        self.fine_tune_var = tk.BooleanVar(value=True)
        self.correction_var = tk.BooleanVar(value=True)

        self._configure_styles()
        self._build_ui()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Root.TFrame", background=BG_MAIN)
        style.configure("Card.TFrame", background=BG_PANEL)
        style.configure("SectionTitle.TLabel", background=BG_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Body.TLabel", background=BG_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background=BG_PANEL, foreground=COLOR_MUTED, font=("Segoe UI", 9))
        style.configure("HeaderTitle.TLabel", background=BG_MAIN, foreground=COLOR_TEXT, font=("Segoe UI", 24, "bold"))
        style.configure("HeaderSub.TLabel", background=BG_MAIN, foreground=COLOR_MUTED, font=("Segoe UI", 11))

        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 14), foreground="#10250E", background=COLOR_ACCENT)
        style.map("Primary.TButton", background=[("active", COLOR_ACCENT_DARK), ("pressed", COLOR_ACCENT_DARK)])

        style.configure("Action.TButton", font=("Segoe UI", 10), padding=(10, 8), background="#E8F8DF")
        style.map("Action.TButton", background=[("active", "#D7F2CA")])

        style.configure("Notebook.TNotebook", background=BG_PANEL, borderwidth=0)
        style.configure("Notebook.TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(16, 10))

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame")
        root_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        root_frame.grid_columnconfigure(0, weight=0)
        root_frame.grid_columnconfigure(1, weight=1)
        root_frame.grid_rowconfigure(2, weight=1)

        self._build_header(root_frame)
        self._build_left_panel(root_frame)
        self._build_main_panel(root_frame)

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Root.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(1, weight=1)

        logo_canvas = tk.Canvas(header, width=68, height=68, bg=BG_MAIN, highlightthickness=0)
        logo_canvas.grid(row=0, column=0, rowspan=2, sticky="w", padx=(4, 12))
        logo_canvas.create_line(22, 16, 22, 56, fill=COLOR_TEXT, width=4)
        logo_canvas.create_line(42, 24, 42, 56, fill=COLOR_TEXT, width=3)
        logo_canvas.create_polygon(22, 12, 8, 30, 36, 30, fill=COLOR_TEXT, outline=COLOR_TEXT)
        logo_canvas.create_polygon(22, 22, 10, 38, 34, 38, fill=COLOR_TEXT, outline=COLOR_TEXT)
        logo_canvas.create_polygon(42, 20, 34, 32, 50, 32, fill=COLOR_TEXT, outline=COLOR_TEXT)

        ttk.Label(header, text="SEQUOIA PARK", style="HeaderTitle.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(
            header,
            text="OCR для рукописных бланков • Бело-салатовая рабочая панель кассира",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=1, sticky="w")

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, style="Card.TFrame", padding=14)
        left.grid(row=1, column=0, rowspan=2, sticky="nsw", padx=(0, 12))

        ttk.Label(left, text="Быстрый старт", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(
            left,
            text="Чаще всего используется: запуск OCR новых сканов. Эта кнопка выделена и всегда под рукой.",
            style="Hint.TLabel",
            wraplength=290,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="▶ Запустить OCR новых сканов", style="Primary.TButton", command=self.run_ocr_batch).pack(fill=tk.X)

        ttk.Checkbutton(
            left,
            text="Включить словарную автокоррекцию ФИО",
            variable=self.correction_var,
        ).pack(anchor="w", pady=(10, 12))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Папки сканов", style="SectionTitle.TLabel").pack(anchor="w", pady=(6, 6))
        scans_root = self.repo_root / "scans"
        inbox = scans_root / "inbox"
        processed = scans_root / "processed"
        failed = scans_root / "failed"
        for folder in (inbox, processed, failed):
            folder.mkdir(parents=True, exist_ok=True)

        ttk.Button(left, text="Открыть scans/inbox", style="Action.TButton", command=lambda: open_folder(inbox)).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Открыть scans/processed", style="Action.TButton", command=lambda: open_folder(processed)).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Открыть scans/failed", style="Action.TButton", command=lambda: open_folder(failed)).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Открыть output.csv", style="Action.TButton", command=self.open_output_file).pack(fill=tk.X, pady=(8, 2))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Порядок работы", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(
            left,
            text="1) Сканер кладет файлы в scans/inbox\n"
                 "2) Нажмите кнопку OCR\n"
                 "3) Отправьте только новых клиентов в CRM",
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")

    def _build_main_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Root.TFrame")
        panel.grid(row=1, column=1, rowspan=2, sticky="nsew")
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_rowconfigure(2, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        notebook_card = ttk.Frame(panel, style="Card.TFrame", padding=10)
        notebook_card.grid(row=0, column=0, sticky="ew")

        notebook = ttk.Notebook(notebook_card, style="Notebook.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True)

        quality_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        train_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        crm_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)

        notebook.add(quality_tab, text="Качество распознавания")
        notebook.add(train_tab, text="Дообучение модели")
        notebook.add(crm_tab, text="CRM и выгрузка")

        self._build_quality_tab(quality_tab)
        self._build_train_tab(train_tab)
        self._build_crm_tab(crm_tab)

        log_card = ttk.Frame(panel, style="Card.TFrame", padding=10)
        log_card.grid(row=1, column=0, rowspan=2, sticky="nsew", pady=(10, 0))
        log_card.grid_rowconfigure(1, weight=1)
        log_card.grid_columnconfigure(0, weight=1)

        ttk.Label(log_card, text="Окно сообщений и логи", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.log_text = tk.Text(
            log_card,
            wrap=tk.WORD,
            height=15,
            bg="#F6FFF1",
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            font=("Consolas", 10),
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")

    def _build_quality_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Инструменты контроля качества OCR", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Label(
            parent,
            text="Группа редких действий: выполняются по необходимости администратором/старшим кассиром.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        row1 = ttk.Frame(parent, style="Card.TFrame")
        row1.pack(anchor="w", fill=tk.X)
        ttk.Button(row1, text="Коррекция координат ячеек", style="Action.TButton", command=self.run_configurator).grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
        ttk.Button(row1, text="Коррекция распознанных бланков", style="Action.TButton", command=self.run_review).grid(row=0, column=1, padx=6, pady=3, sticky="w")
        ttk.Button(row1, text="Запустить invert_dataset_review", style="Action.TButton", command=self.run_invert_review).grid(row=0, column=2, padx=6, pady=3, sticky="w")

        row2 = ttk.Frame(parent, style="Card.TFrame")
        row2.pack(anchor="w", fill=tk.X, pady=(8, 0))
        ttk.Button(row2, text="Открыть dataset_external", style="Action.TButton", command=lambda: open_folder(self.repo_root / "dataset_external")).grid(row=0, column=0, padx=(0, 6), pady=3)
        ttk.Button(row2, text="Открыть dataset_review", style="Action.TButton", command=lambda: open_folder(self.repo_root / "dataset_review")).grid(row=0, column=1, padx=6, pady=3)

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Дообучение OCR модели", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))

        params = ttk.Frame(parent, style="Card.TFrame")
        params.pack(anchor="w", pady=(0, 8))
        ttk.Label(params, text="EPOCHS:", style="Body.TLabel").grid(row=0, column=0, padx=(0, 6), sticky="w")
        ttk.Entry(params, textvariable=self.epochs_var, width=8).grid(row=0, column=1, padx=(0, 10), sticky="w")

        ttk.Checkbutton(params, text="Дообучение существующей модели (fine-tune)", variable=self.fine_tune_var).grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="w")

        ttk.Button(parent, text="Запустить обучение", style="Action.TButton", command=self.run_training).pack(anchor="w")

    def _build_crm_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Выгрузка в CRM (без дублей)", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(
            parent,
            text="Отправляйте только новых клиентов: фильтр по отпечатку ФИО + дата рождения + телефон.",
            style="Hint.TLabel",
            wraplength=720,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.pack(anchor="w", fill=tk.X)

        ttk.Button(actions, text="Сформировать CSV только новых клиентов", style="Action.TButton", command=self.run_crm_export).grid(row=0, column=0, padx=(0, 6), pady=3)
        ttk.Button(actions, text="Открыть crm/exports", style="Action.TButton", command=lambda: open_folder(self.repo_root / "crm" / "exports")).grid(row=0, column=1, padx=6, pady=3)
        ttk.Button(actions, text="Открыть реестр CRM", style="Action.TButton", command=self.open_crm_registry).grid(row=0, column=2, padx=6, pady=3)

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
    app.log("Приложение запущено. Начните со стартовой кнопки OCR слева.")
    root.mainloop()


if __name__ == "__main__":
    main()
