import argparse
import csv
import json
import os
import random
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

PASSWORD_TO_ROLE = {
    "opghpgkh": "root",
    "admin": "user",
}


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
        self.current_role = "user"

        self.notebook: ttk.Notebook | None = None
        self.quality_tab_index = 0
        self.train_tab_index = 1
        self.crm_tab_index = 2
        self.dictionary_tab_index = 3

        self.role_label: ttk.Label | None = None
        self.logo_photo: tk.PhotoImage | None = None
        self.log_text: tk.Text | None = None
        self._login_password_var = tk.StringVar()

        self.dict_field_var = tk.StringVar(value="surname")
        self.dict_search_var = tk.StringVar()
        self.dict_add_var = tk.StringVar()
        self.dict_add_female_var = tk.StringVar()
        self.dict_name_gender_var = tk.StringVar(value="m")
        self.dict_remove_var = tk.StringVar()

        self._configure_styles()
        self.show_login_view(initial=True)

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
        style.configure("HeaderRole.TLabel", background=BG_MAIN, foreground=COLOR_TEXT, font=("Segoe UI", 10, "bold"))

        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 10), foreground="#10250E", background=COLOR_ACCENT)
        style.map("Primary.TButton", background=[("active", COLOR_ACCENT_DARK), ("pressed", COLOR_ACCENT_DARK)])

        style.configure("Action.TButton", font=("Segoe UI", 10), padding=(10, 8), background="#E8F8DF")
        style.map("Action.TButton", background=[("active", "#D7F2CA")])

        style.configure("Notebook.TNotebook", background=BG_PANEL, borderwidth=0)
        style.configure("Notebook.TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(16, 10))

    def _clear_root(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()
        self.notebook = None
        self.role_label = None
        self.log_text = None

    def show_login_view(self, initial: bool) -> None:
        self._clear_root()

        wrap = ttk.Frame(self.root, style="Root.TFrame", padding=16)
        wrap.pack(fill=tk.BOTH, expand=True)
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        card = ttk.Frame(wrap, style="Card.TFrame", padding=20)
        card.grid(row=0, column=0)
        card.grid_columnconfigure(0, weight=1)

        ttk.Label(card, text="Авторизация", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        caption = "Введите пароль для входа"
        if not initial:
            caption = f"Текущий пользователь: {self.current_role}. Введите пароль для смены пользователя"

        ttk.Label(card, text=caption, style="Body.TLabel", wraplength=420, justify=tk.LEFT).grid(row=1, column=0, sticky="w", pady=(0, 10))

        password_row = ttk.Frame(card, style="Card.TFrame")
        password_row.grid(row=2, column=0, sticky="ew")
        ttk.Label(password_row, text="Пароль:", style="Body.TLabel").pack(side=tk.LEFT, padx=(0, 8))

        self._login_password_var.set("")
        password_entry = ttk.Entry(password_row, textvariable=self._login_password_var, show="*", width=32)
        password_entry.pack(side=tk.LEFT)
        password_entry.focus_set()

        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.grid(row=3, column=0, sticky="e", pady=(12, 0))

        ttk.Button(buttons, text="Отмена", style="Action.TButton", command=lambda: self._cancel_login(initial)).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Войти", style="Primary.TButton", command=lambda: self._submit_login(initial)).pack(side=tk.RIGHT)

        password_entry.bind("<Return>", lambda _e: self._submit_login(initial))

    def _submit_login(self, initial: bool) -> None:
        password = self._login_password_var.get().strip()
        role = PASSWORD_TO_ROLE.get(password)
        if role is None:
            messagebox.showerror("Ошибка", "Неверный пароль")
            self._login_password_var.set("")
            return

        self.current_role = role
        self._build_ui()
        self.apply_role_permissions()
        self.log(f"🔐 Выполнен вход: {self.current_role}")
        if initial:
            self.log("Приложение запущено. Войдите под root или user.")

    def _cancel_login(self, initial: bool) -> None:
        if initial:
            self.root.destroy()
            return
        self._build_ui()
        self.apply_role_permissions()
        self.log("ℹ️ Смена пользователя отменена")

    def _build_ui(self) -> None:
        self._clear_root()

        root_frame = ttk.Frame(self.root, style="Root.TFrame")
        root_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        root_frame.grid_columnconfigure(0, weight=0)
        root_frame.grid_columnconfigure(1, weight=1)
        root_frame.grid_rowconfigure(2, weight=1)

        self._build_header(root_frame)
        self._build_left_panel(root_frame)
        self._build_main_panel(root_frame)

    def _load_logo_image(self, logo_path: Path, target_height: int = 128) -> tk.PhotoImage | None:
        try:
            image = tk.PhotoImage(file=str(logo_path))
        except Exception as exc:  # noqa: BLE001
            print(f"[logo] failed to load {logo_path}: {exc}")
            return None

        height = max(1, image.height())
        if height > target_height:
            factor = max(1, int(round(height / target_height)))
            image = image.subsample(factor, factor)
        elif height < target_height:
            factor = max(1, int(round(target_height / height)))
            image = image.zoom(factor, factor)

        return image

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Root.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(2, weight=1)

        logo_path = self.repo_root / "assets" / "Sequoia_logo.svg"
        self.logo_photo = self._load_logo_image(logo_path) if logo_path.exists() else None

        if self.logo_photo is not None:
            logo_label = tk.Label(header, image=self.logo_photo, bg=BG_MAIN, bd=0, highlightthickness=0)
            logo_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(4, 12))
        else:
            fallback = tk.Canvas(header, width=128, height=128, bg=BG_MAIN, highlightthickness=0)
            fallback.grid(row=0, column=0, rowspan=2, sticky="w", padx=(4, 12))
            fallback.create_line(42, 28, 42, 108, fill=COLOR_TEXT, width=6)
            fallback.create_line(78, 42, 78, 108, fill=COLOR_TEXT, width=5)
            fallback.create_polygon(42, 18, 14, 54, 70, 54, fill=COLOR_TEXT, outline=COLOR_TEXT)
            fallback.create_polygon(42, 36, 18, 70, 66, 70, fill=COLOR_TEXT, outline=COLOR_TEXT)
            fallback.create_polygon(78, 34, 62, 58, 94, 58, fill=COLOR_TEXT, outline=COLOR_TEXT)

        ttk.Label(header, text="SEQUOIA PARK", style="HeaderTitle.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(
            header,
            text="OCR для рукописных бланков • Рабочая панель кассира",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=1, sticky="w")

        role_panel = ttk.Frame(header, style="Root.TFrame")
        role_panel.grid(row=0, column=2, rowspan=2, sticky="e")
        self.role_label = ttk.Label(role_panel, text="Роль: user", style="HeaderRole.TLabel")
        self.role_label.pack(anchor="e")
        ttk.Button(role_panel, text="Сменить пользователя", style="Action.TButton", command=self.switch_user).pack(anchor="e", pady=(8, 0))

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, style="Card.TFrame", padding=14)
        left.grid(row=1, column=0, rowspan=2, sticky="nsw", padx=(0, 12))

        ttk.Button(left, text="▶ Запустить OCR", style="Primary.TButton", command=self.run_ocr_batch).pack(fill=tk.X)

        ttk.Checkbutton(
            left,
            text="Включить словарную автокоррекцию ФИО",
            variable=self.correction_var,
        ).pack(anchor="w", pady=(10, 12))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Рабочие папки", style="SectionTitle.TLabel").pack(anchor="w", pady=(6, 6))
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
            "3) Отправьте данные о посетителях в CRM",
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

        self.notebook = ttk.Notebook(notebook_card, style="Notebook.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        quality_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        train_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        crm_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)
        dictionary_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=12)

        self.notebook.add(quality_tab, text="Качество распознавания")
        self.notebook.add(train_tab, text="Дообучение модели")
        self.notebook.add(crm_tab, text="CRM и выгрузка")
        self.notebook.add(dictionary_tab, text="Работа со словарём")

        self._build_quality_tab(quality_tab)
        self._build_train_tab(train_tab)
        self._build_crm_tab(crm_tab)
        self._build_dictionary_tab(dictionary_tab)

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
        ttk.Label(parent, text="Выгрузка в CRM", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))
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

    def _build_dictionary_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Работа со словарём", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(
            parent,
            text=(
                "Поиск/добавление/удаление словарных записей по части ФИО.\n"
                "Добавление всегда генерирует 10 строк в dictionaries/data.csv: 5 мужских и 5 женских."
            ),
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 10))

        field_row = ttk.Frame(parent, style="Card.TFrame")
        field_row.pack(anchor="w", fill=tk.X, pady=(0, 10))
        ttk.Label(field_row, text="Часть ФИО:", style="Body.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        field_combo = ttk.Combobox(
            field_row,
            textvariable=self.dict_field_var,
            values=("surname", "name", "patronymic"),
            state="readonly",
            width=18,
        )
        field_combo.pack(side=tk.LEFT)

        search_card = ttk.Frame(parent, style="Card.TFrame")
        search_card.pack(anchor="w", fill=tk.X, pady=(0, 8))
        ttk.Label(search_card, text="Поиск", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        search_row = ttk.Frame(search_card, style="Card.TFrame")
        search_row.pack(anchor="w", fill=tk.X)
        ttk.Entry(search_row, textvariable=self.dict_search_var, width=34).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(search_row, text="Найти", style="Action.TButton", command=self.search_dictionary).pack(side=tk.LEFT)

        add_card = ttk.Frame(parent, style="Card.TFrame")
        add_card.pack(anchor="w", fill=tk.X, pady=(0, 8))
        ttk.Label(add_card, text="Добавление", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        add_row = ttk.Frame(add_card, style="Card.TFrame")
        add_row.pack(anchor="w", fill=tk.X)
        ttk.Label(add_row, text="Основная форма:", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(add_row, textvariable=self.dict_add_var, width=26).pack(side=tk.LEFT, padx=(0, 10))

        add_row_2 = ttk.Frame(add_card, style="Card.TFrame")
        add_row_2.pack(anchor="w", fill=tk.X, pady=(4, 0))
        ttk.Label(add_row_2, text="Женская форма (для фамилии/отчества):", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(add_row_2, textvariable=self.dict_add_female_var, width=18).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(add_row_2, text="Род имени по умолчанию:", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(add_row_2, textvariable=self.dict_name_gender_var, values=("m", "f"), state="readonly", width=6).pack(side=tk.LEFT)

        add_row_3 = ttk.Frame(add_card, style="Card.TFrame")
        add_row_3.pack(anchor="w", fill=tk.X, pady=(4, 0))
        ttk.Label(
            add_row_3,
            text="Фамилия/отчество: заполните обе формы (для несклоняемых — одинаковые). Имя: заполните только основную форму.",
            style="Hint.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w")

        add_action_row = ttk.Frame(add_card, style="Card.TFrame")
        add_action_row.pack(anchor="w", fill=tk.X, pady=(6, 0))
        ttk.Button(add_action_row, text="Добавить", style="Action.TButton", command=self.add_dictionary_word).pack(side=tk.LEFT)

        remove_card = ttk.Frame(parent, style="Card.TFrame")
        remove_card.pack(anchor="w", fill=tk.X)
        ttk.Label(remove_card, text="Удаление", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        remove_row = ttk.Frame(remove_card, style="Card.TFrame")
        remove_row.pack(anchor="w", fill=tk.X)
        ttk.Entry(remove_row, textvariable=self.dict_remove_var, width=34).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(remove_row, text="Удалить", style="Action.TButton", command=self.remove_dictionary_word).pack(side=tk.LEFT)

    def _dictionary_jsonl_path(self, field: str) -> Path:
        dict_dir = self.repo_root / "dictionaries"
        if field == "surname":
            return dict_dir / "surnames_table.jsonl"
        if field == "name":
            return dict_dir / "names_table.jsonl"
        return dict_dir / "midnames_table.jsonl"

    def _dictionary_csv_path(self) -> Path:
        return self.repo_root / "dictionaries" / "data.csv"

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _all_texts_by_gender(self, field: str, gender: str) -> list[str]:
        rows = self._read_jsonl(self._dictionary_jsonl_path(field))
        values: list[str] = []
        for row in rows:
            text = str(row.get("text", "")).strip()
            row_gender = row.get("gender")
            if not text:
                continue
            if row_gender == gender or row_gender is None:
                values.append(text)
        return values

    def _word_exists(self, rows: list[dict], word: str, gender: str | None) -> bool:
        for row in rows:
            text = str(row.get("text", "")).strip().upper()
            row_gender = row.get("gender")
            if text == word and row_gender == gender:
                return True
            if text == word and row_gender is None and gender is None:
                return True
        return False

    def _fixed_gender_rows(self, field: str, male_value: str, female_value: str) -> list[tuple[str, str, str, str]]:
        male_names = self._all_texts_by_gender("name", "m") or ["АЛЕКСАНДР"]
        female_names = self._all_texts_by_gender("name", "f") or ["АННА"]
        male_mid = self._all_texts_by_gender("patronymic", "m") or ["АЛЕКСАНДРОВИЧ"]
        female_mid = self._all_texts_by_gender("patronymic", "f") or ["АЛЕКСАНДРОВНА"]
        surname_source = self._all_texts_by_gender("surname", "m") or ["ИВАНОВ"]
        surname_source_f = self._all_texts_by_gender("surname", "f") or ["ИВАНОВА"]

        rows: list[tuple[str, str, str, str]] = []
        for _ in range(5):
            if field == "surname":
                rows.append((male_value, random.choice(male_names), random.choice(male_mid), "M"))
            elif field == "name":
                rows.append((random.choice(surname_source), male_value, random.choice(male_mid), "M"))
            else:
                rows.append((random.choice(surname_source), random.choice(male_names), male_value, "M"))

        for _ in range(5):
            if field == "surname":
                rows.append((female_value, random.choice(female_names), random.choice(female_mid), "F"))
            elif field == "name":
                rows.append((random.choice(surname_source_f), female_value, random.choice(female_mid), "F"))
            else:
                rows.append((random.choice(surname_source_f), random.choice(female_names), female_value, "F"))
        return rows

    def _append_csv_rows(self, rows: list[tuple[str, str, str, str]]) -> None:
        csv_path = self._dictionary_csv_path()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for row in rows:
                writer.writerow(row)

    def _remove_from_csv(self, field: str, value: str) -> int:
        csv_path = self._dictionary_csv_path()
        if not csv_path.exists():
            return 0

        index_map = {"surname": 0, "name": 1, "patronymic": 2}
        idx = index_map[field]
        target = value.upper()

        kept: list[list[str]] = []
        removed = 0
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 4:
                    kept.append(row)
                    continue
                if row[idx].strip().upper() == target:
                    removed += 1
                    continue
                kept.append(row)

        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(kept)

        return removed

    def search_dictionary(self) -> None:
        field = self.dict_field_var.get().strip()
        query = self.dict_search_var.get().strip().upper()
        if not query:
            messagebox.showwarning("Поиск", "Введите слово для поиска")
            return

        rows = self._read_jsonl(self._dictionary_jsonl_path(field))
        matches = [row for row in rows if str(row.get("text", "")).strip().upper() == query]
        self.log(f"🔎 Поиск [{field}] '{query}': найдено {len(matches)} совпадений")

    def add_dictionary_word(self) -> None:
        field = self.dict_field_var.get().strip()
        male_value = self.dict_add_var.get().strip().upper()
        female_value = self.dict_add_female_var.get().strip().upper()
        default_name_gender = self.dict_name_gender_var.get().strip()

        if len(male_value) < 2:
            messagebox.showwarning("Добавление", "Введите основную форму (минимум 2 символа)")
            return

        if field in {"surname", "patronymic"} and len(female_value) < 2:
            messagebox.showwarning("Добавление", "Для фамилии/отчества заполните мужскую и женскую форму")
            return

        if field == "name":
            female_value = male_value
            if default_name_gender not in {"m", "f"}:
                messagebox.showwarning("Добавление", "Для имени выберите род по умолчанию: m или f")
                return

        jsonl_path = self._dictionary_jsonl_path(field)
        rows = self._read_jsonl(jsonl_path)

        added_entries: list[dict] = []
        if field in {"surname", "patronymic"}:
            if not self._word_exists(rows, male_value, "m"):
                added_entries.append({"text": male_value, "gender": "m"})
            if not self._word_exists(rows, female_value, "f"):
                added_entries.append({"text": female_value, "gender": "f"})
        else:
            if not self._word_exists(rows, male_value, default_name_gender):
                added_entries.append({"text": male_value, "gender": default_name_gender})

        if added_entries:
            rows.extend(added_entries)
            self._write_jsonl(jsonl_path, rows)

        generated_rows = self._fixed_gender_rows(field, male_value, female_value)
        self._append_csv_rows(generated_rows)

        self.log(
            f"➕ Добавление [{field}] '{male_value}'"
            f"{' / ' + female_value if field in {'surname', 'patronymic'} else ''}: "
            f"добавлено в JSONL {len(added_entries)} записей, сгенерировано в data.csv {len(generated_rows)} строк (5M/5F)"
        )
        self.dict_add_var.set("")
        self.dict_add_female_var.set("")

    def remove_dictionary_word(self) -> None:
        field = self.dict_field_var.get().strip()
        value = self.dict_remove_var.get().strip().upper()
        if not value:
            messagebox.showwarning("Удаление", "Введите слово для удаления")
            return

        jsonl_path = self._dictionary_jsonl_path(field)
        rows = self._read_jsonl(jsonl_path)
        before = len(rows)
        filtered = [r for r in rows if str(r.get("text", "")).strip().upper() != value]
        removed_jsonl = before - len(filtered)
        if removed_jsonl > 0:
            self._write_jsonl(jsonl_path, filtered)

        removed_csv = self._remove_from_csv(field, value)

        self.log(
            f"➖ Удаление [{field}] '{value}': удалено из JSONL {removed_jsonl}, "
            f"удалено из data.csv {removed_csv} строк"
        )
        self.dict_remove_var.set("")

    def switch_user(self) -> None:
        self.show_login_view(initial=False)

    def apply_role_permissions(self) -> None:
        if self.role_label is not None:
            self.role_label.configure(text=f"Роль: {self.current_role}")

        if self.notebook is None:
            return

        is_root = self.current_role == "root"
        self.notebook.tab(self.quality_tab_index, state="normal" if is_root else "hidden")
        self.notebook.tab(self.train_tab_index, state="normal" if is_root else "hidden")
        self.notebook.tab(self.crm_tab_index, state="normal")
        self.notebook.tab(self.dictionary_tab_index, state="normal")

        if not is_root:
            self.notebook.select(self.crm_tab_index)

    def _require_root(self) -> bool:
        if self.current_role == "root":
            return True
        messagebox.showwarning("Недостаточно прав", "Для этого действия нужен root-пароль")
        return False

    def log(self, text: str) -> None:
        if self.log_text is None:
            print(text)
            return
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
        if not self._require_root():
            return
        command = [self.python, str(self.repo_root / "scripts" / "02_configurator.py")]
        self.run_command(command, "Редактор координат")

    def run_review(self) -> None:
        if not self._require_root():
            return
        command = [self.python, str(self.repo_root / "scripts" / "interactive_trainer.py")]
        self.run_command(command, "Коррекция распознавания")

    def run_invert_review(self) -> None:
        if not self._require_root():
            return
        command = [self.python, str(self.repo_root / "invert_dataset_review.py")]
        self.run_command(command, "Invert dataset_review")

    def run_training(self) -> None:
        if not self._require_root():
            return

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
    OCRDesktopApp(root=root, repo_root=Path(args.repo_root))
    root.mainloop()


if __name__ == "__main__":
    main()
