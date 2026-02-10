import json
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow import keras
from ocr_app.preprocessing import load_image, align_image, preprocess_cell
from ocr_app.config import SheetConfig

# Маппинг кириллица <-> папки
CYR_TO_LAT: Dict[str, str] = {
    "А": "A_cyr", "Б": "B_cyr", "В": "V_cyr", "Г": "G_cyr", "Д": "D_cyr",
    "Е": "E_cyr", "Ё": "Yo_cyr", "Ж": "Zh_cyr", "З": "Z_cyr", "И": "I_cyr",
    "Й": "Y_cyr", "К": "K_cyr", "Л": "L_cyr", "М": "M_cyr", "Н": "N_cyr",
    "О": "O_cyr", "П": "P_cyr", "Р": "R_cyr", "С": "S_cyr", "Т": "T_cyr",
    "У": "U_cyr", "Ф": "F_cyr", "Х": "Kh_cyr", "Ц": "Ts_cyr", "Ч": "Ch_cyr",
    "Ш": "Sh_cyr", "Щ": "Shch_cyr", "Ъ": "Hard_cyr", "Ы": "Yery_cyr",
    "Ь": "Soft_cyr", "Э": "E_rev_cyr", "Ю": "Yu_cyr", "Я": "Ya_cyr",
}

LAT_TO_CYR: Dict[str, str] = {v: k for k, v in CYR_TO_LAT.items()}


class CorrectionSession:
    """Хранит исправления текущей сессии"""

    def __init__(self):
        self.corrections: List[Tuple[np.ndarray, str, str]] = []

    def add(self, image: np.ndarray, filename: str, label_folder: str):
        self.corrections.append((image.copy(), filename, label_folder))

    def save_all(self, review_dir: Path):
        if not self.corrections:
            return 0

        review_dir.mkdir(parents=True, exist_ok=True)
        for img, filename, label_folder in self.corrections:
            target_dir = review_dir / label_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target_dir / filename), img)

        return len(self.corrections)


class CorrectorApp:
    def __init__(self, model, labels, img_size, sheet_config, scans_dir, review_dir):
        self.model = model
        self.labels = labels
        self.img_size = img_size
        self.sheet_config = sheet_config
        self.scans_dir = scans_dir
        self.review_dir = review_dir

        self.session = CorrectionSession()
        self.current_items = []  # Список всех ячеек для проверки
        self.current_index = 0

        # Создаем главное окно
        self.root = tk.Tk()
        self.root.title("OCR Corrector - Ручная коррекция")
        self.root.geometry("800x600")
        self.root.configure(bg="#2b2b2b")

        self.build_ui()
        self.load_all_cells()

        if self.current_items:
            self.show_current_cell()
        else:
            messagebox.showwarning("Нет данных", "В папке scans/ нет изображений!")
            self.root.destroy()

    def label_to_char(self, label: str) -> str:
        return LAT_TO_CYR.get(label, label)

    def char_to_label(self, char: str) -> str:
        return CYR_TO_LAT.get(char.upper(), char)

    def build_ui(self):
        """Создаем интерфейс"""

        # === ВЕРХНЯЯ ПАНЕЛЬ (Прогресс) ===
        top_frame = tk.Frame(self.root, bg="#1e1e1e", height=60)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.progress_label = tk.Label(
            top_frame,
            text="Загрузка...",
            font=("Arial", 12),
            bg="#1e1e1e",
            fg="#ffffff"
        )
        self.progress_label.pack(pady=15)

        # === ЦЕНТРАЛЬНАЯ ЧАСТЬ (Картинка + Инфо) ===
        center_frame = tk.Frame(self.root, bg="#2b2b2b")
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Картинка слева
        self.image_label = tk.Label(center_frame, bg="#1e1e1e", borderwidth=2, relief="solid")
        self.image_label.pack(side=tk.LEFT, padx=(0, 20))

        # Информация справа
        info_frame = tk.Frame(center_frame, bg="#2b2b2b")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(info_frame, text="Предсказание AI:", font=("Arial", 11, "bold"),
                 bg="#2b2b2b", fg="#aaaaaa").pack(anchor="w", pady=(0, 5))

        self.prediction_label = tk.Label(
            info_frame,
            text="?",
            font=("Arial", 48, "bold"),
            bg="#2b2b2b",
            fg="#00ff00"
        )
        self.prediction_label.pack(anchor="w", pady=(0, 10))

        self.confidence_label = tk.Label(
            info_frame,
            text="Уверенность: -",
            font=("Arial", 10),
            bg="#2b2b2b",
            fg="#cccccc"
        )
        self.confidence_label.pack(anchor="w", pady=(0, 5))

        self.field_label = tk.Label(
            info_frame,
            text="Поле: -",
            font=("Arial", 9),
            bg="#2b2b2b",
            fg="#888888"
        )
        self.field_label.pack(anchor="w", pady=(0, 20))

        # Разделитель
        tk.Frame(info_frame, bg="#444444", height=1).pack(fill=tk.X, pady=10)

        # Поле коррекции
        tk.Label(info_frame, text="Ваша коррекция:", font=("Arial", 11, "bold"),
                 bg="#2b2b2b", fg="#aaaaaa").pack(anchor="w", pady=(10, 5))

        self.correction_entry = tk.Entry(
            info_frame,
            font=("Arial", 32),
            bg="#1e1e1e",
            fg="#ffff00",
            insertbackground="#ffff00",
            relief="solid",
            borderwidth=2,
            justify="center"
        )
        self.correction_entry.pack(fill=tk.X, ipady=10)
        self.correction_entry.bind("<Return>", lambda e: self.accept_correction())

        # === НИЖНЯЯ ПАНЕЛЬ (Кнопки) ===
        button_frame = tk.Frame(self.root, bg="#2b2b2b", height=80)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        btn_style = {
            "font": ("Arial", 12, "bold"),
            "height": 2,
            "relief": "raised",
            "borderwidth": 2
        }

        self.btn_skip = tk.Button(
            button_frame,
            text="⏭ Пропустить (верно)",
            bg="#4CAF50",
            fg="white",
            command=self.skip_cell,
            **btn_style
        )
        self.btn_skip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.btn_correct = tk.Button(
            button_frame,
            text="✔ Исправить и сохранить",
            bg="#2196F3",
            fg="white",
            command=self.accept_correction,
            **btn_style
        )
        self.btn_correct.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.btn_finish = tk.Button(
            button_frame,
            text="🚪 Завершить сессию",
            bg="#f44336",
            fg="white",
            command=self.finish_session,
            **btn_style
        )
        self.btn_finish.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def load_all_cells(self):
        """Загружаем все ячейки из всех сканов"""
        scans = list(self.scans_dir.glob("*.jpg")) + list(self.scans_dir.glob("*.png"))

        for scan_path in scans:
            try:
                full_img = load_image(str(scan_path))
                aligned = align_image(full_img,
                                      (self.sheet_config.image_width,
                                       self.sheet_config.image_height))
            except Exception as e:
                print(f"⚠ Ошибка чтения {scan_path.name}: {e}")
                continue

            for cell in self.sheet_config.cells:
                crop = aligned[cell.y: cell.y + cell.h, cell.x: cell.x + cell.w]
                processed = preprocess_cell(crop, self.img_size)

                batch = np.expand_dims(np.array([processed]), axis=-1)
                preds = self.model.predict(batch, verbose=0)[0]
                pred_idx = np.argmax(preds)
                confidence = preds[pred_idx]
                pred_label = self.labels[pred_idx]
                pred_char = self.label_to_char(pred_label)

                # Сохраняем всю инфу
                self.current_items.append({
                    "scan_name": scan_path.stem,
                    "crop": crop,
                    "processed": processed,
                    "pred_char": pred_char,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "field": cell.label,
                    "index": cell.index
                })

    def show_current_cell(self):
        """Показывает текущую ячейку"""
        if self.current_index >= len(self.current_items):
            self.finish_session()
            return

        item = self.current_items[self.current_index]

        # Обновляем прогресс
        self.progress_label.config(
            text=f"Проверено: {self.current_index} / {len(self.current_items)} | "
                 f"Исправлений: {len(self.session.corrections)}"
        )

        # Показываем картинку
        display_crop = cv2.resize(item["crop"], (300, 300))
        if len(display_crop.shape) == 2:
            display_crop = cv2.cvtColor(display_crop, cv2.COLOR_GRAY2RGB)
        else:
            display_crop = cv2.cvtColor(display_crop, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(display_crop)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Сохраняем ссылку

        # Обновляем инфо
        self.prediction_label.config(text=item["pred_char"])

        conf_color = "#00ff00" if item["confidence"] > 0.85 else "#ffa500"
        self.confidence_label.config(
            text=f"Уверенность: {item['confidence'] * 100:.1f}%",
            fg=conf_color
        )

        self.field_label.config(text=f"Поле: {item['field']} #{item['index']}")

        # Очищаем поле ввода и ставим туда предсказание как подсказку
        self.correction_entry.delete(0, tk.END)
        self.correction_entry.insert(0, item["pred_char"])
        self.correction_entry.select_range(0, tk.END)
        self.correction_entry.focus()

    def skip_cell(self):
        """Пропускаем ячейку (согласны с предсказанием)"""
        self.current_index += 1
        self.show_current_cell()

    def accept_correction(self):
        """Сохраняем исправление"""
        if self.current_index >= len(self.current_items):
            return

        item = self.current_items[self.current_index]
        user_input = self.correction_entry.get().strip().upper()

        if not user_input:
            user_input = item["pred_char"]

        # Если пользователь исправил - добавляем в сессию
        if user_input != item["pred_char"]:
            final_label = self.char_to_label(user_input)
            filename = f"{item['scan_name']}_{item['field']}_{item['index']}_{np.random.randint(10000)}.png"
            save_img = (item["processed"] * 255).astype(np.uint8)
            self.session.add(save_img, filename, final_label)
            print(f"✏ Исправлено: {item['pred_char']} → {user_input}")

        self.current_index += 1
        self.show_current_cell()

    def finish_session(self):
        """Завершаем сессию и спрашиваем о сохранении"""
        if not self.session.corrections:
            messagebox.showinfo("Сессия завершена", "Исправлений не было.")
            self.root.destroy()
            return

        answer = messagebox.askyesno(
            "Сохранить исправления?",
            f"В этой сессии исправлено {len(self.session.corrections)} примеров.\n\n"
            "Сохранить их для дообучения?"
        )

        if answer:
            count = self.session.save_all(self.review_dir)
            messagebox.showinfo(
                "Готово!",
                f"✓ Сохранено исправлений: {count}\n\n"
                "Теперь запустите дообучение:\n"
                "python scripts/train_model.py --fine-tune --epochs 10"
            )
        else:
            messagebox.showinfo("Отменено", "Исправления не сохранены.")

        self.root.destroy()

    def run(self):
        """Запускаем приложение"""
        self.root.protocol("WM_DELETE_WINDOW", self.finish_session)
        self.root.mainloop()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "scripts" / "model"
    scans_dir = repo_root / "scans"
    review_dir = repo_root / "dataset_review"

    print("🔄 Загрузка модели...")
    try:
        model = keras.models.load_model(model_dir / "ocr_model.keras")
        labels = json.loads((model_dir / "labels.json").read_text("utf-8"))
        img_size_arr = np.load(model_dir / "image_size.npy")
        img_size = (int(img_size_arr[0]), int(img_size_arr[1]))
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Сначала обучите модель: python scripts/train_model.py")
        return

    try:
        sheet_config = SheetConfig.load(repo_root / "sheet_config.json")
    except Exception as e:
        print(f"❌ Ошибка загрузки конфига: {e}")
        return

    print("✓ Модель загружена. Запуск GUI...")
    app = CorrectorApp(model, labels, img_size, sheet_config, scans_dir, review_dir)
    app.run()


if __name__ == "__main__":
    main()
