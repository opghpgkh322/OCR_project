# Сборка OCR приложения в .exe

## 1) Подготовка окружения
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
```

## 2) Сборка
```bash
pyinstaller --noconfirm --onefile --windowed --name OCRFormsApp \
  --add-data "sheet_config.json;." \
  --add-data "dictionaries;dictionaries" \
  --add-data "scripts;scripts" \
  --add-data "ocr_app;ocr_app" \
  app_desktop.py
```

Готовый файл: `dist/OCRFormsApp.exe`.

## 3) Что нужно положить рядом с exe
При `--onefile` PyInstaller упакует код и ресурсы в бинарник, но рабочие папки для пользователя (`scans`, `dataset_review`, `exports`, `app_state`) создадутся автоматически при первом запуске.

## 4) Рекомендованный рабочий процесс кассира
1. Сканер сохраняет новые сканы в `scans/inbox` (или импорт через кнопку в приложении).
2. Кнопка **"Обработать новые сканы в CSV"**.
3. Кнопка **"Отправить в CRM только новых клиентов"**.
4. При необходимости — ручная коррекция и дообучение через вкладки приложения.
