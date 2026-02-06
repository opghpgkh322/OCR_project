LABEL_TO_CHAR = {
    "A_cyr": "А", "B_cyr": "Б", "V_cyr": "В", "G_cyr": "Г", "D_cyr": "Д",
    "E_cyr": "Е", "Yo_cyr": "Ё", "Zh_cyr": "Ж", "Z_cyr": "З", "I_cyr": "И",
    "Y_cyr": "Й", "K_cyr": "К", "L_cyr": "Л", "M_cyr": "М", "N_cyr": "Н",
    "O_cyr": "О", "P_cyr": "П", "R_cyr": "Р", "S_cyr": "С", "T_cyr": "Т",
    "U_cyr": "У", "F_cyr": "Ф", "Kh_cyr": "Х", "Ts_cyr": "Ц", "Ch_cyr": "Ч",
    "Sh_cyr": "Ш", "Shch_cyr": "Щ", "Hard_cyr": "Ъ", "Yery_cyr": "Ы",
    "Soft_cyr": "Ь", "E_rev_cyr": "Э", "Yu_cyr": "Ю", "Ya_cyr": "Я",
    "Empty": ""  # <--- ДОБАВЛЕНО
}

CHAR_TO_LABEL = {value: key for key, value in LABEL_TO_CHAR.items() if value != ""}
# Добавляем Empty в обратный маппинг вручную, если нужно, но обычно нет.

DIGIT_LABELS = {str(num) for num in range(10)}
DIGIT_LABELS.add("Empty")  # Цифровое поле может быть пустым

# Буквенное поле может быть пустым. Исключаем цифры из букв.
LETTER_LABELS = set(k for k in LABEL_TO_CHAR.keys() if k not in DIGIT_LABELS and k != "Empty")
LETTER_LABELS.add("Empty")


def choose_allowed_label(probabilities, labels, allowed):
    if not allowed:
        return labels[int(probabilities.argmax())]

    # Фильтруем индексы, разрешенные для данного поля
    allowed_indices = [idx for idx, label in enumerate(labels) if label in allowed]

    if not allowed_indices:
        return labels[int(probabilities.argmax())]

    allowed_probs = probabilities[allowed_indices]
    best_local_index = int(allowed_probs.argmax())
    best_global_index = allowed_indices[best_local_index]

    return labels[best_global_index]