LABEL_TO_CHAR = {
    "A_cyr": "А",
    "B_cyr": "Б",
    "V_cyr": "В",
    "G_cyr": "Г",
    "D_cyr": "Д",
    "E_cyr": "Е",
    "Yo_cyr": "Ё",
    "Zh_cyr": "Ж",
    "Z_cyr": "З",
    "I_cyr": "И",
    "Y_cyr": "Й",
    "K_cyr": "К",
    "L_cyr": "Л",
    "M_cyr": "М",
    "N_cyr": "Н",
    "O_cyr": "О",
    "P_cyr": "П",
    "R_cyr": "Р",
    "S_cyr": "С",
    "T_cyr": "Т",
    "U_cyr": "У",
    "F_cyr": "Ф",
    "Kh_cyr": "Х",
    "Ts_cyr": "Ц",
    "Ch_cyr": "Ч",
    "Sh_cyr": "Ш",
    "Shch_cyr": "Щ",
    "Hard_cyr": "Ъ",
    "Yery_cyr": "Ы",
    "Soft_cyr": "Ь",
    "E_rev_cyr": "Э",
    "Yu_cyr": "Ю",
    "Ya_cyr": "Я",
}

CHAR_TO_LABEL = {value: key for key, value in LABEL_TO_CHAR.items()}
DIGIT_LABELS = {str(num) for num in range(10)}
LETTER_LABELS = set(LABEL_TO_CHAR.keys())


def choose_allowed_label(probabilities, labels, allowed):
    if not allowed:
        return labels[int(probabilities.argmax())]
    allowed_indices = [idx for idx, label in enumerate(labels) if label in allowed]
    if not allowed_indices:
        return labels[int(probabilities.argmax())]
    allowed_probs = probabilities[allowed_indices]
    best_index = allowed_indices[int(allowed_probs.argmax())]
    return labels[best_index]
