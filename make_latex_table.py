import json

# Фиксированный порядок столбцов
COLUMN_ORDER = [
    "model-dems",
    "dems",
    "accuracy",
    "precision",
    "precision@3",
    "precision@5",
    "recall",
    "f1-score",
    "FMSE",
    "FMAE",
    "time"
]

# Маппинг ключей JSON на названия столбцов
KEY_MAPPING = {
    "model-dem": "model-dems",
    "precision+-1(@3)": "precision@3",
    "precision+-2(@5)": "precision@5",
    "f1-score": "f1-score",
    "FMSE": "FMSE",
    "FMAE": "FMAE",
    "time": "time",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall"
}

def normalize_row(row):
    """Нормализует ключи строки согласно маппингу"""
    normalized = {}
    for json_key, value in row.items():
        # Применяем маппинг, если есть, иначе используем исходный ключ
        display_key = KEY_MAPPING.get(json_key, json_key)
        # Сохраняем только ключи, которые есть в COLUMN_ORDER
        if display_key in COLUMN_ORDER:
            normalized[display_key] = value
    return normalized

def format_value(value):
    """Форматирует значение: округляет числа до 2 знаков, остальное оставляет как есть"""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)

def latex_header():
    # Формируем header для longtable с фиксированным порядком столбцов
    latex = "\\footnotesize\n\\begin{longtable}{|" + "|".join("p{1.3cm}" for _ in COLUMN_ORDER) + "|}\n\\hline\n"
    
    # Первые два столбца без поворота (model-dems, dems), остальные с поворотом
    header_parts = ["\\textbf{model-dems}", "\\textbf{dems}"]
    header_parts.extend([f"\\rotatebox{{90}}{{\\textbf{{{col}}}}}" for col in COLUMN_ORDER[2:]])
    latex += " & ".join(header_parts) + " \\\\\n" + "\\hline\n\\endfirsthead\n\n"
    
    latex += "\\hline\n"
    latex += " & ".join(header_parts) + " \\\\\n" + "\\hline\n\\endhead\n"
    return latex

def make_latex_table(table_rows, filename):
    if not table_rows:
        # Если нет данных, не создаем файл
        return
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_header())
        for row in table_rows:
            normalized_row = normalize_row(row)
            # Формируем строку с фиксированным порядком столбцов
            line_parts = []
            for col in COLUMN_ORDER:
                value = normalized_row.get(col, "")
                line_parts.append(format_value(value))
            line = " & ".join(line_parts) + " \\\\\n"
            f.write(line)
        f.write("\\hline\n\\end{longtable}")

# Загрузка данных и конвертация
with open("final_result.json", "r", encoding="utf-8") as f:
    results = json.load(f)

rows = []
for key, value in results.items():
    value["model-dem"] = key  # Можно использовать ключ как уникальное имя
    rows.append(value)

# 1. Все модели, только dem=avg
table1 = [row for row in rows if row.get("dems") == "avg"]
make_latex_table(table1, "all_models_dem_avg.tex")

# 2. Все демографии, только reformer
table2 = [row for row in rows if "model" in row and str(row.get("model", "")).startswith("reformer")]
make_latex_table(table2, "all_dem_reformer.tex")

# 3. Все демографии, только gru
table3 = [row for row in rows if "model" in row and str(row.get("model", "")).startswith("gru")]
make_latex_table(table3, "all_dem_gru.tex")

# 4. Все демографии, только lstm
table4 = [row for row in rows if "model" in row and str(row.get("model", "")).startswith("lstm")]
make_latex_table(table4, "all_dem_lstm.tex")


import json
import numpy as np

with open("final_result.json", "r", encoding="utf-8") as f:
    results = json.load(f)

rows = []
for key, value in results.items():
    value["model-dem"] = key
    rows.append(value)

# Динамически определяем архитектуры и числовые метрики
architectures = ["reformer", "gru", "lstm"]
exclude_keys = {"model-dem", "dems", "model"}
numeric_metrics = set()
for row in rows:
    for k, v in row.items():
        if k in exclude_keys:
            continue
        if isinstance(v, (int, float)):
            numeric_metrics.add(k)
numeric_metrics = sorted(list(numeric_metrics))

latex = "\\begin{center}\n"
latex += "\\begin{longtable}{|c|c|c|c|c|}\n"
latex += "\\hline\n"
latex += "\\rotatebox{90}{\\textbf{Архитектура}} &\n"
latex += "\\textbf{dems} &\n"
latex += "\\rotatebox{90}{\\textbf{Метрика}} &\n"
latex += "\\rotatebox{90}{\\textbf{Среднее}} &\n"
latex += "\\rotatebox{90}{\\textbf{90\\%-ДИ}} \\\\\n"
latex += "\\hline\n\\endfirsthead\n\n"
latex += "\\hline\n"
latex += "\\rotatebox{90}{\\textbf{Архитектура}} &\n"
latex += "\\textbf{dems} &\n"
latex += "\\rotatebox{90}{\\textbf{Метрика}} &\n"
latex += "\\rotatebox{90}{\\textbf{Среднее}} &\n"
latex += "\\rotatebox{90}{\\textbf{90\\%-ДИ}} \\\\\n"
latex += "\\hline\n\\endhead\n\n"

dems_values = sorted({row.get("dems") for row in rows if row.get("dems") is not None})

for arch in architectures:
    for dems in dems_values:
        arch_dem_rows = [
            row
            for row in rows
            if row.get("model", "").startswith(arch) and row.get("dems") == dems
        ]
        if not arch_dem_rows:
            continue
        for metric in numeric_metrics:
            vals = [row.get(metric) for row in arch_dem_rows if isinstance(row.get(metric), (int, float))]
            if not vals:
                continue
            mean = np.mean(vals)
            ci_low, ci_high = np.percentile(vals, [5, 95])
            latex += f"{arch} & {dems} & {metric.replace('+', '')} & {mean:.3f} & $[{ci_low:.3f},\\ {ci_high:.3f}]$ \\\\\n"
        latex += "\\hline\n"

latex += "\\caption{Интегрированное сравнение моделей}\n"
latex += "\\label{{ris:total_int}}\n"
latex += "\\end{longtable}\n\\end{center}\n"

with open("mean_models_metrics.tex", "w", encoding="utf-8") as f:
    f.write(latex)
