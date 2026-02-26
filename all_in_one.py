# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from deepgene.data import data_gen_np as dgn

"""


В каждлом json вот это

                data = {
                    "model_name": troika,
                    "model_path": path,
                    "dem_name": dname,
                    "model_dist": model_dist,
                    "prior_dist": prior_dist,
                    "Ne_pred": Ne_pred,
                    "Ne_true": Ne_true,
                    "confusion_matrix": final_confusion_matrix.tolist(),
                    "eval_time": eval_time,
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "f1": f1.tolist(),
                    "cls_report": cls_report,
                }

"""


def find_models_in_directory(base_path: str) -> Dict[str, List[str]]:
    """
    Автоматически находит все модели в директории результатов mark_all.py
    
    Структура: {base_path}/{model_type}/{model_name}/{demography}/result.json
    
    Args:
        base_path: Базовая директория с результатами
        
    Returns:
        Словарь {model_type: [model_names]}
    """
    models_dict = {}
    base_path_obj = Path(base_path)
    
    if not base_path_obj.exists():
        return models_dict
    
    # Ожидаемые типы моделей
    model_types = ["gru", "lstm", "reformer"]
    
    for model_type in model_types:
        model_type_dir = base_path_obj / model_type
        if not model_type_dir.exists():
            continue
        
        model_names = []
        # Ищем все поддиректории в директории типа модели
        for item in model_type_dir.iterdir():
            if item.is_dir():
                # Проверяем, есть ли хотя бы один result.json в поддиректориях
                has_results = False
                for dem_dir in item.iterdir():
                    if dem_dir.is_dir():
                        result_file = dem_dir / "result.json"
                        if result_file.exists():
                            has_results = True
                            break
                
                if has_results:
                    model_names.append(item.name)
        
        if model_names:
            # Убираем дубликаты и сортируем
            model_names = sorted(list(set(model_names)))
            models_dict[model_type] = model_names
    
    return models_dict


def find_demographics_for_model(base_path: str, model_type: str, model_name: str) -> List[str]:
    """
    Находит все демографии для конкретной модели
    
    Args:
        base_path: Базовая директория
        model_type: Тип модели (gru/lstm/reformer)
        model_name: Имя модели
        
    Returns:
        Список демографий
    """
    model_dir = Path(base_path) / model_type / model_name
    if not model_dir.exists():
        return []
    
    dems = []
    for item in model_dir.iterdir():
        if item.is_dir():
            result_file = item / "result.json"
            if result_file.exists():
                dems.append(item.name)
    
    return sorted(dems)


models_path = {
    "reformer": [
        "reformer-ce-const", "reformer-ce-hl", "reformer-ce-set1", "reformer-ce-set2", "reformer-ce-set3",
        "reformer-kl-const", "reformer-kl-hl", "reformer-kl-set1", "reformer-kl-set2", "reformer-kl-set3"
    ],
    "lstm": [
        "lstm-ce-const", "lstm-ce-hl", "lstm-ce-set1", "lstm-ce-set2", "lstm-ce-set3",
        "lstm-kl-const", "lstm-kl-hl", "lstm-kl-set1", "lstm-kl-set2", "lstm-kl-set3"
    ],
    "gru": [
        "gru-ce-const", "gru-ce-hl", "gru-ce-set1", "gru-ce-set2", "gru-ce-set3",
        "gru-kl-const", "gru-kl-hl", "gru-kl-set1", "gru-kl-set2", "gru-kl-set3"
    ]
}

# Значения по умолчанию (для обратной совместимости)
bpath = "results"
dems = ["const", "hl", "zigzag"]
fpath = "result.json"



res_dic = {}


def fill_nan_with_previous(data):
    """
    Заменяет все NaN и Infinity в массиве или списке значениями предыдущих элементов.

    Parameters:
        data (list or np.ndarray): Входные данные с возможными NaN.

    Returns:
        np.ndarray: Массив с заменёнными NaN.
    """
    arr = np.array(data, dtype=float)  # Преобразуем в numpy-массив с типом float
    
    if len(arr) == 0:
        return arr
    
    # Если первый элемент невалиден, найти первый валидный и заполнить им начало
    if np.isnan(arr[0]) or np.isinf(arr[0]):
        first_valid_idx = None
        for i in range(len(arr)):
            if not (np.isnan(arr[i]) or np.isinf(arr[i])):
                first_valid_idx = i
                break
        
        # Если нашли валидное значение, заполняем начало
        if first_valid_idx is not None:
            arr[:first_valid_idx] = arr[first_valid_idx]
        else:
            # Если все значения невалидны, возвращаем массив NaN
            return np.full_like(arr, np.nan)

    # Заполняем остальные невалидные значения предыдущими
    for i in range(1, len(arr)):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            arr[i] = arr[i-1]

    return arr

def fill_nan(data):
    """
    Заменяет все NaN в массиве или списке значениями предыдущих элементов.

    Parameters:
        data (list or np.ndarray): Входные данные с возможными NaN.

    Returns:
        np.ndarray: Массив с заменёнными NaN.
    """
    arr = np.array(data, dtype=float)  # Преобразуем в numpy-массив с типом float

    for i in range(1, len(arr)):
        if np.isinf(arr[i]):
            arr[i] = 999_999
        if np.isnan(arr[i]):
            arr[i] = 1

    return arr


def trim_arrays(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    # Находим длину без nan/inf с конца для каждого массива
    def valid_length(x):
        mask = ~(np.isnan(x) | np.isinf(x))
        if not np.any(mask):
            return 0
        return np.argmax(mask[::-1])  # сколько невалидных в конце
    cut_a = len(a) - valid_length(a)
    cut_b = len(b) - valid_length(b)

    # Обрезаем оба массива до минимальной длины без nan/inf
    cut = min(cut_a, cut_b)
    return a[:cut], b[:cut]

def _max_intervals_for_time(max_time: float | None, fallback_max_intervals: int | None = None) -> int | None:
    """
    Определяет количество интервалов по массиву границ dgn.limits,
    которые полностью лежат до max_time поколений назад.

    Возвращает число интервалов (может быть 0) или None, если max_time не задан.
    Если указан fallback_max_intervals, то берётся минимум из двух ограничений.
    """
    if max_time is None:
        return fallback_max_intervals

    limits = np.array(dgn.limits, dtype=float)
    # Ищем последний индекс границы, не превышающей max_time
    idx = 0
    for i in range(1, len(limits)):
        if limits[i] <= max_time:
            idx = i
        else:
            break

    if idx <= 0:
        effective = 0
    else:
        effective = idx  # количество интервалов с верхней границей <= max_time

    if fallback_max_intervals is not None:
        effective = min(effective, fallback_max_intervals)

    return effective or 0


def fmse(y_true, y_pred, max_intervals=30, max_time: float | None = None):
    # Определяем эффективное количество интервалов
    eff_max_intervals = _max_intervals_for_time(max_time, fallback_max_intervals=max_intervals)

    # Обрезаем до первых eff_max_intervals интервалов
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if eff_max_intervals is not None and eff_max_intervals > 0:
        if len(y_true) > eff_max_intervals:
            y_true = y_true[:eff_max_intervals]
        if len(y_pred) > eff_max_intervals:
            y_pred = y_pred[:eff_max_intervals]
    
    # Сначала заполняем невалидные значения предыдущими валидными
    y_true = fill_nan_with_previous(y_true)
    y_pred = fill_nan_with_previous(y_pred)

    # Затем обрезаем невалидные значения с конца
    y_true, y_pred =  trim_arrays(y_true, y_pred)
    
    # Проверка на пустые массивы после обработки
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Проверка на все невалидные значения
    if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    
    # Избегаем деления на ноль
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    
    relative_errors = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return np.mean(relative_errors ** 2)

def rmse(y_true, y_pred, max_intervals=30, max_time: float | None = None):
    """Вычисляет RMSE (Root Mean Squared Error) - корень из FMSE."""
    fmse_val = fmse(y_true, y_pred, max_intervals=max_intervals, max_time=max_time)
    if np.isnan(fmse_val):
        return np.nan
    return np.sqrt(fmse_val)

def fmae(y_true, y_pred, max_intervals=30, max_time: float | None = None):
    # Определяем эффективное количество интервалов
    eff_max_intervals = _max_intervals_for_time(max_time, fallback_max_intervals=max_intervals)

    # Обрезаем до первых eff_max_intervals интервалов
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if eff_max_intervals is not None and eff_max_intervals > 0:
        if len(y_true) > eff_max_intervals:
            y_true = y_true[:eff_max_intervals]
        if len(y_pred) > eff_max_intervals:
            y_pred = y_pred[:eff_max_intervals]
    
    # Сначала заполняем невалидные значения предыдущими валидными
    y_true = fill_nan_with_previous(y_true)
    y_pred = fill_nan_with_previous(y_pred)
    
    # Затем обрезаем невалидные значения с конца
    y_true, y_pred =  trim_arrays(y_true, y_pred)
    
    # Проверка на пустые массивы после обработки
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Проверка на все невалидные значения
    if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    
    # Избегаем деления на ноль
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    
    relative_errors = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return np.mean(abs(relative_errors))

def fmae_v2(y_true, y_pred, max_time: float | None = None):
    y_true = fill_nan_with_previous(np.array(y_true))
    y_pred = fill_nan_with_previous(np.array(y_pred))

    # При наличии max_time обрезаем по времени
    eff_max_intervals = _max_intervals_for_time(max_time, fallback_max_intervals=None)
    if eff_max_intervals is not None and eff_max_intervals > 0:
        y_true = y_true[:eff_max_intervals]
        y_pred = y_pred[:eff_max_intervals]
    
    y_true, y_pred =  trim_arrays(y_true, y_pred)
    
    # Проверка на пустые массивы после обработки
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Проверка на все невалидные значения
    if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    
    relative_errors = (y_true - y_pred)
    return np.mean(abs(relative_errors))

def ae_dist(y_true, y_pred, max_time: float | None = None):
    y_true = fill_nan_with_previous(np.array(y_true))
    y_pred = fill_nan_with_previous(np.array(y_pred))

    # При наличии max_time обрезаем по времени
    eff_max_intervals = _max_intervals_for_time(max_time, fallback_max_intervals=None)
    if eff_max_intervals is not None and eff_max_intervals > 0:
        y_true = y_true[:eff_max_intervals]
        y_pred = y_pred[:eff_max_intervals]
    
    y_true, y_pred =  trim_arrays(y_true, y_pred)
    
    # Чтобы избежать деления на ноль, можно добавить небольшое значение eps
    relative_errors = (y_true - y_pred) / 10_000
    return relative_errors

def full_error(y_true, y_pred, N0=10_000, max_time: float | None = None):
    y_true = fill_nan_with_previous(np.array(y_true))
    y_pred = fill_nan_with_previous(np.array(y_pred))

    # При наличии max_time обрезаем по времени
    eff_max_intervals = _max_intervals_for_time(max_time, fallback_max_intervals=None)
    if eff_max_intervals is not None and eff_max_intervals > 0:
        y_true = y_true[:eff_max_intervals]
        y_pred = y_pred[:eff_max_intervals]
    
    y_true, y_pred =  trim_arrays(y_true, y_pred)
    
    # Проверка на пустые массивы после обработки
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Проверка на все невалидные значения
    if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    
    relative_errors = (y_true - y_pred) / (N0)
    return np.sum(abs(relative_errors))

# Определяем, использовать ли автоматический поиск или хардкод
use_auto_discovery = False
output_file = "all_in_one_30_intervals"

# Парсинг аргументов командной строки
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Агрегация результатов mark_all.py')
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Директория с результатами mark_all.py (по умолчанию: results)")
    parser.add_argument("--output_file", type=str, default="final_result",
                        help="Имя выходного файла без расширения (по умолчанию: final_result)")
    parser.add_argument("--auto", action="store_true",
                        help="Автоматически находить все модели в директории")
    args = parser.parse_args()
    
    if args.input_dir:
        bpath = args.input_dir
        use_auto_discovery = True
    
    if args.auto:
        use_auto_discovery = True
    
    if args.output_file:
        output_file = args.output_file

# Автоматическое обнаружение моделей, если включено
if use_auto_discovery:
    print("Автоматический поиск моделей в директории: {}".format(bpath))
    models_path = find_models_in_directory(bpath)
    if not models_path:
        print("Предупреждение: не найдено моделей в {}, используется хардкод".format(bpath))
        use_auto_discovery = False
    else:
        print("Найдено моделей: {}".format(models_path))

for key in models_path.keys():
    for model in models_path[key]:

        integral_time = 0
        integral_FMSE = 0
        integral_RMSE = 0
        integral_FMAE = 0
        integral_FMAE2 = 0
        integral_full_error = 0
        

        integral_accuracy = 0
        integral_Precision = 0
        integral_Recall = 0
        integral_F1 = 0
        iprecision_3 = 0
        iprecision_5 = 0

        # Определяем список демографий для этой модели
        if use_auto_discovery:
            model_dems = find_demographics_for_model(bpath, key, model)
            if not model_dems:
                print("Предупреждение: не найдено демографий для {}/{}, пропускаем".format(key, model))
                continue
        else:
            model_dems = dems
        
        for dem in model_dems:
            # В рамках этого цикла просчитываем метрики
            
            path = os.path.join(bpath, key, model, dem, fpath)

            print("Обработка: {}".format(path))

            if not os.path.exists(path):
                print("Предупреждение: файл не найден {}, пропускаем".format(path))
                continue

            try:
                with open(path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
            except Exception as e:
                print("Ошибка при чтении {}: {}, пропускаем".format(path, e))
                continue 

            model_time = np.mean(data["eval_time"])

            # Расчет FMSE, RMSE, FMAE только до 100 000 поколений назад
            # (ограничение по времени, а не только по числу интервалов)
            MAX_TIME = 100_000

            model_FMSE = fmse(data["Ne_true"], data["Ne_pred"], max_intervals=30, max_time=MAX_TIME)
            model_RMSE = rmse(data["Ne_true"], data["Ne_pred"], max_intervals=30, max_time=MAX_TIME)
            model_FMAE = fmae(data["Ne_true"], data["Ne_pred"], max_intervals=30, max_time=MAX_TIME)
            model_FMAE2 = fmae_v2(data["Ne_true"], data["Ne_pred"], max_time=MAX_TIME)
            # model_AE_dist = ae_dist(data["Ne_true"], data["Ne_pred"], max_time=MAX_TIME)
            model_full_error = full_error(data["Ne_true"], data["Ne_pred"], max_time=MAX_TIME)

            # Расчет метрик классификации
            
            # Получаем accuracy из cls_report или вычисляем из confusion_matrix
            if "accuracy" in data["cls_report"]:
                model_accuracy = data["cls_report"]["accuracy"]
            elif "confusion_matrix" in data:
                # Вычисляем accuracy из confusion_matrix
                cm = np.array(data["confusion_matrix"])
                model_accuracy = cm.trace() / cm.sum() if cm.sum() > 0 else 0.0
            else:
                # Если нет ни того, ни другого, используем значение по умолчанию
                model_accuracy = 0.0
                print("Предупреждение: accuracy не найден для {}/{}/{}".format(key, model, dem))
            
            precision_3 =  data.get("precision+-1(@3)", 0.0)
            precision_5 =  data.get("precision+-2(@5)", 0.0)

            # Проверяем наличие weighted avg в cls_report
            if "weighted avg" in data["cls_report"]:
                model_Precision = data["cls_report"]["weighted avg"]["precision"]
                model_Recall = data["cls_report"]["weighted avg"]["recall"]
                model_F1  = data["cls_report"]["weighted avg"]["f1-score"]
            else:
                # Если нет weighted avg, используем значения по умолчанию
                model_Precision = 0.0
                model_Recall = 0.0
                model_F1 = 0.0
                print("Предупреждение: weighted avg не найден в cls_report для {}/{}/{}".format(key, model, dem))
            
            # Сохранение метрик модели как строка в таблице

            res_dic["{} - {}".format(model, dem)] = {
                    "model": model,
                    "dems": dem,
                    "accuracy": model_accuracy,
                    "precision": model_Precision,
                    "precision+-1(@3)": precision_3,
                    "precision+-2(@5)": precision_5,
                    "recall": model_Recall,
                    "f1-score": model_F1,
                    "FMSE": model_FMSE,
                    "RMSE": model_RMSE,
                    "FMAE": model_FMAE,
                    "time": model_time,
                    "full-error": model_full_error,
                    "FMAE2": model_FMAE2,
                    
            }

            # Для просчета средних метрик модели

            integral_time = integral_time + model_time
            integral_FMSE = integral_FMSE + model_FMSE
            integral_RMSE = integral_RMSE + model_RMSE
            integral_FMAE = integral_FMAE + model_FMAE
            integral_FMAE2 = integral_FMAE2 + model_FMAE2
            integral_full_error = integral_full_error + model_full_error

            integral_accuracy = integral_accuracy + model_accuracy
            iprecision_3 = precision_3 + iprecision_3
            iprecision_5 = precision_5 + iprecision_5

            integral_Precision = integral_Precision + model_Precision
            integral_Recall = integral_Recall + model_Recall
            integral_F1 = integral_F1  + model_F1

        
        # Просчет средних метрик модели
        # Используем количество обработанных демографий
        num_processed_dems = len(model_dems) if use_auto_discovery else len(dems)
        
        if num_processed_dems == 0:
            print("Предупреждение: нет обработанных демографий для {}, пропускаем средние метрики".format(model))
            continue

        integral_time = integral_time/num_processed_dems
        integral_FMSE = integral_FMSE/num_processed_dems
        integral_RMSE = integral_RMSE/num_processed_dems
        integral_FMAE = integral_FMAE/num_processed_dems
        integral_FMAE2 = integral_FMAE2/num_processed_dems
        integral_full_error = integral_full_error/num_processed_dems

        integral_accuracy = integral_accuracy/num_processed_dems
        integral_Precision = integral_Precision/num_processed_dems
        integral_Recall = integral_Recall/num_processed_dems
        integral_F1 = integral_F1/num_processed_dems

        iprecision_3 = iprecision_3 / num_processed_dems
        iprecision_5 = iprecision_5 / num_processed_dems


        # Сохранение средних метрик модели как строка в таблице
        

        res_dic["{} - avg".format(model)] = {
                "model": model,
                "dems": "avg",
                "accuracy": integral_accuracy,
                "precision": integral_Precision,
                "precision+-1(@3)": iprecision_3,
                "precision+-2(@5)": iprecision_5,
                "recall": integral_Recall,
                "f1-score": integral_F1,
                "FMSE": integral_FMSE,
                "RMSE": integral_RMSE,
                "FMAE": integral_FMAE,
                "time": integral_time,
                "full-error": integral_full_error,
                "FMAE2": integral_FMAE2,
        }


# Сохранение таблицы
output_dir = Path(__file__).parent / 'new_res_30'
output_dir.mkdir(parents=True, exist_ok=True)

output_path_json = output_dir / '{}.json'.format(output_file)
output_path_csv = output_dir / '{}.csv'.format(output_file)
output_path_xlsx = output_dir / '{}.xlsx'.format(output_file)

with open(output_path_json, 'w', encoding='utf-8') as file:
    json.dump(res_dic, file, indent=2, ensure_ascii=False)

print("\nРезультаты сохранены в {}".format(output_path_json))

# Создаем DataFrame из словаря
if res_dic:
    df = pd.DataFrame(res_dic)

    print("\nПервые строки результата:")
    print(df.T.head())

    # Сохраняем DataFrame в Excel-файл или CSV, если openpyxl недоступен
    try:
        df.T.to_excel(output_path_xlsx)
        print("Результаты сохранены в {}".format(output_path_xlsx))
    except ModuleNotFoundError as e:
        if 'openpyxl' in str(e):
            print("Предупреждение: модуль openpyxl не установлен. Сохраняем в CSV вместо Excel.")
            df.T.to_csv(output_path_csv, encoding='utf-8')
            print("Результаты сохранены в {}".format(output_path_csv))
        else:
            raise
else:
    print("Предупреждение: нет данных для сохранения")