#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ метрик FMSA/FMSE для каждой модели как функция от правого граничного интервала.
Начиная с 10-го интервала, вычисляем метрики на первых N точках (10, 11, 12, ...).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util
import os
import sys

# Добавляем путь для импорта plot_results
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Импортируем limits из data_gen_np
spec = importlib.util.spec_from_file_location(
    "data_gen_np", 
    Path(__file__).parent.parent / "deepgene" / "data" / "data_gen_np.py"
)
dgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dgn)
limits = dgn.limits

# Импортируем plot_results для работы с PSMC
try:
    from psmc_checker import plot_results
except ImportError:
    print("Предупреждение: не удалось импортировать plot_results. PSMC данные не будут использованы.")
    plot_results = None

# Функции для расчета метрик (из all_in_one.py)
def fill_nan_with_previous(data):
    """Заменяет все NaN и Infinity значениями предыдущих элементов."""
    arr = np.array(data, dtype=float)
    
    if len(arr) == 0:
        return arr
    
    if np.isnan(arr[0]) or np.isinf(arr[0]):
        first_valid_idx = None
        for i in range(len(arr)):
            if not (np.isnan(arr[i]) or np.isinf(arr[i])):
                first_valid_idx = i
                break
        
        if first_valid_idx is not None:
            arr[:first_valid_idx] = arr[first_valid_idx]
        else:
            return np.full_like(arr, np.nan)
    
    for i in range(1, len(arr)):
        if np.isnan(arr[i]) or np.isinf(arr[i]):
            arr[i] = arr[i-1]
    
    return arr

def trim_arrays(a, b):
    """Обрезает массивы до минимальной длины без nan/inf с конца."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    def valid_length(x):
        mask = ~(np.isnan(x) | np.isinf(x))
        if not np.any(mask):
            return 0
        return np.argmax(mask[::-1])
    
    cut_a = len(a) - valid_length(a)
    cut_b = len(b) - valid_length(b)
    cut = min(cut_a, cut_b)
    return a[:cut], b[:cut]

def fmse(y_true, y_pred):
    """Вычисляет FMSE (Fractional Mean Squared Error)."""
    y_true = fill_nan_with_previous(np.array(y_true))
    y_pred = fill_nan_with_previous(np.array(y_pred))
    y_true, y_pred = trim_arrays(y_true, y_pred)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    
    relative_errors = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return np.mean(relative_errors ** 2)

def fmae(y_true, y_pred):
    """Вычисляет FMAE (Fractional Mean Absolute Error) - это FMSA."""
    y_true = fill_nan_with_previous(np.array(y_true))
    y_pred = fill_nan_with_previous(np.array(y_pred))
    y_true, y_pred = trim_arrays(y_true, y_pred)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return np.nan
    
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    
    relative_errors = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return np.mean(np.abs(relative_errors))

def find_all_result_files(base_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Находит все result.json файлы.
    Возвращает список (model_type, model_name, dem_name, file_path).
    """
    results = []
    base_path_obj = Path(base_path)
    
    if not base_path_obj.exists():
        return results
    
    model_types = ["gru", "lstm", "reformer"]
    
    for model_type in model_types:
        model_type_dir = base_path_obj / model_type
        if not model_type_dir.exists():
            continue
        
        for model_dir in model_type_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            for dem_dir in model_dir.iterdir():
                if not dem_dir.is_dir():
                    continue
                
                result_file = dem_dir / "result.json"
                if result_file.exists():
                    results.append((
                        model_type,
                        model_dir.name,
                        dem_dir.name,
                        str(result_file)
                    ))
    
    return results

def _ne_at_times(*, limits_x: list[float], ne_values: np.ndarray, times: list[float]) -> np.ndarray:
    """
    Интерполирует Ne значения на заданные временные точки.
    Использует step-функцию (пост-интерполяция).
    """
    limits_x = np.asarray(limits_x, dtype=float)
    ne_values = np.asarray(ne_values, dtype=float)
    times = np.asarray(times, dtype=float)
    
    # Для step-функции: находим индекс интервала для каждой временной точки
    # Используем searchsorted для нахождения правой границы
    idx = np.searchsorted(limits_x[1:], times, side='right')
    idx = np.clip(idx, 0, len(ne_values) - 1)
    return ne_values[idx]

def calculate_metrics_for_intervals(ne_true: List[float], ne_pred: List[float], 
                                     start_interval: int = 10, max_intervals: int = None) -> Dict[int, Dict[str, float]]:
    """
    Вычисляет метрики для разных правых граничных интервалов.
    
    Args:
        ne_true: Истинные значения Ne
        ne_pred: Предсказанные значения Ne
        start_interval: Начальный интервал (по умолчанию 10)
        max_intervals: Максимальное количество интервалов для анализа (None = без ограничений)
    
    Returns:
        Словарь {right_boundary: {"FMSE": ..., "FMAE": ...}}
    """
    ne_true = np.array(ne_true, dtype=float)
    ne_pred = np.array(ne_pred, dtype=float)
    
    # Определяем максимальный интервал
    max_interval = min(len(ne_true), len(ne_pred))
    if max_intervals is not None:
        max_interval = min(max_interval, max_intervals)
    
    results = {}
    
    for right_boundary in range(start_interval, max_interval + 1):
        # Берем первые right_boundary интервалов
        ne_true_subset = ne_true[:right_boundary].tolist()
        ne_pred_subset = ne_pred[:right_boundary].tolist()
        
        # Вычисляем метрики
        fmse_val = fmse(ne_true_subset, ne_pred_subset)
        fmae_val = fmae(ne_true_subset, ne_pred_subset)
        
        results[right_boundary] = {
            "FMSE": fmse_val,
            "FMAE": fmae_val
        }
    
    return results

def main():
    # Путь к результатам
    results_base_path = Path(__file__).parent.parent / "test_all_v1" / "_results"
    
    if not results_base_path.exists():
        print(f"Директория результатов не найдена: {results_base_path}")
        return
    
    # Находим все файлы результатов
    result_files = find_all_result_files(str(results_base_path))
    print(f"Найдено {len(result_files)} файлов результатов")
    
    # Словарь для хранения результатов: {model_key: {right_boundary: {FMSE, FMAE}}}
    all_results = {}
    
    # Обрабатываем каждый файл
    for model_type, model_name, dem_name, file_path in result_files:
        model_key = f"{model_type}_{model_name}_{dem_name}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ne_true = data.get("Ne_true", [])
            ne_pred = data.get("Ne_pred", [])
            
            if not ne_true or not ne_pred:
                print(f"Пропуск {model_key}: нет данных Ne")
                continue
            
            # Вычисляем метрики для разных интервалов
            metrics = calculate_metrics_for_intervals(ne_true, ne_pred, start_interval=10, max_intervals=30)
            all_results[model_key] = {
                "model_type": model_type,
                "model_name": model_name,
                "dem_name": dem_name,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")
            continue
    
    # Определяем директорию для сохранения результатов
    output_dir = Path(__file__).parent
    
    # Вычисляем метрики PSMC если доступны
    psmc_results = None
    if plot_results is not None:
        psmc_results = calculate_psmc_metrics(output_dir, limits)
    
    # Сохраняем результаты в JSON (только для первых 30 интервалов)
    output_data = {
        "models": all_results,
        "psmc": psmc_results if psmc_results is not None else {}
    }
    with open(output_dir / "borders_analysis_results_30_intervals.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены в {output_dir / 'borders_analysis_results_30_intervals.json'}")
    
    # Создаем таблицы и графики
    create_latex_table(all_results, output_dir)
    create_plots(all_results, output_dir, limits, psmc_results)
    
    print("Анализ завершен!")

def create_latex_table(all_results: Dict, output_dir: Path):
    """Создает LaTeX таблицу с результатами."""
    
    # Собираем все уникальные правые границы
    all_boundaries = set()
    for model_data in all_results.values():
        all_boundaries.update(model_data["metrics"].keys())
    all_boundaries = sorted(all_boundaries)
    
    # Группируем по моделям (без демографии) для усреднения
    model_groups = {}
    for model_key, model_data in all_results.items():
        model_type = model_data["model_type"]
        model_name = model_data["model_name"]
        group_key = f"{model_type}_{model_name}"
        
        if group_key not in model_groups:
            model_groups[group_key] = {
                "model_type": model_type,
                "model_name": model_name,
                "metrics_by_boundary": {}
            }
        
        # Добавляем метрики для каждой границы
        for boundary, metrics in model_data["metrics"].items():
            if boundary not in model_groups[group_key]["metrics_by_boundary"]:
                model_groups[group_key]["metrics_by_boundary"][boundary] = {
                    "FMSE": [],
                    "FMAE": []
                }
            model_groups[group_key]["metrics_by_boundary"][boundary]["FMSE"].append(metrics["FMSE"])
            model_groups[group_key]["metrics_by_boundary"][boundary]["FMAE"].append(metrics["FMAE"])
    
    # Вычисляем средние значения
    for group_key, group_data in model_groups.items():
        for boundary in all_boundaries:
            if boundary in group_data["metrics_by_boundary"]:
                fmse_vals = group_data["metrics_by_boundary"][boundary]["FMSE"]
                fmae_vals = group_data["metrics_by_boundary"][boundary]["FMAE"]
                group_data["metrics_by_boundary"][boundary]["FMSE_avg"] = np.nanmean(fmse_vals)
                group_data["metrics_by_boundary"][boundary]["FMAE_avg"] = np.nanmean(fmae_vals)
    
    # Создаем LaTeX таблицу
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Метрики FMSE и FMAE в зависимости от правого граничного интервала}")
    latex_lines.append("\\label{tab:borders_analysis}")
    
    # Определяем количество столбцов
    num_cols = 1 + len(all_boundaries) * 2  # модель + (FMSE, FMAE) для каждой границы
    latex_lines.append(f"\\begin{{tabular}}{{{'l' + 'c' * (num_cols - 1)}}}")
    latex_lines.append("\\hline")
    
    # Заголовок
    header = "Модель"
    for boundary in all_boundaries:
        header += f" & \\multicolumn{{2}}{{c}}{{Интервал {boundary}}}"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\cline{2-" + str(num_cols) + "}")
    
    subheader = ""
    for _ in all_boundaries:
        subheader += " & FMSE & FMAE"
    subheader += " \\\\"
    latex_lines.append(subheader)
    latex_lines.append("\\hline")
    
    # Данные
    for group_key in sorted(model_groups.keys()):
        group_data = model_groups[group_key]
        model_display = f"{group_data['model_type']}_{group_data['model_name']}"
        row = model_display.replace("_", "\\_")
        
        for boundary in all_boundaries:
            if boundary in group_data["metrics_by_boundary"]:
                fmse_val = group_data["metrics_by_boundary"][boundary]["FMSE_avg"]
                fmae_val = group_data["metrics_by_boundary"][boundary]["FMAE_avg"]
                
                if np.isnan(fmse_val):
                    row += " & -- & --"
                else:
                    row += f" & {fmse_val:.4f} & {fmae_val:.4f}"
            else:
                row += " & -- & --"
        
        row += " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Сохраняем таблицу
    with open(output_dir / "borders_analysis_table.tex", 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX таблица сохранена в {output_dir / 'borders_analysis_table.tex'}")

def calculate_psmc_metrics(output_dir: Path, limits: List[float]) -> Optional[Dict]:
    """Вычисляет метрики PSMC vs TRUE для разных интервалов."""
    if plot_results is None:
        return None
    
    generation_time = 25.0
    bin_size = 100
    mutation_rate = getattr(dgn, "MU_HUMAN", 1.25e-8)
    
    # Маппинг имен демографий
    dem_mapping = {
        "const": "get_const_demographcs",
        "hl": "get_humanlike",
        "zigzag": "get_zigzag"
    }
    
    psmc_base_path = REPO_ROOT / "psmc_checker" / "runs"
    if not psmc_base_path.exists():
        print(f"Директория PSMC не найдена: {psmc_base_path}")
        return None
    
    psmc_results = {}
    limits_x = [float(t) * generation_time for t in limits]
    
    for dem_name, dem_fn_name in dem_mapping.items():
        psmc_file = psmc_base_path / dem_name / f"exp1_{dem_name}.psmc"
        if not psmc_file.exists():
            print(f"PSMC файл не найден: {psmc_file}")
            continue
        
        try:
            # Загружаем PSMC данные
            psmc_times, psmc_sizes, n0 = plot_results.psmc_parse_times_sizes(
                psmc_file,
                bin_size=bin_size,
                mutation_rate=mutation_rate,
                generation_time=generation_time,
                N0_override=None,
                time_mult=2.0,
                size_mult=1.0,
            )
            
            # Создаем TRUE демографию
            dem = plot_results.build_demography(
                dem_fn_name,
                initial_size=10000,
                random_seed=42
            )
            true_ne_full = np.asarray(plot_results.demography_to_Ne(dem, limits), dtype=float)
            
            # Интерполируем TRUE на временные точки PSMC
            # true_ne_full имеет длину len(limits), нужно создать полный массив для step-функции
            true_ne_full_array = np.asarray(true_ne_full, dtype=float).reshape(-1)
            true_at_psmc = _ne_at_times(
                limits_x=limits_x,
                ne_values=true_ne_full_array,
                times=psmc_times
            )
            
            # Вычисляем метрики для разных интервалов (только первые 30)
            metrics_by_boundary = {}
            max_boundary = min(30, len(limits))
            for boundary in range(10, max_boundary + 1):
                # Временной лимит в годах для этого интервала
                time_limit_years = limits_x[boundary] if boundary < len(limits_x) else limits_x[-1]
                
                # Фильтруем точки PSMC, которые попадают в этот интервал
                mask = np.asarray(psmc_times) <= time_limit_years
                
                if not np.any(mask):
                    continue
                
                # Вычисляем метрики
                true_subset = true_at_psmc[mask]
                psmc_subset = np.asarray(psmc_sizes, dtype=float)[mask]
                
                fmse_val = fmse(true_subset.tolist(), psmc_subset.tolist())
                fmae_val = fmae(true_subset.tolist(), psmc_subset.tolist())
                
                metrics_by_boundary[boundary] = {
                    "FMSE": fmse_val,
                    "FMAE": fmae_val
                }
            
            psmc_results[dem_name] = metrics_by_boundary
            
        except Exception as e:
            print(f"Ошибка при обработке PSMC для {dem_name}: {e}")
            continue
    
    # Усредняем по демографиям
    if psmc_results:
        all_boundaries = set()
        for dem_metrics in psmc_results.values():
            all_boundaries.update(dem_metrics.keys())
        
        averaged_psmc = {}
        for boundary in sorted(all_boundaries):
            fmse_vals = []
            fmae_vals = []
            for dem_metrics in psmc_results.values():
                if boundary in dem_metrics:
                    if not np.isnan(dem_metrics[boundary]["FMSE"]):
                        fmse_vals.append(dem_metrics[boundary]["FMSE"])
                    if not np.isnan(dem_metrics[boundary]["FMAE"]):
                        fmae_vals.append(dem_metrics[boundary]["FMAE"])
            
            if fmse_vals or fmae_vals:
                averaged_psmc[boundary] = {
                    "FMSE": np.nanmean(fmse_vals) if fmse_vals else np.nan,
                    "FMAE": np.nanmean(fmae_vals) if fmae_vals else np.nan
                }
        
        return averaged_psmc
    
    return None

def create_plots(all_results: Dict, output_dir: Path, limits: List[float], psmc_results: Optional[Dict] = None):
    """Создает графики для анализа."""
    
    # Группируем по типам моделей для усреднения
    model_type_groups = {}
    for model_key, model_data in all_results.items():
        model_type = model_data["model_type"]
        
        if model_type not in model_type_groups:
            model_type_groups[model_type] = {}
        
        for boundary, metrics in model_data["metrics"].items():
            if boundary not in model_type_groups[model_type]:
                model_type_groups[model_type][boundary] = {
                    "FMSE": [],
                    "FMAE": []
                }
            model_type_groups[model_type][boundary]["FMSE"].append(metrics["FMSE"])
            model_type_groups[model_type][boundary]["FMAE"].append(metrics["FMAE"])
    
    # График 1: Средняя ошибка от порядкового номера интервала
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_type in ["gru", "lstm", "reformer"]:
        if model_type not in model_type_groups:
            continue
        
        boundaries = sorted(model_type_groups[model_type].keys())
        fmse_means = []
        fmae_means = []
        
        for boundary in boundaries:
            fmse_vals = model_type_groups[model_type][boundary]["FMSE"]
            fmae_vals = model_type_groups[model_type][boundary]["FMAE"]
            fmse_means.append(np.nanmean(fmse_vals))
            fmae_means.append(np.nanmean(fmae_vals))
        
        axes[0].plot(boundaries, fmse_means, marker='o', label=f"{model_type.upper()} FMSE", linewidth=2)
        axes[1].plot(boundaries, fmae_means, marker='s', label=f"{model_type.upper()} FMAE", linewidth=2)
    
    axes[0].set_xlabel("Порядковый номер правого граничного интервала", fontsize=12)
    axes[0].set_ylabel("FMSE", fontsize=12)
    axes[0].set_title("FMSE в зависимости от порядкового номера интервала", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Порядковый номер правого граничного интервала", fontsize=12)
    axes[1].set_ylabel("FMAE", fontsize=12)
    axes[1].set_title("FMAE в зависимости от порядкового номера интервала", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "borders_analysis_plot_interval_number.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # График 2: Средняя ошибка от временных лимитов
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_type in ["gru", "lstm", "reformer"]:
        if model_type not in model_type_groups:
            continue
        
        boundaries = sorted(model_type_groups[model_type].keys())
        fmse_means = []
        fmae_means = []
        time_limits = []
        
        for boundary in boundaries:
            # Используем правую границу интервала (boundary-й индекс в limits)
            if boundary < len(limits):
                time_limit = limits[boundary]
            else:
                time_limit = limits[-1] if limits else 0
            
            fmse_vals = model_type_groups[model_type][boundary]["FMSE"]
            fmae_vals = model_type_groups[model_type][boundary]["FMAE"]
            fmse_means.append(np.nanmean(fmse_vals))
            fmae_means.append(np.nanmean(fmae_vals))
            time_limits.append(time_limit)
        
        axes[0].plot(time_limits, fmse_means, marker='o', label=f"{model_type.upper()} FMSE", linewidth=2)
        axes[1].plot(time_limits, fmae_means, marker='s', label=f"{model_type.upper()} FMAE", linewidth=2)
    
    axes[0].set_xlabel("Временной лимит (поколения)", fontsize=12)
    axes[0].set_ylabel("FMSE", fontsize=12)
    axes[0].set_title("FMSE в зависимости от временного лимита", fontsize=14)
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Временной лимит (поколения)", fontsize=12)
    axes[1].set_ylabel("FMAE", fontsize=12)
    axes[1].set_title("FMAE в зависимости от временного лимита", fontsize=14)
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "borders_analysis_plot_time_limits.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # График 3: Средняя ошибка от временных лимитов в годах
    generation_time = 25.0  # Стандартное время генерации для человека (лет)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_type in ["gru", "lstm", "reformer"]:
        if model_type not in model_type_groups:
            continue
        
        boundaries = sorted(model_type_groups[model_type].keys())
        fmse_means = []
        fmae_means = []
        time_limits_years = []
        
        for boundary in boundaries:
            # Используем правую границу интервала (boundary-й индекс в limits)
            if boundary < len(limits):
                time_limit_generations = limits[boundary]
            else:
                time_limit_generations = limits[-1] if limits else 0
            
            # Конвертируем в годы
            time_limit_years = time_limit_generations * generation_time
            
            fmse_vals = model_type_groups[model_type][boundary]["FMSE"]
            fmae_vals = model_type_groups[model_type][boundary]["FMAE"]
            fmse_means.append(np.nanmean(fmse_vals))
            fmae_means.append(np.nanmean(fmae_vals))
            time_limits_years.append(time_limit_years)
        
        axes[0].plot(time_limits_years, fmse_means, marker='o', label=f"{model_type.upper()} FMSE", linewidth=2)
        axes[1].plot(time_limits_years, fmae_means, marker='s', label=f"{model_type.upper()} FMAE", linewidth=2)
    
    axes[0].set_xlabel("Временной лимит (годы)", fontsize=12)
    axes[0].set_ylabel("FMSE", fontsize=12)
    axes[0].set_title("FMSE в зависимости от временного лимита (годы)", fontsize=14)
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Временной лимит (годы)", fontsize=12)
    axes[1].set_ylabel("FMAE", fontsize=12)
    axes[1].set_title("FMAE в зависимости от временного лимита (годы)", fontsize=14)
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "borders_analysis_plot_time_limits_years.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # График 4: С PSMC, ось OX в log, ось OY обычная
    if psmc_results is not None:
        generation_time = 25.0
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for model_type in ["gru", "lstm", "reformer"]:
            if model_type not in model_type_groups:
                continue
            
            boundaries = sorted(model_type_groups[model_type].keys())
            fmse_means = []
            fmae_means = []
            time_limits_years = []
            
            for boundary in boundaries:
                if boundary < len(limits):
                    time_limit_years = limits[boundary] * generation_time
                else:
                    time_limit_years = limits[-1] * generation_time if limits else 0
                
                fmse_vals = model_type_groups[model_type][boundary]["FMSE"]
                fmae_vals = model_type_groups[model_type][boundary]["FMAE"]
                fmse_means.append(np.nanmean(fmse_vals))
                fmae_means.append(np.nanmean(fmae_vals))
                time_limits_years.append(time_limit_years)
            
            axes[0].plot(time_limits_years, fmse_means, marker='o', label=f"{model_type.upper()} FMSE", linewidth=2)
            axes[1].plot(time_limits_years, fmae_means, marker='s', label=f"{model_type.upper()} FMAE", linewidth=2)
        
        # Добавляем PSMC
        if psmc_results:
            psmc_boundaries = sorted(psmc_results.keys())
            psmc_fmse = []
            psmc_fmae = []
            psmc_times = []
            
            for boundary in psmc_boundaries:
                if boundary < len(limits):
                    time_limit_years = limits[boundary] * generation_time
                else:
                    time_limit_years = limits[-1] * generation_time if limits else 0
                
                psmc_fmse.append(psmc_results[boundary]["FMSE"])
                psmc_fmae.append(psmc_results[boundary]["FMAE"])
                psmc_times.append(time_limit_years)
            
            axes[0].plot(psmc_times, psmc_fmse, marker='^', label="PSMC FMSE", linewidth=2, linestyle='--', color='red')
            axes[1].plot(psmc_times, psmc_fmae, marker='v', label="PSMC FMAE", linewidth=2, linestyle='--', color='red')
        
        axes[0].set_xlabel("Временной лимит (годы)", fontsize=12)
        axes[0].set_ylabel("FMSE", fontsize=12)
        axes[0].set_title("FMSE в зависимости от временного лимита (годы) с PSMC", fontsize=14)
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel("Временной лимит (годы)", fontsize=12)
        axes[1].set_ylabel("FMAE", fontsize=12)
        axes[1].set_title("FMAE в зависимости от временного лимита (годы) с PSMC", fontsize=14)
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "borders_analysis_plot_time_limits_years_with_psmc.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # График 5: С PSMC, обе оси в log
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for model_type in ["gru", "lstm", "reformer"]:
            if model_type not in model_type_groups:
                continue
            
            boundaries = sorted(model_type_groups[model_type].keys())
            fmse_means = []
            fmae_means = []
            time_limits_years = []
            
            for boundary in boundaries:
                if boundary < len(limits):
                    time_limit_years = limits[boundary] * generation_time
                else:
                    time_limit_years = limits[-1] * generation_time if limits else 0
                
                fmse_vals = model_type_groups[model_type][boundary]["FMSE"]
                fmae_vals = model_type_groups[model_type][boundary]["FMAE"]
                fmse_means.append(np.nanmean(fmse_vals))
                fmae_means.append(np.nanmean(fmae_vals))
                time_limits_years.append(time_limit_years)
            
            axes[0].plot(time_limits_years, fmse_means, marker='o', label=f"{model_type.upper()} FMSE", linewidth=2)
            axes[1].plot(time_limits_years, fmae_means, marker='s', label=f"{model_type.upper()} FMAE", linewidth=2)
        
        # Добавляем PSMC
        if psmc_results:
            psmc_boundaries = sorted(psmc_results.keys())
            psmc_fmse = []
            psmc_fmae = []
            psmc_times = []
            
            for boundary in psmc_boundaries:
                if boundary < len(limits):
                    time_limit_years = limits[boundary] * generation_time
                else:
                    time_limit_years = limits[-1] * generation_time if limits else 0
                
                psmc_fmse.append(psmc_results[boundary]["FMSE"])
                psmc_fmae.append(psmc_results[boundary]["FMAE"])
                psmc_times.append(time_limit_years)
            
            axes[0].plot(psmc_times, psmc_fmse, marker='^', label="PSMC FMSE", linewidth=2, linestyle='--', color='red')
            axes[1].plot(psmc_times, psmc_fmae, marker='v', label="PSMC FMAE", linewidth=2, linestyle='--', color='red')
        
        axes[0].set_xlabel("Временной лимит (годы)", fontsize=12)
        axes[0].set_ylabel("FMSE", fontsize=12)
        axes[0].set_title("FMSE в зависимости от временного лимита (годы, log-log) с PSMC", fontsize=14)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel("Временной лимит (годы)", fontsize=12)
        axes[1].set_ylabel("FMAE", fontsize=12)
        axes[1].set_title("FMAE в зависимости от временного лимита (годы, log-log) с PSMC", fontsize=14)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "borders_analysis_plot_time_limits_years_with_psmc_loglog.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Графики сохранены в {output_dir}")

if __name__ == "__main__":
    main()
