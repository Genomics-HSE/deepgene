#!/usr/bin/env python3
"""
Calculate metrics between PSMC and NN Ne curves.

Computes relative error metrics (FMSE, RMSE, FMAE) comparing NN predictions
against PSMC results (treating PSMC as reference).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Load plot_results module for helper functions
REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels: utiles/1000 геномов -> utiles -> repo root
sys.path.insert(0, str(REPO_ROOT))

from psmc_checker import plot_results  # noqa: E402


def _load_deepgene_data_gen_np():
    """Load deepgene/data/data_gen_np.py to get limits."""
    module_path = REPO_ROOT / "deepgene" / "data" / "data_gen_np.py"
    if not module_path.exists():
        raise ImportError(f"Cannot locate deepgene/data/data_gen_np.py at {module_path}")
    spec = importlib.util.spec_from_file_location("deepgene_data_gen_np_standalone", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def from_probability_to_N(priors: list[float], limits: list[float]) -> list[float]:
    """Convert probability mass per interval to Ne (PSMC-like)."""
    priors = list(priors)
    tau = np.array(limits[1:], dtype=float) - np.array(limits[:-1], dtype=float)
    if len(priors) != len(tau):
        raise ValueError(
            f"Expected priors length {len(tau)} (len(limits)-1), got {len(priors)}"
        )

    l = 1.0
    N: list[float] = []
    for j in range(len(tau)):
        pj = float(priors[j])
        eps = 1e-12
        val = max(min(1.0 - pj / l, 1.0 - eps), eps)
        Nj = float(np.round(tau[j] / (-2.0 * np.log(val))))
        N.append(Nj)
        l *= float(np.exp(-tau[j] / (2.0 * Nj)))
    return N


def _relative_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, int]:
    """Compute relative error metrics: FMSE, RMSE, FMAE."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}")

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0.0)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan"), 0

    err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    fmse = float(np.mean(err**2))
    fmae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(fmse)) if math.isfinite(fmse) else float("nan")
    return fmse, rmse, fmae, int(np.sum(mask))


def _ne_at_times(*, limits_x: list[float], ne_values: np.ndarray, times: list[float]) -> np.ndarray:
    """Interpolate Ne values at given time points."""
    x = np.asarray(limits_x, dtype=float)
    y = np.asarray(ne_values, dtype=float).reshape(-1)
    t = np.asarray(times, dtype=float)
    idx = np.searchsorted(x, t, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calculate metrics between PSMC and NN Ne curves.")
    ap.add_argument("--psmc", required=True, help="Path to .psmc file")
    ap.add_argument("--nn-priors-json", required=True, help="Path to nn_priors.json from inference")
    ap.add_argument("--out", required=True, help="Output markdown file path")
    ap.add_argument("--bin-size", type=int, default=100, help="PSMC bin size (default: 100)")
    ap.add_argument("--generation-time", type=float, default=25.0, help="Generation time in years (default: 25.0)")
    ap.add_argument("--mutation-rate", type=float, default=2.5e-8, help="Mutation rate per base per generation (default: 2.5e-8, matching plot_results.py)")
    ap.add_argument("--psmc-time-mult", type=float, default=2.0, help="PSMC time multiplier (default: 2.0)")
    ap.add_argument("--psmc-size-mult", type=float, default=1.0, help="PSMC size multiplier (default: 1.0)")
    ap.add_argument("--nn-key", default="priors", help="Key in JSON containing priors (default: 'priors')")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    
    # Load deepgene limits
    dgn = _load_deepgene_data_gen_np()
    limits_full = list(getattr(dgn, "limits"))
    
    # Обрезаем на 120000 поколений назад (это соответствует limits[30] = 120000)
    # Используем первые 30 интервалов NN (limits[0:31] дает 30 интервалов)
    CUTOFF_GENERATIONS = 120000  # Обрезка на 120000 поколений назад
    # Находим индекс, соответствующий 120000 поколениям
    cutoff_idx = None
    for i, limit in enumerate(limits_full):
        if limit >= CUTOFF_GENERATIONS:
            cutoff_idx = i
            break
    if cutoff_idx is None:
        cutoff_idx = len(limits_full) - 1
    
    # Ограничиваем до первых 30 интервалов (до 120000 поколений)
    limits = limits_full[:31]  # 31 элемент для 30 интервалов (limits[30] = 120000)
    limits_x = [float(t) * float(args.generation_time) for t in limits]
    
    # Максимальное время в годах для обрезки (120000 поколений * время генерации)
    max_time_years = CUTOFF_GENERATIONS * float(args.generation_time)
    
    # Load PSMC data
    # Используем параметры из plot_results.py:
    # - MUTATION_RATE = 2.5e-8 (вместо MU_HUMAN = 1.25e-8)
    # - BIN_SIZE = 100
    # - GENERATION_TIME = 25
    # - time_mult = 2.0 (формула: times = GENERATION_TIME * 2 * N0 * time_window)
    # - size_mult = 1.0 (формула: sizes = N0 * lambda)
    psmc_path = Path(args.psmc)
    psmc_times, psmc_sizes, n0 = plot_results.psmc_parse_times_sizes(
        psmc_path,
        bin_size=args.bin_size,
        mutation_rate=args.mutation_rate,  # Используем значение из аргументов (2.5e-8 по умолчанию)
        generation_time=args.generation_time,
        N0_override=None,
        time_mult=args.psmc_time_mult,
        size_mult=args.psmc_size_mult,
    )
    
    # Load NN priors and convert to Ne
    priors_path = Path(args.nn_priors_json)
    data = json.loads(priors_path.read_text(encoding="utf-8"))
    
    if args.nn_key not in data:
        raise KeyError(f"Key '{args.nn_key}' not found in {priors_path}")
    priors = data[args.nn_key]
    if not isinstance(priors, list):
        raise ValueError(f"Key '{args.nn_key}' is not a list in {priors_path}")
    priors = [float(x) for x in priors]
    
    # Обрезаем priors до первых 30 элементов для первых 30 интервалов
    priors_30 = priors[:30]
    # Convert priors to Ne (используем обрезанные limits и priors)
    nn_ne = from_probability_to_N(priors_30, limits)
    
    # Interpolate NN Ne at PSMC time points (только для первых 30 интервалов)
    nn_full = np.asarray(list(nn_ne) + [nn_ne[-1]], dtype=float)
    
    # Фильтруем PSMC точки, которые попадают до 120000 поколений (в годах)
    psmc_mask = np.asarray(psmc_times) <= max_time_years
    psmc_times_filtered = [t for i, t in enumerate(psmc_times) if psmc_mask[i]]
    psmc_sizes_filtered = [s for i, s in enumerate(psmc_sizes) if psmc_mask[i]]
    
    nn_at_psmc = _ne_at_times(limits_x=limits_x, ne_values=nn_full, times=psmc_times_filtered)
    
    # Compute metrics (PSMC as reference, NN as prediction) - только для первых 30 интервалов
    fmse, rmse, fmae, n_points = _relative_metrics(
        np.asarray(psmc_sizes_filtered, dtype=float),
        nn_at_psmc
    )
    
    # Get additional info
    n_windows = data.get("n_windows", "N/A")
    window_len = data.get("window_len", "N/A")
    checkpoint = data.get("checkpoint", "N/A")
    
    # Write markdown report
    out_dir = Path("new_res_30") / "psmc_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(args.out).name
    
    md_content = f"""# Метрики сравнения PSMC и NN для NA20512

## Параметры анализа

- **PSMC файл**: `{psmc_path.name}`
- **NN checkpoint**: `{Path(checkpoint).name if checkpoint != "N/A" else checkpoint}`
- **Размер бина**: {args.bin_size} bp
- **Время генерации**: {args.generation_time} лет
- **Мутационная частота**: {args.mutation_rate:.2e} (соответствует plot_results.py)
- **PSMC N0**: {n0:.2f}
- **Количество окон NN**: {n_windows}
- **Длина окна NN**: {window_len} бинов

## Метрики относительной ошибки

Метрики вычислены на временных точках PSMC, где PSMC используется как референс (ground truth), а NN - как предсказание.

### Формулы метрик

- **FMSE** (Fractional Mean Squared Error): среднее квадратичное относительной ошибки
  - `FMSE = mean((PSMC - NN) / PSMC)²`
  
- **RMSE** (Root Mean Squared Error): корень из FMSE
  - `RMSE = sqrt(FMSE)`
  
- **FMAE** (Fractional Mean Absolute Error): среднее абсолютное значение относительной ошибки
  - `FMAE = mean(|(PSMC - NN) / PSMC|)`

### Результаты

| Метрика | Значение | Описание |
|---------|----------|----------|
| **FMSE** | {fmse:.6f} | Среднее квадратичное относительной ошибки |
| **RMSE** | {rmse:.6f} | Корень из среднего квадратичного относительной ошибки |
| **FMAE** | {fmae:.6f} | Среднее абсолютное значение относительной ошибки |
| **Количество точек** | {n_points} | Число временных точек, использованных для вычисления метрик |

## Интерпретация

- **FMSE** и **RMSE**: показывают общую величину ошибки, при этом RMSE имеет ту же размерность, что и относительная ошибка
- **FMAE**: показывает среднюю абсолютную относительную ошибку, более устойчивую к выбросам
- Меньшие значения всех метрик указывают на лучшее соответствие между PSMC и NN

## Временной диапазон

Метрики вычислены только для первых 30 интервалов NN (до {max_time_years:.0f} лет = {CUTOFF_GENERATIONS} поколений) на временных точках PSMC, где оба значения (PSMC и NN) конечны и PSMC ≠ 0.

## Примечания

- Метрики вычисляются как относительные ошибки, что позволяет сравнивать значения Ne на разных временных масштабах
- PSMC используется как референс, так как это стандартный метод для реконструкции демографической истории
"""
    
    out_path.write_text(md_content, encoding="utf-8")
    print(f"✓ Метрики сохранены в: {out_path}")
    print(f"\nРезультаты:")
    print(f"  FMSE: {fmse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  FMAE: {fmae:.6f}")
    print(f"  Точки: {n_points}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
