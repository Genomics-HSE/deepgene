#!/usr/bin/env python3
"""
Генерация графиков Ne для всех true ground демографий из data_gen_np.
Для функций get_set1(), get_set2(), get_set2_plus(), get_set3() визуализирует каждую демографию отдельно.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import sys

# Импортируем data_gen_np напрямую, минуя __init__.py чтобы избежать загрузки torch
spec = importlib.util.spec_from_file_location(
    "data_gen_np", 
    Path(__file__).parent / "deepgene" / "data" / "data_gen_np.py"
)
dgn = importlib.util.module_from_spec(spec)
sys.modules["data_gen_np"] = dgn
spec.loader.exec_module(dgn)

# Получаем limits из модуля
limits = dgn.limits


def dem2Ne(demography, limits=limits):
    """Преобразует демографию в траекторию Ne"""
    debug = demography.debug()
    return debug.population_size_trajectory(limits)


def create_true_dem_plot(dem, dem_name, file_name):
    """
    Создает график Ne с полосатым фоном
    
    Args:
        dem: объект демографии msprime.Demography
        dem_name: имя демографии (для легенды)
        file_name: путь к файлу для сохранения
    """
    x = np.linspace(0, 250*10**4, 100000)
    y = [25000 for i in x]
    
    fig, ax = plt.subplots()
    
    # Полосатый фон (чередование lightgray/white)
    for i in range(len(limits) - 1):
        x1, x2 = limits[i], limits[i+1]
        # Чередуем серый и белый
        color = 'lightgray' if i % 2 == 0 else 'white'
        ax.fill_between(x, y, where=(x1 <= x) & (x <= x2), 
                        step='post', color=color, alpha=0.3)
    
    # Вычисляем Ne траекторию
    Ne = dem2Ne(dem, limits)
    
    # Настройка осей
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim(0, 100000)
    plt.ylim(0, 25000)
    
    # График Ne траектории
    plt.step(limits, Ne, where='post', label="True", color="red")
    
    plt.legend()
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Создан график: {file_name}")


def generate_set_variants(set_func, set_name, num_variants, variant_names, output_dir, use_plus=False):
    """
    Генерирует графики для всех вариантов функции get_set*()
    
    Args:
        set_func: функция get_set1, get_set2, get_set2_plus или get_set3
        set_name: имя набора (например, 'set1')
        num_variants: количество вариантов
        variant_names: список имен вариантов (например, ['const', 'hl', 'random'])
        output_dir: директория для сохранения
        use_plus: использовать current_K_plus вместо current_K (для get_set2_plus)
    """
    # Сохраняем начальное значение current_K или current_K_plus
    if use_plus:
        initial_k = dgn.current_K_plus
    else:
        initial_k = dgn.current_K
    
    try:
        for k in range(num_variants):
            # Устанавливаем current_K или current_K_plus в нужное значение
            if use_plus:
                dgn.current_K_plus = k
            else:
                dgn.current_K = k
            
            # Получаем демографию
            dem = set_func()
            
            # Имя файла
            variant_name = variant_names[k] if k < len(variant_names) else f'variant_{k}'
            file_name = output_dir / f"{set_name}_{variant_name}.png"
            
            # Создаем график
            create_true_dem_plot(dem, f"{set_name}_{variant_name}", file_name)
            
    finally:
        # Восстанавливаем начальное значение
        if use_plus:
            dgn.current_K_plus = initial_k
        else:
            dgn.current_K = initial_k


def main():
    """Основная функция"""
    # Создаем папку для результатов
    output_dir = Path("all_true_dems")
    output_dir.mkdir(exist_ok=True)
    print(f"Создана папка: {output_dir}")
    
    # Простые демографии (одна демография на функцию)
    print("\n=== Обработка простых демографий ===")
    
    # const
    dem_const = dgn.get_const_demographcs()
    create_true_dem_plot(dem_const, "const", output_dir / "const.png")
    
    # hl (humanlike)
    dem_hl = dgn.get_humanlike()
    create_true_dem_plot(dem_hl, "hl", output_dir / "hl.png")
    
    # zigzag
    dem_zigzag = dgn.get_zigzag()
    create_true_dem_plot(dem_zigzag, "zigzag", output_dir / "zigzag.png")
    
    # Множественные демографии (get_set*)
    print("\n=== Обработка множественных демографий ===")
    
    # get_set1() - 3 варианта
    print("\n--- set1 ---")
    generate_set_variants(
        dgn.get_set1,
        "set1",
        num_variants=3,
        variant_names=['const', 'hl', 'random'],
        output_dir=output_dir
    )
    
    # get_set2() - 5 вариантов
    print("\n--- set2 ---")
    generate_set_variants(
        dgn.get_set2,
        "set2",
        num_variants=5,
        variant_names=['const', 'hl', 'random', 'growth_1.1', 'growth_0.9'],
        output_dir=output_dir
    )
    
    # get_set2_plus() - 6 вариантов
    print("\n--- set2_plus ---")
    generate_set_variants(
        dgn.get_set2_plus,
        "set2_plus",
        num_variants=6,
        variant_names=['const', 'hl', 'random', 'growth_1.1', 'growth_0.9', 'commented_dem'],
        output_dir=output_dir,
        use_plus=True
    )
    
    # get_set3() - 6 вариантов
    print("\n--- set3 ---")
    generate_set_variants(
        dgn.get_set3,
        "set3",
        num_variants=6,
        variant_names=['const', 'hl', 'random', 'growth_1.1', 'growth_0.9', 'zigzag'],
        output_dir=output_dir
    )
    
    print(f"\n✓ Все графики созданы в папке: {output_dir}")


if __name__ == "__main__":
    main()
