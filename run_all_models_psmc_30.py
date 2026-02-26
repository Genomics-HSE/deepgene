#!/usr/bin/env python3
"""
Скрипт для запуска сравнения PSMC vs NN для всех моделей на синтетических данных
с ограничением до первых 30 интервалов.
"""

import subprocess
import sys
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    model_dir = base_dir / "test_all_v1" / "model"
    runs_dir = base_dir / "psmc_checker" / "runs"
    out_base_dir = base_dir / "new_res_30" / "psmc_synthetic"
    
    # Находим все checkpoint файлы
    ckpt_files = sorted(model_dir.glob("*.ckpt"))
    
    if not ckpt_files:
        print(f"Не найдено checkpoint файлов в {model_dir}")
        return 1
    
    print(f"Найдено {len(ckpt_files)} моделей для обработки")
    
    failed = []
    success = []
    
    for ckpt in ckpt_files:
        model_name = ckpt.stem
        print(f"\n{'='*60}")
        print(f"Обработка модели: {model_name}")
        print(f"{'='*60}")
        
        # Создаем поддиректорию для этой модели
        model_out_dir = out_base_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Запускаем plot_runs_with_nn.py
        cmd = [
            sys.executable,
            "-m", "psmc_checker.plot_runs_with_nn",
            "--runs-dir", str(runs_dir),
            "--out-dir", str(model_out_dir),
            "--checkpoint", str(ckpt),
            "--overwrite"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(base_dir)
            )
            
            if result.returncode == 0:
                print(f"✓ Успешно обработана модель {model_name}")
                success.append(model_name)
            else:
                print(f"✗ Ошибка при обработке {model_name}:")
                print(result.stderr)
                failed.append(model_name)
        except Exception as e:
            print(f"✗ Исключение при обработке {model_name}: {e}")
            failed.append(model_name)
    
    print(f"\n{'='*60}")
    print(f"Итоги:")
    print(f"  Успешно: {len(success)}")
    print(f"  Ошибок: {len(failed)}")
    if failed:
        print(f"\nМодели с ошибками:")
        for model in failed:
            print(f"  - {model}")
    print(f"{'='*60}")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
