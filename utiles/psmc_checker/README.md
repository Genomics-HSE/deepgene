## PSMC generator (deepgene)

Этот каталог содержит актуальный генератор `.psmcfa`, который использует демографии из текущего пакета `deepgene`, и helper для загрузки `.psmcfa` в нейросеть.

### Генерация `.psmcfa`

Скрипт: `psmc_checker/generate_psmcfa_deepgene.py`

- **Режим “как в NN-обучении”** (по умолчанию): `sequence_length=3_000_000`, `bin_size=100` ⇒ 30_000 символов в `.psmcfa`.\n
```bash
python psmc_checker/generate_psmcfa_deepgene.py \
  --demography get_zigzag \
  --num-replicates 100 \
  --sequence-length 3000000 \
  --bin-size 100 \
  --out /tmp/zigzag_3M.psmcfa \
  --overwrite
```

- **Режим “как классический PSMC”**: `sequence_length=30_000_000`, `bin_size=100` ⇒ 300_000 символов.\n
```bash
python psmc_checker/generate_psmcfa_deepgene.py \
  --demography get_humanlike \
  --num-replicates 10 \
  --sequence-length 30000000 \
  --bin-size 100 \
  --out /tmp/humanlike_30M.psmcfa \
  --overwrite
```

Можно писать **по одному файлу на репликат**:

```bash
python psmc_checker/generate_psmcfa_deepgene.py \
  --demography get_set3 \
  --num-replicates 100 \
  --sequence-length 3000000 \
  --bin-size 100 \
  --out-dir /tmp/set3_psmcfa \
  --overwrite
```

### Загрузка `.psmcfa` в NN-формат

Helper: `psmc_checker/psmcfa_helper.py`

Пример: прочитать multi-fasta `.psmcfa` и получить бинарный вектор 0/1:

```python
from psmc_checker.psmcfa_helper import read_psmcfa_records, psmcfa_seq_to_binary

for rec in read_psmcfa_records("/tmp/zigzag_3M.psmcfa"):
    x = psmcfa_seq_to_binary(rec.sequence)  # np.uint8, shape=(L_bins,)
    break
```

Пример: нарезать на окна под `seq_len` deepgene (например 30_000):

```python
from psmc_checker.psmcfa_helper import load_psmcfa_as_binary, iter_windows

x = load_psmcfa_as_binary("/tmp/zigzag_3M.psmcfa", record_index=0)
windows = list(iter_windows(x, window_len=30000, step=30000, pad=False))
```

### Реальные данные (1000 Genomes): запуск PSMC на готовом `.psmcfa`

Мы подготовили пример `.psmcfa` из 1000G (sample `HG00096`, chr22) и скопировали сюда:

- `psmc_checker/runs/1000g/HG00096_chr22.psmcfa`

Запуск PSMC (аргументы как в `psmc_checker/Makefile`):

```bash
cd psmc_checker
psmc -N25 -t15 -r5 -p 4+25*2+4+6 \
  -o "runs/1000g/HG00096_chr22.psmc" \
  "runs/1000g/HG00096_chr22.psmcfa"
```

Примечание: бинарь `psmc_checker/psmc` в этом репозитории — **Linux ELF**, на macOS он не запускается. На macOS используйте `psmc` из conda/bioconda или соберите PSMC локально.

### Демографии

Параметр `--demography` принимает **имя функции** из `deepgene.data.data_gen_np`, например:\n
- `get_const_demographcs`\n
- `get_humanlike`\n
- `get_zigzag`\n
- `get_set1`, `get_set2`, `get_set2_plus`, `get_set3`\n
- `get_one_of_many`, `get_one_of_two`\n
- `get_strong_bootleneck`\n

### Визуализация PSMC vs TRUE демография (из `psmc_checker/runs`)

Сгенерировать PNG рядом с каждым `*.psmc` из `psmc_checker/runs/**`:

```bash
python psmc_checker/plot_runs.py --runs-dir psmc_checker/runs
```

По умолчанию время на оси X будет в **поколениях** (generation_time=1). Для отображения в годах:

```bash
python psmc_checker/plot_runs.py --runs-dir psmc_checker/runs --generation-time 25
```
