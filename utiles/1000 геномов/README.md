# 1000 Genomes → PSMC + Deepgene (NN) input

Эта директория содержит **инструкцию и скрипты**, чтобы подготовить данные из **1000 Genomes** (вход: **CRAM**) в:

- формат **PSMC**: файл `.psmcfa` (FASTA-подобный формат PSMC);
- формат **deepgene NN**: окна 0/1 (мутация/нет) длиной `window_len` бинов (по умолчанию **30 000** бинов, как в наших экспериментах).

**Новое:** Поддержка **нескольких хромосом** для увеличения размера валидационного датасета. Целевой размер: **~30 Mb** (сопоставимо с `L_HUMAN=30_000_000` из валидации на искусственных данных).

Вы просили: **инференс вынесен в отдельный скрипт**.

---

## Что считается “нашим NN-форматом”

В репозитории есть готовый хелпер для `.psmcfa`:

- [`psmc_checker/psmcfa_helper.py`](../psmc_checker/psmcfa_helper.py) — чтение `.psmcfa` и преобразование в бинарный `np.uint8` вектор (0/1).

Наши окна строятся из этого вектора:

- `window_len=30000`, `step=30000` при `bin_size=100` соответствует окнам по **3 Mb** (потому что 30 000 * 100 bp = 3 000 000 bp).

---

## Требования (утилиты)

Скрипт подготовки (`prepare_1000g_cram_to_psmc_nn.sh`) ожидает в `$PATH`:

- `samtools`
- `bcftools`
- `vcfutils.pl` (обычно ставится вместе со `bcftools` / `samtools` пакетом; файл-скрипт из набора bcftools)
- `python` (для преобразования `.psmcfa` → окна)

PSMC бинарь для запуска собственно PSMC:

- либо системный `psmc` в `$PATH`,
- либо `../psmc_checker/psmc` (если у вас он собран/исполняемый).

Если вы клонировали репозиторий и скрипты не исполняемые, сделайте:

```bash
chmod +x "1000 геномов/"*.sh "1000 геномов/"*.py
```

---

## Входные данные

- **CRAM** (`*.cram`) + индекс `*.cram.crai`
- референс **GRCh37** (hg19) FASTA + индекс `*.fai`
  - Для 1000G часто нужен именно **`hs37d5.fa`** (decoy). Важно, чтобы **континги совпадали** между CRAM и reference.

Скрипт поддерживает **один или несколько хромосом**:
- Одна хромосома: `--chrom 22` или `--chrom chr22`
- Несколько хромосом: `--chrom 21,22` или `--chrom chr21,chr22`

**Рекомендация:** Для валидации используйте несколько хромосом, чтобы достичь целевого размера **~30 Mb** (сопоставимо с размерами на искусственных данных).

---

## Быстрый запуск (подготовка)

### Одна хромосома

Пример (chr22, GRCh37):

```bash
bash "1000 геномов/prepare_1000g_cram_to_psmc_nn.sh" \
  --sample NA12878 \
  --cram /path/to/NA12878.cram \
  --ref /path/to/hs37d5.fa \
  --chrom 22 \
  --outdir /path/to/out_1000g
```

### Несколько хромосом (рекомендуется для валидации)

Для достижения целевого размера **~30 Mb** используйте несколько хромосом:

```bash
bash "1000 геномов/prepare_1000g_cram_to_psmc_nn_v2_stream.sh" \
  --sample NA12878 \
  --cram /path/to/NA12878.cram \
  --ref /path/to/hs37d5.fa \
  --chrom 21,22 \
  --outdir /path/to/out_1000g
```

### Автоматический подбор хромосом

Для автоматического подбора хромосом, близких к целевому размеру (~30 Mb):

```bash
CHROMS=$(python "1000 геномов/select_chromosomes.py" --target-size 30000000)
bash "1000 геномов/prepare_1000g_cram_to_psmc_nn_v2_stream.sh" \
  --sample NA12878 \
  --cram /path/to/NA12878.cram \
  --ref /path/to/hs37d5.fa \
  --chrom "$CHROMS" \
  --outdir /path/to/out_1000g
```

### V2: без сохранения гигантского VCF (рекомендуется)

Если не хотите писать на диск огромный `calls.vcf`, используйте streaming-версию:

```bash
bash "1000 геномов/prepare_1000g_cram_to_psmc_nn_v2_stream.sh" \
  --sample NA12878 \
  --cram /path/to/NA12878.cram \
  --ref /path/to/hs37d5.fa \
  --chrom 21,22 \
  --outdir /path/to/out_1000g
```

Опционально, для отладки можно добавить `--keep-vcf`, чтобы всё же сохранить `calls.vcf`.

### Структура выходных данных

**Одна хромосома:**
```
out_1000g/
  NA12878/
    chr22/
      calls.vcf (если --keep-vcf)
      consensus.fq
      input.psmcfa
      nn_windows.npz
      nn_windows_manifest.tsv
      nn_meta.json
```

**Несколько хромосом:**
```
out_1000g/
  NA12878/
    chr21/
      consensus.fq
      input.psmcfa
      nn_windows.npz (промежуточный)
      nn_windows_manifest.tsv (промежуточный)
      nn_meta.json (промежуточный)
    chr22/
      consensus.fq
      input.psmcfa
      nn_windows.npz (промежуточный)
      nn_windows_manifest.tsv (промежуточный)
      nn_meta.json (промежуточный)
    combined/
      nn_windows.npz (финальный, все хромосомы)
      nn_windows_manifest.tsv (финальный, все хромосомы)
      nn_meta.json (финальный, метаданные всех хромосом)
```

**Для инференса используйте файлы из `combined/` директории.**

---

## Инференс (отдельно)

После подготовки можно прогнать инференс:

**Одна хромосома:**
```bash
bash "1000 геномов/run_nn_inference.sh" \
  --checkpoint /path/to/model.ckpt \
  --windows /path/to/out_1000g/NA12878/chr22/nn_windows.npz \
  --outdir /path/to/out_1000g/NA12878/chr22/nn_infer
```

**Несколько хромосом (используйте объединённые окна):**
```bash
bash "1000 геномов/run_nn_inference.sh" \
  --checkpoint /path/to/model.ckpt \
  --windows /path/to/out_1000g/NA12878/combined/nn_windows.npz \
  --outdir /path/to/out_1000g/NA12878/combined/nn_infer
```

Скрипт также умеет принимать `.psmcfa` напрямую (с генерацией окон на лету).

Подсказка: чтобы получить кривую Ne из NN-выхода и сравнить с PSMC, можно использовать:

```bash
python psmc_checker/plot_results.py \
  --psmc /path/to/output.psmc \
  --nn-priors-json /path/to/nn_infer/nn_priors.json \
  --nn-key priors
```

---

## Автоматический подбор хромосом

Скрипт `select_chromosomes.py` помогает автоматически выбрать хромосомы для достижения целевого размера:

```bash
# Подобрать хромосомы для ~30 Mb
python "1000 геномов/select_chromosomes.py" --target-size 30000000

# С предпочтением маленьких хромосом (21, 22, и т.д.)
python "1000 геномов/select_chromosomes.py" --target-size 30000000 --prefer-small

# С другим форматом вывода
python "1000 геномов/select_chromosomes.py" --target-size 30000000 --format space
```

Параметры:
- `--target-size`: целевой размер в bp (по умолчанию 30,000,000 = 30 Mb)
- `--tolerance`: допуск как доля от целевого размера (по умолчанию 0.2 = 20%)
- `--prefer-small`: предпочитать маленькие хромосомы (21, 22, и т.д.)
- `--format`: формат вывода (`comma`, `space`, `list`)

## Типичные проблемы / troubleshooting

- **CRAM vs reference mismatch**: если `bcftools mpileup` ругается на континги/MD5 — почти всегда нужен правильный reference (часто `hs37d5.fa`) и `.fai`.
- **Имя хромосомы**: в CRAM может быть `22`, а не `chr22` (или наоборот). Передайте `--chrom` ровно как в CRAM header. При использовании нескольких хромосом используйте одинаковый формат для всех (либо все с `chr`, либо все без).
- **Нет `fq2psmcfa` / `vcfutils.pl`**: установите PSMC и bcftools-утилиты; скрипт остановится и подскажет, чего не хватает.
- **Нет `vcfutils.pl`**: установите bcftools/samtools пакет (часто `vcfutils.pl` идёт вместе).
- **Несколько хромосом**: при использовании нескольких хромосом убедитесь, что все они присутствуют в CRAM файле и reference. Проверить можно через `samtools view -H sample.cram | grep "^@SQ"`.

