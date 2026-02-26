# -*- coding: utf-8 -*-
import gin
import numpy as np
import torch
import pytorch_lightning
from deepgene.models import GruLabeler, LstmLabeler, ReformerLabeler 
from deepgene.data import DatasetXY
from tqdm import tqdm
import scipy 
import seaborn as sns
from collections import Counter

from deepgene.data import data_gen_np as dgn

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from os.path import join

from pathlib import Path

import json
import os
import sys
import argparse

import msprime


from tqdm import tqdm
import matplotlib 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report

import time
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import classification_report

L_HUMAN = 3000000 * 1  # int(float(sys.argv[3])) #30_000_000
RHO_HUMAN = 1.6 * 10e-9
MU_HUMAN = 1.25 * 10e-8

limits = dgn.limits

# Определяем базовую директорию скрипта для правильных путей к конфигам
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_checkpoint_files(input_dir):
    """
    Находит все .ckpt файлы в директории {input_dir}/model/
    
    Args:
        input_dir: Базовая директория, содержащая папку model
        
    Returns:
        Список полных путей к .ckpt файлам
    """
    model_dir = os.path.join(input_dir, "model")
    if not os.path.exists(model_dir):
        raise ValueError("Директория {} не существует".format(model_dir))
    
    ckpt_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.ckpt'):
            ckpt_files.append(os.path.join(model_dir, file))
    
    if not ckpt_files:
        raise ValueError("Не найдено .ckpt файлов в директории {}".format(model_dir))
    
    return sorted(ckpt_files)


def get_device(device_arg=None):
    """
    Определяет устройство для вычислений (GPU или CPU)
    
    Args:
        device_arg: "cuda", "cpu" или "auto" (по умолчанию)
        
    Returns:
        torch.device объект
    """
    if device_arg is None or device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Предупреждение: CUDA недоступна, используется CPU")
            return torch.device('cpu')
        return torch.device('cuda')
    elif device_arg == "cpu":
        return torch.device('cpu')
    else:
        raise ValueError("Неизвестное устройство: {}. Используйте 'cuda', 'cpu' или 'auto'".format(device_arg))


def clear_gpu_memory():
    """Очищает память GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def detect_model_type_from_checkpoint(path):
    """
    Определяет тип модели из содержимого checkpoint файла
    
    Args:
        path: Путь к checkpoint файлу
        
    Returns:
        Тип модели: "gru", "lstm" или "reformer"
    """
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Проверяем ключи в state_dict
    keys = list(state_dict.keys())
    
    # Проверяем наличие ключей, характерных для каждой модели
    if any('gru.' in k for k in keys):
        return "gru"
    elif any('lstm.' in k for k in keys):
        return "lstm"
    elif any('reformer' in k.lower() or 'config' in k.lower() for k in keys):
        return "reformer"
    else:
        # Если не можем определить, пробуем по имени файла
        filename = os.path.basename(path).lower()
        if 'gru' in filename:
            return "gru"
        elif 'lstm' in filename:
            return "lstm"
        elif 'reformer' in filename:
            return "reformer"
        else:
            raise ValueError("Не удалось определить тип модели из checkpoint: {}".format(path))


def load_model(key, path, device): 

    gin.clear_config()
    
    # Определяем реальный тип модели из checkpoint
    actual_model_type = detect_model_type_from_checkpoint(path)
    
    # Строгая проверка: не подменяем архитектуру "молча", иначе результаты могут попасть в неверные папки
    if key != actual_model_type:
        raise ValueError(
            "Тип модели из аргумента --model_type ('{}') не совпадает с типом в checkpoint ('{}') для файла: {}".format(
                key, actual_model_type, path
            )
        )
    
    # Извлекаем имя модели и loss function из пути к checkpoint
    checkpoint_filename = os.path.basename(path)
    model_name = os.path.splitext(checkpoint_filename)[0]
    
    # Определяем loss function из имени файла
    if '_ce' in model_name.lower() or '_crossentropy' in model_name.lower():
        loss_function = 'CrossEntropyLoss'
    elif '_kl' in model_name.lower() or '_kldiv' in model_name.lower():
        loss_function = 'KLDivLoss'
    else:
        loss_function = 'CrossEntropyLoss'  # по умолчанию
    
    # Определяем seq_len (по умолчанию 3000000)
    seq_len = 3000000
    
    # Определяем output директорию (временная, не используется для mark_all)
    output_dir = os.path.join(os.path.dirname(path), "..")
    
    # Формируем gin параметры
    gin_params = [
        "model_name='{}'".format(model_name),
        "output='{}'".format(output_dir),
        "seq_len={}".format(seq_len),
        "loss_function='{}'".format(loss_function)
    ]

    # Важно: LSTM чекпойнты могут быть обучены с разными embedding (ConvEmbedding или NoEmbedding).
    # Если выбрать неправильный embedding через gin, Lightning упадет на load_state_dict (missing keys).
    if key == "lstm":
        checkpoint_obj = torch.load(path, map_location='cpu')
        state_dict = checkpoint_obj.get('state_dict', checkpoint_obj)
        has_conv_embed = any(k.startswith("embedding.conv_embeds.") for k in state_dict.keys())
        if has_conv_embed:
            gin_params.append("LstmLabeler.embedding = @ConvEmbedding()")
            print("LSTM embedding (detected from checkpoint): ConvEmbedding")
        else:
            gin_params.append("LstmLabeler.embedding = @NoEmbedding()")
            print("LSTM embedding (detected from checkpoint): NoEmbedding (ConvEmbedding weights not found in checkpoint)")

    if key == "reformer":
        config_dir = os.path.join(SCRIPT_DIR, "configs")
        gin.parse_config_files_and_bindings(
        [
            os.path.join(config_dir, "data_universal.gin"), 
            os.path.join(config_dir, "train_universal.gin"), 
            os.path.join(config_dir, "model_reformer.gin")
        ], gin_params)

        ret = ReformerLabeler()

        state_dict = torch.load(path, map_location=device)

        models_set = set(ret.state_dict().keys())
        chekpoint_set = set(state_dict['state_dict'].keys())

        remove = chekpoint_set - models_set
        state_dict = state_dict['state_dict']

        for key_to_remove in remove:
            if key_to_remove in state_dict: del state_dict[key_to_remove]

        model=ReformerLabeler()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    elif key == "lstm":
        config_dir = os.path.join(SCRIPT_DIR, "configs")
        gin.parse_config_files_and_bindings(
        [
            os.path.join(config_dir, "data_universal.gin"), 
            os.path.join(config_dir, "train_universal.gin"), 
            os.path.join(config_dir, "model_lstm.gin")
        ], gin_params)

        model = LstmLabeler.load_from_checkpoint(checkpoint_path=path, map_location=device)
        model.to(device)
        model.eval()


    elif key == "gru":
        config_dir = os.path.join(SCRIPT_DIR, "configs")
        gin.parse_config_files_and_bindings(
        [
            os.path.join(config_dir, "data_universal.gin"), 
            os.path.join(config_dir, "train_universal.gin"), 
            os.path.join(config_dir, "model_gru.gin")
        ], gin_params)

        model = GruLabeler.load_from_checkpoint(checkpoint_path=path, map_location=device)
        model.to(device)
        model.eval()

    return model





dems_name = ["const", "hl", "zigzag"]

def load_dem(dem_name):
    if dem_name == "const": return dgn.get_const_demographcs()
    elif dem_name == "hl": return dgn.get_humanlike()
    elif dem_name == "zigzag": return dgn.get_zigzag()


def do_filter(mutations):
    l = int(len(mutations) / 100)
    genome = [0] * l
    for j in range(int(len(mutations) / 100)):
        genome[j] = max(mutations[j * 100:(j + 1) * 100])
    return genome


def myexp(data):
    return scipy.special.softmax(data, axis=-1)

### Разные способы округления

def g(x): # текущая фнкция, которая используется
    return myexp(x[0]).argmax(axis=1)

def f(x):
    return myexp(x[0])

###

def dem2Ne(demography, limits=limits):
    debug = demography.debug()
    return debug.population_size_trajectory(limits)

def NN2pd_v2(X, num_intervals = 32):
    return np.array(X).sum(axis=0)/(L_HUMAN/100)

def absolute_error(pred, ground_true):
  return [(i - j)/j for (i,j) in zip(pred, ground_true)]

def from_probability_to_N(priors, limits):
    l = 1
    tau = np.array(limits[1:]) - np.array(limits[:-1])
    N = []
    for j in range(len(tau)):
        N.append(
            np.round(tau[j]/(-2*np.log(1 - (priors[j]/l))))
        )
        l *= np.exp(-tau[j]/(2*N[j]))
    return N


def create_NE_plot(NE_pred, NE_true, dem, file_name):
    x = np.linspace(0, 250*10**4, 100000)
    y = [25000 for i in x]

    fig, ax = plt.subplots()

    # Создание цветовой карты
    try:
        # Для новых версий matplotlib (3.5+)
        cmap = matplotlib.colormaps.get_cmap('viridis')
    except AttributeError:
        # Для старых версий matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap('viridis')
    # cmap = ['orange', 'blue', 'green']

    # Заполнение области разными цветами
    # for color, (x1, x2) in enumerate(zip(limits, limits[1:] +  [10**10])):
    #     # ax.fill_between(x, y, where=(x1 <= x) & (x <= x2), step="post" ,color = cmap(color))
    #     # print(f"{color}: {x1} < x < {x2} L= {x2-x1}")
    #     ax.fill_between(x, y, where=(x1 <= x) & (x <= x2), step="post" ,color = cmap(8*color))
    #     # ax.text(x, 15000, f"{round(p[min(color,6)][0],4)}")
    #     # ax.scatter([x1],[10000], color = 'red')


    # Нанесение графика линии поверх заполненных областей

    # Настройка осей и отображение графика
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim(0, 100000)
    plt.ylim(0, 25000)
    # plt.ylim(0, max(NE_pred) + 5000)
    # plt.title('Раскрашивание пространства по разным x')


    if dem is not None: 
        Ne = dem2Ne(dem)
        plt.step(limits, Ne, where='post', label="True", color="red")

    if NE_true is not None:     
        plt.step(limits, NE_true + [NE_true[-1]], where='post', label="From data", color="blue")

    plt.step(limits, NE_pred + [NE_pred[-1]], where='post', label="NN", color="black", linestyle='-.', linewidth = 1.5)

    # plt.xscale('log')
    # plt.yscale('log')

    plt.legend()
    plt.savefig(file_name)

def Ne_plot(p_pred, p_true, dem, file_name):

    Ne_pred = from_probability_to_N(p_pred ,limits)
    if p_true is not None: Ne_true = from_probability_to_N(p_true ,limits)
    else: Ne_true = None
    round_to = 3
    # discrepancies = sum([round(abs(round(p1, round_to) - round(p2, round_to)), 4) for p1,p2 in zip(p_pred, p_true)])
    # print(f"Расхождения: {discrepancies}")

    create_NE_plot(Ne_pred, Ne_true, dem, file_name)

def create_heatmap(y_pred,
                   y_true, 
                   file_name = "output.png",
                   length=5000):
    
    # print(f"\nLen of dist data in heatmap: {len(y_true)}")

    sns.set()  # Set Seaborn style
    plt.clf()  # Clear existing figure

    sns.set()  # Setting seaborn as default style even if use only matplotlib
    sns.set_context("paper")
    sns.set(rc={'figure.figsize': (11.7, 8.27),
                'axes.labelsize': 20,
                'axes.titlesize': 20,
                'figure.dpi': 100
                }
            )
    
    # Prepare data
    data = np.transpose(y_pred[:length])
    
    # Create custom colormap from gray to plasma color (without red)
    # Get plasma colormap colors, but exclude red tones (use only up to 0.8 to avoid red)
    try:
        plasma_cmap = matplotlib.colormaps.get_cmap('plasma')
    except AttributeError:
        plasma_cmap = plt.cm.get_cmap('plasma')
    # Create gradient from lightgray to plasma colors (without red - stop at 0.8)
    colors = ['lightgray', plasma_cmap(0.0), plasma_cmap(0.4), plasma_cmap(0.8)]
    n_bins = 256
    cmap_custom = LinearSegmentedColormap.from_list('gray_to_plasma_no_red', colors, N=n_bins)
    
    # Set axes background to gray for zero values
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    ax.set_facecolor('lightgray')
    
    # Use custom colormap (gray to plasma without red) - no need for mask, colormap handles it
    ax = sns.heatmap(data, cmap=cmap_custom, yticklabels=5, xticklabels=1000, ax=ax, cbar_kws={'label': 'Probability'})
    
    if y_true is not None: 
        ax.plot(y_true[:length], c='red', linewidth=1.5)
    _ = ax.set(xlabel='Позиция вдоль генома', ylabel='Интервал коалесенции')
    # _ = ax.set(xlabel='Genome wide', ylabel='Coalescent time')
    ax.figure.savefig(file_name)

    return ax


### Надо разивать на 3000000, чтобы он мог скушать 


real_data_files = [
    "real_data/NA20504_22chr.txt"
]

lenght_chr22 = 52000000

def mark_real_data(model, save_path, device):
    for fpath in real_data_files:

        fname = os.path.basename(fpath)
        fname = os.path.splitext(fname)[0]
        print("Сейчас тестирую на реальных данных на {}".format(fname))
        with open(fpath, 'r') as file:
        # Читаем все строки, убираем пробельные символы и преобразуем в целые числа
            numbers = [int(line.strip()) for line in file]
        
        mutations = [0] * lenght_chr22 
        for m in numbers: mutations[int(m)] = 1

        mutations = do_filter(mutations)

        model_dist = [0.0] * number_intervals

        for i in tqdm(range(0, len(mutations), 30000)): 

            block = mutations[i:i+30000]
            X = torch.tensor([block]).to(device)
            y_pred = model(X).detach().cpu().numpy()
            yy_pred = [i for i in f(y_pred)]
            # y_pred = [i for i in g(y_pred)]
            p_pred = NN2pd_v2(yy_pred)
            model_dist = model_dist + p_pred
        

        ss = sum(model_dist)
        nn = [p/ss for p in model_dist]


        Ne_plot(nn, None, None, file_name='{}/NE_{}.png'.format(save_path, fname))
        create_heatmap(yy_pred, None, file_name='{}/heatmap_{}.png'.format(save_path, fname))


def precision_at_k_with_neighbors(y_true, y_pred, k=3):
    if len(y_true) != len(y_pred): raise ValueError("Длины y_true и y_pred должны совпадать.")
    half_k = k // 2
    hits = sum(abs(t - p) <= half_k for t, p in zip(y_true, y_pred))
    return hits / len(y_true)

def precision_at_k_with_neighbors(y_true, y_pred, k=3, file_path="debug_precision.json"):
    if len(y_true) != len(y_pred):
        raise ValueError("Длины y_true и y_pred должны совпадать.")
    if len(y_true) == 0:
        raise ValueError("Входные массивы не должны быть пустыми.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k должно быть положительным целым числом.")

    half_k = k // 2
    hits = sum(abs(t - p) <= half_k for t, p in zip(y_true, y_pred))
    precision = hits / len(y_true)

    if precision > 1.0:
        # Сохраняем массивы в файл
        data = {
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "precision": precision
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("ВНИМАНИЕ: precision = {} > 1.0. Последовательности сохранены в {}".format(precision, file_path))
        sys.exit(1)

    return precision

def save_arrays_to_file(y_true, y_pred, file_path):
    """
    Сохраняет два массива в файл в формате JSON.

    Аргументы:
        y_true (list): Массив истинных значений.
        y_pred (list): Массив предсказанных значений.
        file_path (str): Путь к файлу для сохранения.
    """
    data = {
        "y_true": y_true,
        "y_pred": y_pred
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


splitter = dgn.exponent_split
recombination_rate = dgn.RHO_HUMAN
mutation_rate = dgn.MU_HUMAN
llen = L_HUMAN
num_replicates = 100
# demographic_events =  choice_dem(demographic_events_name) #get_special() # get_const_demographcs #get_strong_bootleneck # get_decline # get_dem22012025 # get_strong_antibootleneck
model = "hudson"
# exp_name = "test"
sample_size =  1
number_intervals = 32

real_data = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Оценка моделей на различных демографиях')
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Директория с папкой model, содержащей .ckpt файлы")
    parser.add_argument("--model_type", type=str, required=True, choices=["gru", "lstm", "reformer"],
                        help="Тип архитектуры модели")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Директория для сохранения результатов (по умолчанию {input_dir}/results)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Устройство для вычислений (auto/cuda/cpu)")
    parser.add_argument("--real_data", action="store_true",
                        help="Запустить тестирование на реальных данных")
    parser.add_argument("--allow_mixed", action="store_true",
                        help="Разрешить смесь архитектур в {input_dir}/model: будут обработаны только checkpoint'ы типа --model_type, остальные будут пропущены (с логом).")
    
    args = parser.parse_args()
    
    # Определяем устройство
    device = get_device(args.device)
    print("Используемое устройство: {}".format(device))
    
    # Определяем директорию для результатов
    if args.output_dir is None:
        output_dir = os.path.join(args.input_dir, "results")
    else:
        output_dir = args.output_dir
    
    # Находим все checkpoint файлы
    checkpoint_files = find_checkpoint_files(args.input_dir)
    print("Найдено {} checkpoint файлов".format(len(checkpoint_files)))

    # Проверка типов checkpoint'ов (и опциональная фильтрация при allow_mixed)
    mismatches = []
    skipped_by_type = {}
    for checkpoint_path in checkpoint_files:
        detected_type = detect_model_type_from_checkpoint(checkpoint_path)
        if detected_type != args.model_type:
            mismatches.append((checkpoint_path, detected_type))
            skipped_by_type.setdefault(detected_type, []).append(checkpoint_path)

    if mismatches:
        if not args.allow_mixed:
            mismatch_lines = "\n".join(
                ["- {} (detected: {})".format(p, t) for (p, t) in mismatches]
            )
            raise ValueError(
                "В директории {} обнаружены checkpoint'ы разных архитектур.\n"
                "Ожидаемый тип (--model_type): {}\n"
                "Несовпадающие файлы:\n{}\n\n"
                "Подсказка: добавьте --allow_mixed, чтобы пропускать неподходящие checkpoint'ы.".format(
                    args.input_dir, args.model_type, mismatch_lines
                )
            )

        # allow_mixed: фильтруем только нужную архитектуру и логируем пропуски
        total_before = len(checkpoint_files)
        mismatched_paths = {p for (p, _) in mismatches}
        kept = [cp for cp in checkpoint_files if cp not in mismatched_paths]
        print("Включен --allow_mixed: будут обработаны только checkpoint'ы типа '{}'.".format(args.model_type))
        print("Пропущены архитектуры: {}".format(", ".join(sorted(skipped_by_type.keys()))))
        for t in sorted(skipped_by_type.keys()):
            print("Пропущено {} файлов типа '{}':".format(len(skipped_by_type[t]), t))
            for p in skipped_by_type[t]:
                print("  - {}".format(p))
        checkpoint_files = kept
        print("К обработке: {} checkpoint файлов (из {}).".format(len(checkpoint_files), total_before))
    
    model = None
    for checkpoint_path in checkpoint_files:
        # Очищаем память GPU перед загрузкой новой модели
        if model is not None:
            del model
            clear_gpu_memory()
        
        # Загружаем модель
        print("Загрузка модели: {}".format(checkpoint_path))
        model = load_model(args.model_type, checkpoint_path, device)
        
        # Извлекаем имя модели из имени файла
        filename_with_ext = os.path.basename(checkpoint_path)
        troika = os.path.splitext(filename_with_ext)[0]
        base_dir = os.path.join(output_dir, args.model_type, troika)
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        
        ### Real data section ###
        
        # Для реформера надо подумать
        # Мб уже ок
        #if key != "reformer":
        if args.real_data:
            save_path =  base_dir + "/real_data"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            mark_real_data(model, save_path, device)

        for dname in dems_name:
            dem_dir = base_dir + "/{}".format(dname)

            print("Сейчас проверяю модель {} на демографии {}".format(checkpoint_path, dname))

            Path(dem_dir).mkdir(parents=True, exist_ok=True)

            Ne_dir = dem_dir + "/Ne"
            heatmap_dir = dem_dir + "/heatmap"

            Path(Ne_dir).mkdir(parents=True, exist_ok=True)
            Path(heatmap_dir).mkdir(parents=True, exist_ok=True)

            demographic_events = load_dem(dname)

            data = msprime.sim_ancestry(
                recombination_rate=recombination_rate,
                sequence_length=llen,
                num_replicates=num_replicates,
                demography=demographic_events,
                model="hudson",
                samples=sample_size,
                # ploidy=1,
                ## new 
                discrete_genome = True,
                end_time = dgn.limits[-1] + 1
            )
            
            
            eval_time = []

            prior_dist = [0.0] * number_intervals # [0.0] * number_intervals
            model_dist = [0.0] * number_intervals # [0.0] * number_intervals # try [1.0/number_intervals] * number_intervals

            # prior_dist_v = []
            # errors_array = []

            full_ground_true = []
            full_predicted = []

            for e, tree in tqdm(enumerate(data)):
                # print(f"Смотрю на кусочек {e} из {num_replicates}")

                start_time = time.perf_counter()

                mutated_ts = msprime.sim_mutations(tree, rate=mutation_rate)
                d_times = [0] * llen
                mutations = [0] * llen
                # ppp = [0.0] * number_intervals

                for m in mutated_ts.tables.sites.position: mutations[int(m)] = 1
                
                for t in mutated_ts.aslist():
                    interval = t.get_interval()
                    left = interval.left
                    right = interval.right
                    ## Returns the length of the branch (in units of time) joining the specified node to its parent. This is equivalent to
                    ttime = t.get_total_branch_length()/2 #/ (2*25)
                    # times[int(left):int(right)] = [time]*int(right-left)
                    d_times[int(left):int(right)] = [splitter(ttime, number_intervals)] * int(right - left)
                    
                    prior_dist[splitter(ttime, number_intervals)] += (int(right - left)) / llen
                    # ppp[splitter(time, number_intervals)] += (int(right - left)) / llen
                
                mutations = do_filter(mutations)
                X = torch.tensor([mutations]).to(device)
                
                y_pred = model(X).detach().cpu().numpy()


                yy_pred = [i for i in f(y_pred)]
                y_pred = [i for i in g(y_pred)]

                ground_true = do_filter(d_times)

                full_ground_true = np.concatenate([full_ground_true, ground_true])
                full_predicted= np.concatenate([full_predicted, y_pred]) # Мб тут брать yy_pred

                p_pred = NN2pd_v2(yy_pred)
                model_dist = model_dist + p_pred
            
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                
                eval_time.append(elapsed_time)

                if (e > 0) and (e % 25 == 0): 
                    print(" Step {}".format(e))
                    #print("Prior dist")
                    ss = sum(prior_dist)
                    prior = [p/ss for p in prior_dist]
                    #print([round(p/sum(prior_dist), 4) for p in prior_dist])
                    #print("NN dist")
                    ss = sum(model_dist)
                    nn = [p/ss for p in model_dist]
                    # print([round(p/sum(model_dist), 4) for p in model_dist])
                    
                    Ne_plot(nn, prior, demographic_events, file_name='{}/NE_{}.png'.format(Ne_dir, e))
                    create_heatmap(yy_pred, ground_true, file_name='{}/heatmap_{}.png'.format(heatmap_dir, e))


            # Тут анализ
            ss = sum(prior_dist)
            prior_dist = [p/ss for p in prior_dist]
            ss = sum(model_dist)
            model_dist = [p/ss for p in model_dist]
                
            # Различие распределений

            probabipity_errors = absolute_error(model_dist, prior_dist)

            # Различие Ne

            Ne_pred = from_probability_to_N(model_dist, dgn.limits)
            Ne_true = from_probability_to_N(prior_dist, dgn.limits)

            # Ne_errors = absolute_error(Ne_pred, Ne_true)

            # Суммарная матрица ошибок
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
            #                     
            final_confusion_matrix = confusion_matrix(full_predicted, full_ground_true)
            disp = ConfusionMatrixDisplay(confusion_matrix=final_confusion_matrix)

            fig, ax = plt.subplots(figsize=(25, 25))
            ax.set_facecolor('lightgray')

            # Create custom colormap from gray to current color (blue)
            colors = ['lightgray', 'steelblue']
            n_bins = 100
            cmap_custom = LinearSegmentedColormap.from_list('gray_to_blue', colors, N=n_bins)
            
            # Не рисуем цифры в ячейках confusion matrix
            disp.plot(ax=ax, include_values=False, cmap=cmap_custom).figure_.savefig('{}/confusion_matrix.png'.format(dem_dir))

            fig.clear()
            # Сохранить итоговый график Ne
            # С путями разобараться
            Ne_plot(model_dist, prior_dist, demographic_events, file_name='{}/NE_final.png'.format(dem_dir))


            # full_ground_true
            # full_predicted

            # Список уникальных классов
            classes = sorted(set([i for i in range(number_intervals)]))

            y_true = full_ground_true
            y_pred = full_predicted

            # Сохраняем все



            # Вычисление Precision, Recall, F1 для каждого класса (One-vs-Rest)
            precision = precision_score(y_true, y_pred, average=None, labels=classes)
            recall = recall_score(y_true, y_pred, average=None, labels=classes)
            f1 = f1_score(y_true, y_pred, average=None, labels=classes)

            cls_report =  classification_report(y_true, y_pred, labels=classes, output_dict=True)


            # Тут 
            
            precision_3 = precision_at_k_with_neighbors(y_true, y_pred, k=3)
            precision_5 = precision_at_k_with_neighbors(y_true, y_pred, k=5)

            if (precision_3 > 1.0) or (precision_5 > 1.0):
                save_arrays_to_file(y_true, y_pred, "errors.json") 
                assert 1 == 2


            data = {
                "model_name": troika,
                "model_path": checkpoint_path,
                "dem_name": dname,
                "model_dist": model_dist,
                "prior_dist": prior_dist,
                "Ne_pred": Ne_pred,
                "Ne_true": Ne_true,
                "confusion_matrix": final_confusion_matrix.tolist(),
                "eval_time": eval_time,
                "precision": precision.tolist(),
                "precision+-1(@3)": precision_3,
                "precision+-2(@5)": precision_5,
                "recall": recall.tolist(),
                "f1": f1.tolist(),
                "cls_report": cls_report,
            }

            with open('{}/result.json'.format(dem_dir), 'w', encoding='utf-8') as file:
                json.dump(data, file)