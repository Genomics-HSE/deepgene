import seaborn as sns

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import hamming_loss

from deepgene.data import limits
from .add_utils import from_probability_to_N, from_N_to_probability

patch_typeguard()


@typechecked
def make_coalescent_heatmap_chunk(y_pred: TensorType["hidden_size", "genome_length"],
                                  y_true: TensorType["genome_length"],
                                  dpi: int = 100,
                                  title: str = "Coalescent heatmap distribution"):
    
    print(f"\nGenome length is genome_length{len(y_true)}")
    # f = plt.figure(, dpi=dpi)
    f, ax = plt.subplots(1, 1, dpi=dpi)  # figsize=(200, 10)
    im0 = ax.imshow(y_pred, cmap='Wistia', aspect='auto')
    # ax.plot(torch.sum(averaged_data_tuple[1], dim=1), lw=1, c='black', label="True")
    ax.plot(y_true, lw=1, c='black', label="True")
    # ax.plot(torch.argmax(averaged_data_tuple[0], dim=0), lw=1, c='green', label="Model")

    # ax.plot(np.argmax(averaged_data_tuple[0], axis=0), lw=1, c='black', linestyle="--", label="Model")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)
    ax.legend()
    plt.suptitle(title, fontsize=20)
    ax.set_title("Softmax")
    plt.xlabel('site position')
    plt.ylabel('')
    ax.yaxis.set_ticks(np.arange(0, y_pred.shape[0], step=1))
    return f


@typechecked
def viz_heatmap(y_pred: TensorType[1, "genome_length", "hidden_size"],
                y_true: TensorType[1, "genome_length"]) -> None:
    y_pred = y_pred.squeeze(0)
    y_true = y_true.squeeze(0)
    step = 32768

    for j in range(0, len(y_true), step):
        f = make_coalescent_heatmap_chunk(y_pred[j:j + step].T, y_true[j:j + step])
        plt.show()

    return


def plot_comparison_with_true(model_name, averaged_data_tuple, dpi=200):
    f, ax = plt.subplots(1, 1, dpi=dpi)  # figsize=(200, 10)
    ax.plot(averaged_data_tuple[0], cmap='Wistia', aspect='auto')
    ax.plot(averaged_data_tuple[1], lw=1, c='black', )

    plt.suptitle("Comparison true TMRCA by {} model".format(model_name), fontsize=20)
    ax.set_title("Softmax")
    plt.xlabel('site position')
    plt.ylabel('')
    ax.yaxis.set_ticks(np.arange(0, averaged_data_tuple[0].shape[0], step=1))
    return f


@typechecked
def create_heatmap(y_pred: TensorType["genome_length", "hidden_size"],
                   y_true: TensorType["genome_length"], length: int):
    
    print(f"\nLen of dist data in heatmap: {len(y_true)}")

    # Конвертируем тензоры в numpy для seaborn
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    sns.set()  # Setting seaborn as default style even if use only matplotlib
    sns.set_context("paper")
    sns.set(rc={'figure.figsize': (8, 6),
                'axes.labelsize': 20,
                'axes.titlesize': 20,
                'figure.dpi': 50
                }
            )
    ax = sns.heatmap(y_pred[:length].T, cmap="viridis", yticklabels=5, xticklabels=1000)
    ax.plot(y_true[:length], c='red')
    # _ = ax.set(xlabel='Genome wide', ylabel='Coalescent time')
    _ = ax.set(xlabel='Позиция вдоль генома', ylabel='Интервал коалесенции')
    # ax.set_ylim([23, 0])
    return ax


@typechecked
def create_dist_plot(y_pred: TensorType["genome_length", "hidden_size"],
                   y_true: TensorType["genome_length"],
                   dpi: int = 50):

    plt.figure(dpi=dpi)
    
    y_pred = y_pred.sum(dim=[0]).detach().numpy()
    y_pred = y_pred / sum(y_pred)

    y_true = torch.bincount(y_true.long(), minlength=32).numpy()
    y_true = y_true / sum(y_true)

    print(f"\nLen of dist data {len(y_true)}")

    w = 0.35
    # print(len(y_pred))
    # print(len(y_true))
    # y_pred_one = y_pred.argmax(axis=1)
    # hl = hamming_loss(y_pred_one, y_true)

    plt.bar(np.arange(32) - w, y_pred, width=w, label=f'prediction', align='center', edgecolor='none')
    plt.bar(np.arange(32), y_true, width=w, label='true', align='center', edgecolor='none')

    plt.ylabel("Probability mass", fontsize=20)
    plt.xlabel("TMRCA", fontsize=18)
    plt.title("Comparison of predicted and true TMRCA distributions", fontsize=18)
    plt.legend(fontsize=20)
    return


@typechecked
def create_NE_plot(
    y_pred: TensorType["genome_length", "hidden_size"],
    y_true: TensorType["genome_length"],
    dpi: int = 50
):
    y_pred = y_pred.sum(dim=[0]).detach().numpy()
    y_pred = y_pred / sum(y_pred)

    y_true = torch.bincount(y_true.long(), minlength=32).numpy()
    y_true = y_true / sum(y_true)

    Ne1 = from_probability_to_N(y_pred, limits)
    Ne2 = from_probability_to_N(y_true, limits)

    x = np.linspace(0, 250*10**4, 10_0000)
    y = [25_000 for i in x]

    fig, ax = plt.subplots(dpi=dpi)


    # Создание цветовой карты
    cmap = plt.cm.get_cmap('viridis')
    # cmap = ['orange', 'blue', 'green']

    # Заполнение области разными цветами
    # for color, (x1, x2) in enumerate(zip(limits, limits[1:] +  [10000000 + 1])):
    #     # ax.fill_between(x, y, where=(x1 <= x) & (x <= x2), step="post" ,color = cmap(color))
    #     print(f"{color}: {x1} < x < {x2} L= {x2-x1}")
    #     ax.fill_between(x, y, where=(x1 <= x) & (x <= x2), step="post" ,color = cmap(8*color))
    #     # ax.text(x, 15_000, f"{round(p[min(color,6)][0],4)}")
    #     # ax.scatter([x1],[10_000], color = 'red')

    for i in range(len(limits) - 1):
        x1, x2 = limits[i], limits[i+1]
        # Чередуем серый и белый
        color = 'lightgray' if i % 2 == 0 else 'white'
        ax.fill_between(x, y, where=(x1 <= x) & (x <= x2), 
                        step='post', color=color, alpha=0.3)


    # Нанесение графика линии поверх заполненных областей

    # Настройка осей и отображение графика
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim(0, 50_000)
    plt.ylim(0, 25_000)
    # plt.title('Раскрашивание пространства по разным x')

    plt.step(limits, Ne2 + [Ne2[-1]], where='post', label="From data", color="blue")
    plt.step(limits, Ne1 + [Ne1[-1]], where='post', label="NN", color="red", linestyle='-.', linewidth = 0.5)

    # plt.xscale('log')
    # plt.yscale('log')

    plt.legend()
    plt.show()
    return
