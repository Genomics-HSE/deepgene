import seaborn as sns

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

patch_typeguard()


@typechecked
def make_coalescent_heatmap_chunk(y_pred: TensorType["hidden_size", "genome_length"],
                                  y_true: TensorType["genome_length"],
                                  dpi: int = 100,
                                  title: str = "Coalescent heatmap distribution"):
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
    sns.set()  # Setting seaborn as default style even if use only matplotlib
    sns.set_context("paper")
    sns.set(rc={'figure.figsize': (11.7, 8.27),
                'axes.labelsize': 20,
                'axes.titlesize': 20,
                'figure.dpi': 100
                }
            )
    ax = sns.heatmap(y_pred[:length].T, cmap="viridis", yticklabels=5, xticklabels=1000)
    ax.plot(y_true[:length], c='red')
    _ = ax.set(xlabel='Genome wide', ylabel='Coalescent time')
    # ax.set_ylim([23, 0])
    return ax


@typechecked
def create_dist_plot(y_pred: TensorType["genome_length", "hidden_size"],
                   y_true: TensorType["genome_length"]):

    y_pred = y_pred.sum(dim=[0]).detach().numpy()
    y_pred = y_pred / sum(y_pred)

    y_true = torch.bincount(y_true.long(), minlength=32).numpy()
    y_true = y_true / sum(y_true)

    w = 0.35
    plt.bar(np.arange(32) - w, y_pred, width=w, label='prediction', align='center', edgecolor='none')
    plt.bar(np.arange(32), y_true, width=w, label='true', align='center', edgecolor='none')

    plt.ylabel("Probability mass", fontsize=20)
    plt.xlabel("TMRCA", fontsize=18)
    plt.title("Comparison of predicted and true TMRCA distributions", fontsize=18)
    plt.legend(fontsize=20)
    return
