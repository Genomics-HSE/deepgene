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
                                  dpi: int = 200,
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
