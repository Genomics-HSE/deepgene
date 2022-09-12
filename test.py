from os.path import join
import gin
import numpy as np
import torch
import pytorch_lightning
from deepgene.models import GruLabeler
from deepgene.data import DatasetXY
from deepgene.utils import train_model, test_model
from tqdm import tqdm


if __name__ == '__main__':
    gin.parse_config_file('configs/big_gru.gin')
    # gru_model = GruLabeler()
    gru_model_const = GruLabeler.load_from_checkpoint(checkpoint_path="o1/GRU.ckpt").eval()
    gru_model_ms = GruLabeler.load_from_checkpoint(checkpoint_path="o2/GRU.ckpt").eval()
    dataset = DatasetXY()

    with torch.no_grad():
        path = "o1/const"
        for i, data in tqdm(enumerate(dataset.test_dataloader())):
            x, y_true = data
            y_pred_1 = gru_model_const(x)
            y_pred_2 = gru_model_ms(x)
            torch.save(x, join(path, "x_" + str(i) + ".pt"))
            torch.save(y_true, join(path, "y_true_" + str(i) + ".pt"))
            torch.save(y_pred_1, join(path, "1_y_pred_" + str(i) + ".pt"))
            torch.save(y_pred_2, join(path, "2_y_pred_" + str(i) + ".pt"))

        # "ms model on const data"
        # path = "../Output/May5_final_weights/const-2"
        # for i in tqdm(range(100)):
        #     x_path = join(path, "x_" + str(i) + ".pt")
        #     x = torch.load(x_path)
        #     out = gru_model_ms(x)
        #     torch.save(out, join(path, "ms-2", "y_pred_" + str(i) + ".pt"))