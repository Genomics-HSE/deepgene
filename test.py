import gin
import numpy as np
import torch
import pytorch_lightning
from deepgen.models import GruLabeler
from deepgen.utils import train_model, test_model


if __name__ == '__main__':
    gin.parse_config_file('configs/big_gru.gin')
    gru_model = GruLabeler()
    gru_model = gru_model.load_from_checkpoint(checkpoint_path="GRU_trained.ckpt").eval()
    
    for i in range(100):
        filename = "to_compare/" + str(i) + ".txt"
        genome = open(filename).read().strip()
        a = [int(num) for num in list(genome)]
        input_genome = torch.FloatTensor(a).unsqueeze(0)
        
        out = gru_model(input_genome)
        out.squeeze(0)
        torch.save("to_compare_results/" + str(i) + ".pt", out)