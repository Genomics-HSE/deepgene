import gin
import numpy as np
import torch
import pytorch_lightning
from deepgen.models import GruLabeler
from deepgen.utils import train_model, test_model
from tqdm import tqdm


if __name__ == '__main__':
    gin.parse_config_file('configs/big_gru.gin')
    #gru_model = GruLabeler()
    gru_model = GruLabeler.load_from_checkpoint(checkpoint_path="output_ms/GRU.ckpt").eval()
    
    with torch.no_grad():
        for i in tqdm(range(100)):
            filename = "to_compare/" + str(i) + ".txt"
            genome = open(filename).read().strip()
            a = [int(num) for num in list(genome)]
            input_genome = torch.FloatTensor(a).unsqueeze(0)
    
            out = gru_model(input_genome)
            out.squeeze(0)
            torch.save(out, "output_ms/to_compare_results/" + str(i) + ".pt")