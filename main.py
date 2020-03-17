from models.gcn import create_gcn_model
from data_utils import load_data
from train import run

import random
import numpy as np
import torch
random.seed(17)
np.random.seed(17)
torch.manual_seed(17)

if __name__ == '__main__':
    data = load_data('ppi')
    model = create_gcn_model(data["train"])
    run(data, model, lr=0.01, weight_decay=5e-4, epochs=100000, patience=100, niter=5)
