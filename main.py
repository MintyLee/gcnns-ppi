from models.gcn import create_gcn_model
from models.gat import create_gat_model
from models.sgc import create_sgc_model
from models.gfnn import create_gfnn_model
from models.masked_gcn import create_masked_gcn_model
from models.appnp import create_appnp_model
from data.data import load_data
from train import run
from utils import preprocess_features

import random
import numpy as np
import torch
random.seed(17)
np.random.seed(17)
torch.manual_seed(17)

if __name__ == '__main__':
    data = load_data('ppi')
    data.features = preprocess_features(data.features)
    model, optimizer = create_gcn_model(data)
    run(data, model, optimizer, epochs=100000, patience=100, niter=5)
