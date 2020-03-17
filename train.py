import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
from numpy import mean, std
from sklearn.metrics import f1_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience, verbose, use_loss, use_f1, save_model):
        assert use_loss or use_f1, 'use loss or (and) acc'
        self.patience = patience
        self.use_loss = use_loss
        self.use_f1 = use_f1
        self.save_model = save_model
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def check(self, evals, model, epoch):
        if self.use_loss and self.use_f1:
            # For GAT, based on https://github.com/PetarV-/GAT/blob/master/execute_cora.py
            if evals['loss'] <= self.best_val_loss or evals['f1_score'] >= self.best_val_acc:
                if evals['loss'] <= self.best_val_loss and evals['f1_score'] >= self.best_val_acc:
                    if self.save_model:
                        self.state_dict = deepcopy(model.state_dict())
                self.best_val_loss = min(self.best_val_loss, evals['loss'])
                self.best_val_acc = max(self.best_val_acc, evals['f1_score'])
                self.counter = 0
            else:
                self.counter += 1
        elif self.use_loss:
            if evals['loss'] < self.best_val_loss:
                self.best_val_loss = evals['loss']
                self.counter = 0
                if self.save_model:
                    self.state_dict = deepcopy(model.state_dict())
            else:
                self.counter += 1
        elif self.use_f1:
            if evals['f1_score'] > self.best_val_acc:
                self.best_val_acc = evals['f1_score']
                self.counter = 0
                if self.save_model:
                    self.state_dict = deepcopy(model.state_dict())
            else:
                self.counter += 1
        stop = False
        if self.counter >= self.patience:
            stop = True
            if self.verbose:
                print("Stop training, epoch:", epoch)
            if self.save_model:
                model.load_state_dict(self.state_dict)
        return stop


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.binary_cross_entropy_with_logits(output, data.labels)
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        output = model(data)

    outputs = {}
    loss = F.binary_cross_entropy_with_logits(output, data.labels).item()
    predict = np.where(output.cpu().numpy() >= 0.5, 1, 0)
    score = f1_score(data.labels.data.cpu().numpy(),
                     predict, average='micro')
    outputs['loss'] = loss
    outputs['f1_score'] = score

    return outputs


def run(data, model, lr, weight_decay, epochs=200, niter=100, early_stopping=True, patience=10,
        use_loss=True, use_f1=False, save_model=False, verbose=False):
    # for GPU
    train_data, val_data, test_data = data["train"], data["val"], data["test"]
    train_data.to(device)
    val_data.to(device)
    test_data.to(device)

    val_acc_list = []
    test_acc_list = []

    for _ in tqdm(range(niter)):
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # for early stopping
        if early_stopping:
            stop_checker = EarlyStopping(patience, verbose, use_loss, use_f1, save_model)

        for epoch in range(1, epochs + 1):
            train(model, optimizer, train_data)
            train_evals = evaluate(model, train_data)
            val_evals = evaluate(model, val_data)

            if verbose:
                print('epoch: {: 4d}'.format(epoch),
                      'train loss: {:.5f}'.format(train_evals['loss']),
                      'train f1: {:.5f}'.format(train_evals['acc']),
                      'val loss: {:.5f}'.format(val_evals['loss']),
                      'val f1: {:.5f}'.format(val_evals['acc']))

            if early_stopping:
                if stop_checker.check(val_evals, model, epoch):
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        evals = evaluate(model, test_data)
        if verbose:
            for met, val in evals.items():
                print(met, val)

        # val_acc_list.append(evals['val_acc'])
        test_acc_list.append(evals['acc'])

    print(mean(test_acc_list))
    print(std(test_acc_list))
    return {
        # 'val_acc': mean(val_acc_list),
        'test_acc': mean(test_acc_list),
        'test_acc_std': std(test_acc_list)
    }
