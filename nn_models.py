import numpy as np
import pandas as pd
import sklearn
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, labels):
        self.labels = labels
        self.datas = datas
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        x_ = self.datas[index]
        y_ = self.labels[index]
        
        return x_, y_

class KH_MLP(torch.nn.Module):
    def __init__(self, input_sz=10, output_sz=2, use_bias=False):
        super(KH_MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_sz, 64, bias=use_bias)
        self.fc2 = torch.nn.Linear(64, 64, bias=use_bias)
        self.fc4 = torch.nn.Linear(64, 32, bias=use_bias)
        self.fc7 = torch.nn.Linear(32, output_sz, bias=use_bias)
        
    def forward(self, x):
        o = self.fc1(x)
        o = torch.nn.functional.relu(o)
        
        o = self.fc2(o)
        o = torch.nn.functional.relu(o)
        
        o = self.fc4(o)
        o = torch.nn.functional.relu(o)
        
        o = self.fc7(o)        
        return o


class KH_MLP_MLT(torch.nn.Module):
    def __init__(self, input_sz=10, output_sz=1, use_bias=False):
        super(KH_MLP_MLT, self).__init__()
        # backbone
        self.fc1 = torch.nn.Linear(input_sz, 64, bias=use_bias)
        self.fc2 = torch.nn.Linear(64, 64, bias=use_bias)

        # head
        self.turbo_t_fc1 = torch.nn.Linear(64, 32, bias=use_bias)
        self.turbo_t_fc2 = torch.nn.Linear(32, 1, bias=use_bias)

        self.fuel_fc1 = torch.nn.Linear(64, 32, bias=use_bias)
        self.fuel_fc2 = torch.nn.Linear(32, 1, bias=use_bias)

    def forward(self, x):
        o = self.fc1(x)
        o = torch.relu(o)

        o = self.fc2(o)
        o = torch.relu(o)

        # temp
        temp_o = self.turbo_t_fc1(o)
        temp_o = torch.relu(temp_o)
        temp_o = self.turbo_t_fc2(temp_o)
        
        # fuel
        fuel_o = self.fuel_fc1(o)
        fuel_o = torch.relu(fuel_o)
        fuel_o = self.fuel_fc2(fuel_o)

        batch_o = torch.cat([temp_o, fuel_o], dim=1)
        # print("AAAA", batch_o)
        # return [temp_o, fuel_o]
        return batch_o

# multi-head
class MODEL(torch.nn.Module):
    def __init__(self, input_sz1 = 8, input_sz2=2, output_sz = 4, use_bias=False):
        super(MODEL, self).__init__()

        self.fc1 = torch.nn.Linear(input_sz1, 64, bias=use_bias)
        self.fc2 = torch.nn.Linear(input_sz2, 16, bias=use_bias)
        self.fc3 = torch.nn.Linear(64, 64, bias=use_bias)

        self.temp_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.pn_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.nox_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.thc_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.cov_head = torch.nn.Linear(64+16, 32, bias=use_bias)

        self.fc4 = torch.nn.Linear(32, 1)

        # self.temp_bn = torch.nn.BatchNorm1d(32, track_running_stats=True)
        # self.pn_bn = torch.nn.BatchNorm1d(32, track_running_stats=True)
        # self.thc_bn = torch.nn.BatchNorm1d(32, track_running_stats=True)
        # self.nox_bn = torch.nn.BatchNorm1d(32, track_running_stats=True)
        # self.cov_bn = torch.nn.BatchNorm1d(32, track_running_stats=True)

        # self.whole_bn = torch.nn.BatchNorm1d(64+16)

    def forward(self, x_params, x_khs):
        # backbone
        o_params = self.fc1(x_params)
        o_params = torch.relu(o_params)
        o_params = self.fc3(o_params)
        o_params = torch.relu(o_params)

        o_khs = self.fc2(x_khs)
        o_khs = torch.relu(o_khs)

        o = torch.cat([o_khs, o_params], dim=1)
        # o = self.whole_bn(o)
        o = torch.relu(o)
        

        # temp
        temp_o = self.temp_head(o)
        temp_o = torch.relu(temp_o)
        # temp_o = self.temp_bn(temp_o)
        temp_o = self.fc4(temp_o)
        
        # PN
        pn_o = self.pn_head(o)
        pn_o = torch.relu(pn_o)
        # pn_o = self.pn_bn(pn_o)
        pn_o = self.fc4(pn_o)

        # nox
        nox_o = self.nox_head(o)
        nox_o = torch.relu(nox_o)
        # nox_o = self.nox_bn(nox_o)
        nox_o = self.fc4(nox_o)

        # thc
        thc_o = self.thc_head(o)
        thc_o = torch.relu(thc_o)
        # thc_o = self.thc_bn(thc_o)
        thc_o = self.fc4(thc_o)

        # cov
        cov_o = self.cov_head(o)
        cov_o = torch.relu(cov_o)
        # cov_o = self.cov_bn(cov_o)
        cov_o = self.fc4(cov_o)
        cov_o = torch.sigmoid(cov_o)

        batch_o = torch.cat([temp_o, pn_o, nox_o, thc_o, cov_o], dim=1)
        return batch_o

class MODEL_INCEPTION(torch.nn.Module):
    " add inspetion "
    def __init__(self, input_sz1 = 8, input_sz2=2, output_sz = 4, use_bias=False):
        super(MODEL_INCEPTION, self).__init__()

        self.fc1 = torch.nn.Linear(input_sz1 + input_sz2, 64, bias=use_bias)
        self.fc2 = torch.nn.Linear(input_sz2, 16, bias=use_bias)
        self.fc3 = torch.nn.Linear(64, 64, bias=use_bias)

        self.temp_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.pn_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.nox_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.thc_head = torch.nn.Linear(64+16, 32, bias=use_bias)
        self.cov_head = torch.nn.Linear(64+16, 32, bias=use_bias)

        self.fc4 = torch.nn.Linear(32, 1, bias=use_bias)
        # self.sigmoid = torch.nn.Sigmoid(32, 1)

    def forward(self, x_params, x_khs):
        # backbone
        # add inception
        x_params = torch.cat([x_params, x_khs], dim=1)
        o_params = self.fc1(x_params)
        o_params = torch.relu(o_params)
        o_params = self.fc3(o_params)
        o_params = torch.relu(o_params)

        o_khs = self.fc2(x_khs)
        o_khs = torch.relu(o_khs)

        o = torch.cat([o_khs, o_params], dim=1)

        # temp
        temp_o = self.temp_head(o)
        temp_o = torch.relu(temp_o)
        temp_o = self.fc4(temp_o)
        
        # PN
        pn_o = self.pn_head(o)
        pn_o = torch.relu(pn_o)
        pn_o = self.fc4(pn_o)

        # nox
        nox_o = self.nox_head(o)
        nox_o = torch.relu(nox_o)
        nox_o = self.fc4(nox_o)

        # thc
        thc_o = self.thc_head(o)
        thc_o = torch.relu(thc_o)
        thc_o = self.fc4(thc_o)

        # cov
        cov_o = self.cov_head(o)
        # cov_o = torch.relu(cov_o)
        cov_o = self.fc4(cov_o)

        batch_o = torch.cat([temp_o, pn_o, nox_o, thc_o, cov_o], dim=1)
        return batch_o

class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_true, y_pred):
        sample_weight = torch.Tensor(np.asarray([0.25, 0.25, 0.25, 0.25]))
        # sample_weight = -torch.log(y_true)
        return torch.mean(sample_weight * torch.pow((y_true - y_pred), 2))

class MultiTaskLossWrapper(torch.nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = torch.nn.Parameter(torch.zeros(self.task_num))
        self.logloss = torch.nn.BCELoss()
    
    def forward(self,  x1, x2, targets):
        outputs = self.model(x1, x2)

        # temp_o, pn_o, nox_o, thc_o, cov_o
        # temp
        precision0 = torch.exp(-self.log_vars[0])
        loss = torch.sum(precision0 * (targets[:, 0] - outputs[:, 0]) ** 2. + self.log_vars[0], -1)
        # pn
        precision1 = torch.exp(-self.log_vars[1])
        loss += torch.sum(precision1 * (targets[:,1] - outputs[:,1]) ** 2. + self.log_vars[1], -1)
        # nox
        precision2 = torch.exp(-self.log_vars[2])
        loss += torch.sum(precision2 * (targets[:,2] - outputs[:,2]) ** 2. + self.log_vars[2], -1)
        # thc
        precision3 = torch.exp(-self.log_vars[3])
        loss += torch.sum(precision3 * (targets[:,3] - outputs[:,3]) ** 2. + self.log_vars[3], -1)
        # cov
        precision4 = torch.exp(-self.log_vars[4])
        # loss += torch.sum(precision4 * (targets[4] - outputs[4]) ** 2. + self.log_vars[4], -1)
        # print(targets[:,4], outputs[:,4])
        loss += precision4 * self.logloss(outputs[:,4], targets[:,4] )

        return loss, self.log_vars.data.tolist()


if __name__ == "__main__":
    y1 = [0,1,2,3]
    y2 = [2,4,6,8]
