"""
**Libraries, classes, and functions for ML training**
"""
# for math
import numpy as np
from numpy import vstack
import math
import copy

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import interpolate
from sklearn import metrics
import pickle

# for ML (PyTorch)
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import SGD, Adam
# optimiser
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR

# NODE
adjoint = False
if adjoint == True:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# for timing
import time
import nvidia_smi


# # Define classes and functions

# ## Select cases
def get_casename(casenum):
    if casenum == 1:  # F4R32
        casename  = 'F4R32'
    elif casenum == 12:  # F4R64
        casename  = 'F4R64'
    elif casenum == 13:  # F2R32
        casename  = 'F2R32'
    elif casenum == 101:  # pendulum
        casename  = 'pendulum'
    else:
        raise Exception(f"Case number {casenum} is invalid.")
    return casename

# ## Class: Dataset definition
class SSTDataset(Dataset):
    # load the dataset
    def __init__(self, datafull_IP, datafull_OP):   
        # store all the inputs and outputs
        self.X = datafull_IP
        self.Y = datafull_OP

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# ## Class: Initialize models of different cases
class initialize():
    
    def pendulum_unforced():
        
        input_dim = 3
        output_dim = 3
        num_layers = 1 # 10
        neurons = 60 # 40 
        activation = nn.SiLU()
        lr = 5e-2 # 1e-1 for 1000-1250 data points
        gamma = 0.99 # 0.9999 for iteration step 1e-1 for 1000-1250 data points
        
        return input_dim, output_dim, num_layers, neurons, activation, lr, gamma
    
    def pendulum_forced():
        
        input_dim = 3
        output_dim = 3
        num_layers = 10 # 10
        neurons = 20 # 40 
        activation = nn.SiLU()
        lr = 5e-2 # 1e-1 for 1000-1250 data points
        gamma = 0.99 # 0.9999 for iteration step 1e-1 for 1000-1250 data points
        
        return input_dim, output_dim, num_layers, neurons, activation, lr, gamma
        
    def RANS():
        
        input_dim = 4
        output_dim = 4
        num_layers = 10 # 10
        neurons = 40 # 40 
        activation = nn.SiLU()
        lr = 5e-2 # 1e-1 for 1000-1250 data points
        gamma = 0.99 # 0.9999 for iteration step 1e-1 for 1000-1250 data points
        
        # input_dim = 4
        # output_dim = 4
        # num_layers = 10 # 10
        # neurons = 40 # 40 
        # activation = nn.LeakyReLU(0.1)
        # lr = 5e-2 # 1e-1 for 1000-1250 data points
        # gamma = 0.99 # 0.9999 for iteration step 1e-1 for 1000-1250 data points
        
        return input_dim, output_dim, num_layers, neurons, activation, lr, gamma

    
# ## Class: Network model definition

# ### Singe layer MLP
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, nneurons, leakyReLU_alpha, n_outputs, drp_in, drp_hd):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.drpout1 = nn.Dropout(p=drp_in, inplace=False)  # inplace=False by default
        self.hidden1 = nn.Linear(n_inputs, nneurons)
        nn.init.xavier_normal_(self.hidden1.weight, gain=nn.init.calculate_gain('leaky_relu',leakyReLU_alpha))
        self.act1 = nn.LeakyReLU(leakyReLU_alpha)
        
#         # second hidden layer and output
#         self.hidden2 = nn.Linear(nneurons, nneurons)
#         nn.init.xavier_normal_(self.hidden2.weight, gain=nn.init.calculate_gain('leaky_relu',leakyReLU_alpha))
#         self.act2 = nn.LeakyReLU(leakyReLU_alpha)
        
        # second hidden layer and output
        self.drpout3 = nn.Dropout(p=drp_hd, inplace=False)
        self.hidden3 = nn.Linear(nneurons, n_outputs, bias=True)
        nn.init.xavier_normal_(self.hidden3.weight, gain=nn.init.calculate_gain('leaky_relu',leakyReLU_alpha))

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.drpout1(X)
        X = self.hidden1(X)
        X = self.act1(X)
#         # second hidden layer
#         X = self.hidden2(X)
#         X = self.act2(X)
        # second hidden layer and output
        X = self.drpout3(X)
        X = self.hidden3(X)
        
        return X

def model_SingleMLP(n_inputs, n_outputs, n_per_layer, drp_in, drp_hd, lkyReLU_alpha, device_name):
    model = MLP(n_inputs, n_per_layer, lkyReLU_alpha, n_outputs, drp_in, drp_hd).to(device_name)
    print(model)
    print("Single layer MLP with number of parameters = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# ### DenseNet
# Normal DenseNet block w/o batchnorm
class DenseBlock(nn.Module):
    def __init__(self, n1, n2, leakyReLU_alpha):
        super(DenseBlock, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(n1, n2), 
                                        nn.LeakyReLU(leakyReLU_alpha))
    def forward(self, x):
        identity = x
        out = self.sequential(x)
        out = torch.cat((out, identity), 1)
        
        return out

# DenseNet block with batchnorm
class DenseBlockBatchNorm(nn.Module):
    def __init__(self, n1, n2, leakyReLU_alpha):
        super(DenseBlockBatchNorm, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(n1, n2), 
                                        nn.BatchNorm1d(n2, track_running_stats=True),  
                                        nn.LeakyReLU(leakyReLU_alpha))
    def forward(self, x):
        identity = x
        out = self.sequential(x)
        out = torch.cat((out, identity), 1)
        
        return out

def model_DenseNet(n_inputs, n_outputs, n_per_layer, n_layers, drp_in, drp_hd, lkyReLU_alpha, device_name):
    # Input layer
    if drp_in >= 0 and drp_hd >= 0:        # dropout instead of batchnorm
        block_drp = nn.Dropout(p=drp_in, inplace=False)
        block_ip  = DenseBlock(n_inputs, n_per_layer, lkyReLU_alpha)
        model     = nn.Sequential(block_drp, block_ip)
    else:                                  # batchnorm
        block_ip = DenseBlockBatchNorm(n_inputs, n_per_layer, lkyReLU_alpha)
        model    = nn.Sequential(block_ip)
    n_in = n_inputs + n_per_layer
    # Hidden layers
    for i in range(n_layers):
        if drp_in >= 0 and drp_hd >= 0:        # dropout instead of batchnorm
            block_drp = nn.Dropout(p=drp_hd, inplace=False)
            block_hd  = DenseBlock(n_in, n_per_layer, lkyReLU_alpha)
            model     = nn.Sequential(model, block_drp, block_hd)
        else:                                  # batchnorm
            block_hd = DenseBlockBatchNorm(n_in, n_per_layer, lkyReLU_alpha)
            model    = nn.Sequential(model, block_hd)
        n_in = n_in + n_per_layer
    # Output layer
    block_op = nn.Linear(n_in, n_outputs)
    if drp_in >= 0 and drp_hd >= 0:        # dropout instead of batchnorm
        block_drp = nn.Dropout(p=drp_hd, inplace=False)
        model     = nn.Sequential(model, block_drp, block_op).to(device_name)
    else:                                  # no need for batchnorm at o/p layer
        model = nn.Sequential(model, block_op).to(device_name)

    if n_layers < 3: print(model)
    print(f"DenseNet with number of parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


# ### ResNet
# Normal ResNet block w/o batchnorm
class ResNetBlock(nn.Module):
    def __init__(self, n1, n2, leakyReLU_alpha):
        super(ResNetBlock, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(n1, n2), 
                                        nn.LeakyReLU(leakyReLU_alpha))
    def forward(self, x):
        residual = x
        out = self.sequential(x)
        out += residual
        return out

# ResNet block with batchnorm
class ResNetBlockBatchNorm(nn.Module):
    def __init__(self, n1, n2, leakyReLU_alpha):
        super(ResNetBlockBatchNorm, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(n1, n2), 
                                        nn.BatchNorm1d(n2, track_running_stats=True),  
                                        nn.LeakyReLU(leakyReLU_alpha))
    def forward(self, x):
        residual = x
        out = self.sequential(x)
        out += residual
        return out

def model_ResNet(n_inputs, n_outputs, n_per_layer, n_layers, drp_in, drp_hd, lkyReLU_alpha, device_name):
    # Input layer
    block_ip = nn.Linear(n_inputs, n_per_layer)
    if drp_in >= 0 and drp_hd >= 0:        # dropout instead of batchnorm
        block_drp = nn.Dropout(p=drp_in, inplace=False)
        model     = nn.Sequential(block_drp, block_ip, nn.LeakyReLU(lkyReLU_alpha))
    else:                                  # batchnorm
        model    = nn.Sequential(block_ip, nn.LeakyReLU(lkyReLU_alpha))
    n_in = n_per_layer
    # Hidden layers
    for i in range(n_layers):
        if drp_in >= 0 and drp_hd >= 0:        # dropout instead of batchnorm
            block_drp = nn.Dropout(p=drp_hd, inplace=False)
            block_hd  = ResNetBlock(n_in, n_per_layer, lkyReLU_alpha)
            model     = nn.Sequential(model, block_drp, block_hd)
        else:                                  # batchnorm
            block_hd = ResNetBlockBatchNorm(n_in, n_per_layer, lkyReLU_alpha)
            model    = nn.Sequential(model, block_hd)
    # Output layer
    block_op = nn.Linear(n_in, n_outputs)
    if drp_in >= 0 and drp_hd >= 0:        # dropout instead of batchnorm
        block_drp = nn.Dropout(p=drp_hd, inplace=False)
        model     = nn.Sequential(model, block_drp, block_op).to(device_name)
    else:                                  # no need for batchnorm at o/p layer
        model = nn.Sequential(model, block_op).to(device_name)

    if n_layers < 3: print(model)
    print(f"ResNet with number of parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model

# MLP model within NODE
class ODEFunc_MLP(nn.Module):
    """ Func to map [t, y] -> dydt. Here, we do not use t for the mapping, even though it is used as an input to the model (required by NODE).
    """
    def __init__(self, input_dim, output_dim, num_layers=1, neurons=20):
        super(ODEFunc_MLP, self).__init__()

        activation=nn.SiLU()
        # layers
        layers = [nn.Linear(input_dim, neurons)]
        for i in range(num_layers-1):
            layers.append(activation)
            layers.append(nn.Linear(neurons, neurons))

        layers.append(activation)
        layers.append(nn.Linear(neurons, output_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            #if isinstance(m, nn.Linear):
            #    nn.init.normal_(m.weight, mean=0, std=0.1)
            #    nn.init.constant_(m.bias, val=0)
                
            if isinstance(m, nn.Linear):
                std = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-1.0 * std, 1.0 * std)
                m.bias.data.zero_()

    def forward(self, t, y):
        # Note that t is not used as an input to the model
        return self.net(y)

def model_ODEFunc_MLP(n_inputs, n_outputs, n_per_layer, n_layers, device_name):
    model = ODEFunc_MLP(n_inputs, n_outputs, n_layers, n_per_layer).to(device_name)
    # print(model)
    print("MLP with number of parameters = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

# ### LSTM
class LSTMmodel(nn.Module):
    def __init__(self, input_size, output_size, input_seq_len, hidden_size, n_lstm_layers, n_layers, nneurons, drp_in, drp_hd, lkyReLU_alpha, device_name):
        super(LSTMmodel, self).__init__()
        self.n_lstm_layers = n_lstm_layers      # number of layers
        self.hidden_size = hidden_size    # hidden state
        self.seq_len = input_seq_len    # length of sequence
        self.n_layers = n_layers       # number of ResNet layers
        self.device_name = device_name
        
        # LSTM layers
        self.lstm   = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=n_lstm_layers, batch_first=True) # lstm
        self.relu   = nn.LeakyReLU(lkyReLU_alpha)
        # Dense layers
        # fully connected: i/p
        if drp_in > 0:
            block_drp   = nn.Dropout(p=drp_in, inplace=False)
            self.fc_1   = nn.Sequential( block_drp, nn.Linear(hidden_size*input_seq_len, nneurons),
                                       nn.LeakyReLU(lkyReLU_alpha) )
        else:
            self.fc_1   = nn.Sequential( nn.Linear(hidden_size*input_seq_len, nneurons),
                                       nn.LeakyReLU(lkyReLU_alpha) )
        # fully connected: hidden layers
        if drp_hd > 0:
            block_drp   = nn.Dropout(p=drp_hd, inplace=False)
            self.resblk = nn.Sequential( block_drp, nn.Linear(nneurons, nneurons), 
                                        nn.LeakyReLU(lkyReLU_alpha) )
        else:
            self.resblk = nn.Sequential( nn.Linear(nneurons, nneurons), 
                                        nn.BatchNorm1d(nneurons, track_running_stats=True),  
                                        nn.LeakyReLU(lkyReLU_alpha) )
        # fully connected: o/p
        if drp_hd > 0:
            block_drp   = nn.Dropout(p=drp_hd, inplace=False)
            self.fc_out = nn.Sequential( block_drp, nn.Linear(nneurons, output_size) )
        else:
            self.fc_out = nn.Linear(nneurons, output_size) # fully connected last layer
    
    def forward(self,x):
        # hidden state [num_layers, batch_size, hiddenstate_size]
        hidden_state = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_size).to(self.device_name, non_blocking=False)
        # internal state [num_layers, batch_size, hiddenstate_size]
        cell_state   = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_size).to(self.device_name, non_blocking=False)
        hidden       = (hidden_state, cell_state) # Pytorch needs them to be a tuple
        # Propagate input through LSTM
        output, (hidden_state, cell_state) = self.lstm(x, hidden) # lstm with input and [hidden and internal states]
        output = output.reshape(output.shape[0],-1)
        out = self.relu(output)
        
        # MLP (ResNet)
        out = self.fc_1(out) #first Dense
        residual = out
        for i in range(self.n_layers-1):  # remaining dense layers
            out = self.resblk(out)
            out += residual
            residual = out
        out = self.fc_out(out) #Final Output
        return out

def model_LSTM(input_size, output_size, input_seq_len, hidden_size, n_lstm_layers, n_layers, nneurons, drp_in, drp_hd, lkyReLU_alpha, device_name):
    model = LSTMmodel(input_size, output_size, input_seq_len, hidden_size, n_lstm_layers, n_layers, nneurons, drp_in, drp_hd, lkyReLU_alpha, device_name).to(device_name)
    # print(model)
    print("Multi-layer LSTM block + single dense layer: number of parameters = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# ### Define model from the list
def defNNmodel(model_name, n_inputs=1, n_outputs=1, seq_len=1, hidden_size=1, n_lstm_layers=1, 
               n_layers=1, n_per_layer=1, drp_in=0, drp_hd=0, lkyReLU_alpha=0.1, device_name='cpu'):
    if   model_name == 'SingleMLP':
        model = model_SingleMLP(n_inputs, n_outputs, n_per_layer, drp_in, drp_hd, lkyReLU_alpha, device_name)
    elif model_name == 'DenseNet' :
        model = model_DenseNet(n_inputs, n_outputs, n_per_layer, n_layers, drp_in, drp_hd, lkyReLU_alpha, device_name)
    elif model_name == 'ResNet'   :
        model = model_ResNet(n_inputs, n_outputs, n_per_layer, n_layers, drp_in, drp_hd, lkyReLU_alpha, device_name)
    elif model_name == 'NODE_MLP'   :
        model = model_ODEFunc_MLP(n_inputs, n_outputs, n_per_layer, n_layers, device_name)
    elif model_name == 'LSTM'     :
        model = model_LSTM(n_inputs, n_outputs, seq_len, hidden_size, n_lstm_layers, n_layers, n_per_layer, drp_in, drp_hd, lkyReLU_alpha, device_name)
    else:
        print("Enter valid model name...")
    return model


# ## Function: Scaling dataset
# prepare dataset with input and output scalers, can be none
def get_scaleddataset(train_IP, train_OP, test_IP, test_OP, input_scaler, output_scaler):
    # scale inputs
    if input_scaler is not None:
        # fit scaler
        input_scaler.fit(train_IP)
        # transform training dataset
        train_IP = input_scaler.transform(train_IP)
        # transform test dataset
        test_IP = input_scaler.transform(test_IP)
    if output_scaler is not None:
        # fit scaler on training dataset
        output_scaler.fit(train_OP)
        # transform training dataset
        train_OP = output_scaler.transform(train_OP)
        # transform test dataset
        test_OP = output_scaler.transform(test_OP)
    return train_IP, train_OP, test_IP, test_OP, input_scaler, output_scaler


# ## Function: Prepare the dataset
def prepare_data(datafull_IP, datafull_OP, n_val, batch_size):
    # load the dataset
    dataset = SSTDataset(datafull_IP, datafull_OP)
    # calculate split
    train, val = dataset.get_splits(n_val)
    # prepare data loaders
    # ============cuda settings: pinned memory============
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)
    val_dl   = DataLoader(val  , batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_dl, val_dl

# ## Function: Custom loss function
def PHYLoss_sumdt(output, target, reduction='mean'):
    """Sum of time derivative of energies should be <0"""
    loss = (torch.sum(output[:,0:3], axis=-1) - torch.sum(target[:,0:3], axis=-1))**2
    if reduction=='none':
        return loss
    elif reduction=='mean':
        return torch.mean(loss)
    elif reduction=='sum':
        return torch.sum(loss)
    else:
        raise Exception('Invalide value for reduction.')

# ## Function: Running average
class running_mean():
  
    def __init__(self):
        self.total = 0
        self.count = 0

    def add(self,value):
        self.total+=value
        self.count+=1

    def value(self):
        return self.total/self.count if self.count else 0


# ## Function: NN model running average
class running_mean_model():
    def __init__(self, modelAvg):
        self.model = copy.deepcopy(modelAvg)
        self.SDsum = self.model.state_dict()
        self.count = 1

    def update_parameters(self,modelRunSD):
        for key in self.SDsum:
            self.SDsum[key] = self.SDsum[key] + modelRunSD[key]
        self.count+=1
        
    def ensemble_model(self):
        for key in self.SDsum:
            self.SDsum[key] = self.SDsum[key]/self.count
        self.model.load_state_dict(self.SDsum)
        return self.model


# ## Function: Train the model
def train_model(datatrain_IP, datatrain_OP, n_val, batch_size, model, nepoch, lrate, reg_factor, reg_type, device_name,
                epoch_in=0, train_loss_history=(), val_loss_history = ()):
    # print GPU details
    if device_name.type == 'cuda':
        # print(f'CUDA version: {torch.version.cuda}')
        # print(f'Default current GPU used: {torch.cuda.current_device()}')
        # print(f'Device count: {torch.cuda.device_count()}')
        # for i in range(torch.cuda.device_count()):
        #     print(f'Device name: {torch.cuda.get_device_name(i)}')
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)     # hard coded device index as 0
        # print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
        # print(f'Cached:    {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')
    # define the loss_func & optimizer
    loss_func = nn.MSELoss() # (reduction='sum')
    running_loss = 0.0;
    optimizer = optim.NAdam(model.parameters(), lr=lrate)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # manual weight regularization
    reg_flag = False
    if reg_type=='L1':
        reg_crit = nn.L1Loss(size_average=False)
        reg_flag = True
    elif reg_type=='L2':
        reg_crit = nn.MSELoss(size_average=False)
        reg_flag = True
    # randomly split into train & validation 
    train_dl, val_dl = prepare_data(datatrain_IP, datatrain_OP, n_val, batch_size)
    # enumerate epochs
    for epoch in range(epoch_in,nepoch):
        # ===========================TRAININING=========================================
        model.train()
        
        # ============cuda settings============
        gpuUse     = running_mean()
        gpuMem     = running_mean()
        t0         = time.time()
        
        # =====uncomment following for random cross validation======
#         # randomly split into train & validation 
#         train_dl, val_dl = prepare_data(datatrain_IP, datatrain_OP, n_val, batch_size)
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # ============cuda settings============
            inputs  = inputs.to(device_name, non_blocking=False)
            targets = targets.to(device_name, non_blocking=False)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = loss_func(yhat, targets) # / (len(inputs)*targets.shape[1])
            # loss = loss_func(yhat, targets) + PHYLoss_sumdt(yhat, targets)
            # weight regularization
            if reg_flag:
                # manual regularization
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += reg_crit(param, torch.zeros_like(param))
                loss += reg_factor*reg_loss
            # find gradient of loss w.r.t tensors
            loss.backward()
            # update model weights
            optimizer.step()
            
            # calculate statistics
            running_loss += loss.item()
            
            if device_name.type == "cuda": 
                gpuUse.add(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
                gpuMem.add(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).memory)

        # calculate average mse amongst all mini-batches in current epoch
        train_loss_history = train_loss_history + (running_loss/(i+1),)        
        running_loss = 0.0
        
        # ===========================VALIDATION=========================================
        model.eval()
        # validation loss
        # ============cuda settings============
        mse, _, _ = evaluate_model(val_dl, model, device_name)
        val_loss_history = val_loss_history + (mse,)

        if epoch%10 == 0: 
            print(f"Epoch = {(epoch+1):4d}/{nepoch}, dt = {time.time()-t0:4.3f}s, "
                  f"gpu = {gpuUse.value():3.1f}%, gpu-mem = {gpuMem.value():3.3f}%, "
                  f"training loss = {train_loss_history[-1]:1.3e}, validation loss = {val_loss_history[-1]:1.3e}")
        
    return train_loss_history, val_loss_history, optimizer, epoch, loss

# ## Function: Train the model & compute ensemble automatically
# Compute avg loss between a particular epoch window
# Compute model weight ensemble based on above avg loss after a particular epoch
def train_model_ensemble_auto(datatrain_IP, datatrain_OP, n_val, batch_size, model, modelEnsmb, 
                              nepoch, lrate, reg_factor, reg_type, device_name,
                              epoch_mdlens_start, epoch_avgls,
                              epoch_in=0, train_loss_history=(), val_loss_history = ()):
    # print GPU details
    if device_name.type == 'cuda':
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)     # hard coded device index as 0
    # define the loss_func & optimizer
    loss_func = nn.MSELoss()
    running_loss = 0.0;
    optimizer = optim.NAdam(model.parameters(), lr=lrate)
    # manual weight regularization
    reg_flag = False
    if reg_type=='L1':
        reg_crit = nn.L1Loss(size_average=False)
        reg_flag = True
    elif reg_type=='L2':
        reg_crit = nn.MSELoss(size_average=False)
        reg_flag = True
    # randomly split into train & validation 
    train_dl, val_dl = prepare_data(datatrain_IP, datatrain_OP, n_val, batch_size)
    # initialize running mean
    running_flag = 0
    if nepoch < epoch_mdlens_start:
        raise Exception(f"Number of epochs is less than ensemble starting epoch ({nepoch} < {epoch_mdlens_start}).")
    mse, _, _ = evaluate_model(val_dl, model, device_name)
    print(f'validation loss (model init) = {mse:1.3e}')
    mse, _, _ = evaluate_model(val_dl, modelEnsmb, device_name)
    print(f'validation loss (ensemble model init) = {mse:1.3e}')

    # enumerate epochs
    for epoch in range(epoch_in,nepoch):
        # ===========================TRAININING=========================================
        model.train()
        
        # ============cuda settings============
        gpuUse     = running_mean()
        gpuMem     = running_mean()
        t0         = time.time()
        
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # ============cuda settings============
            inputs  = inputs.to(device_name, non_blocking=False)
            targets = targets.to(device_name, non_blocking=False)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = loss_func(yhat, targets)
            # loss = loss_func(yhat, targets) + PHYLoss_sumdt(yhat, targets)
            # weight regularization
            if reg_flag:
                # manual regularization
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += reg_crit(param, torch.zeros_like(param))
                loss += reg_factor*reg_loss
            # find gradient of loss w.r.t tensors
            loss.backward()
            # update model weights
            optimizer.step()
            
            # calculate statistics
            running_loss += loss.item()
            
            if device_name.type == "cuda": 
                gpuUse.add(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
                gpuMem.add(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).memory)

        # calculate average mse amongst all mini-batches in current epoch
        train_loss_history = train_loss_history + (running_loss/(i+1),)        
        running_loss = 0.0
        
        # ===========================VALIDATION=========================================
        model.eval()
        # validation loss
        # ============cuda settings============
        mse, _, _ = evaluate_model(val_dl, model, device_name)
        val_loss_history = val_loss_history + (mse,)
        
        if epoch%10 == 0: 
            print(f"Epoch = {(epoch+1):4d}/{nepoch}, dt = {time.time()-t0:4.3f}s, "
                  f"gpu = {gpuUse.value():3.1f}%, gpu-mem = {gpuMem.value():3.3f}%, "
                  f"training loss = {train_loss_history[-1]:1.3e}, validation loss = {val_loss_history[-1]:1.3e}")
        
        # =============================Ensemble goes here================================
        if epoch == epoch_mdlens_start and running_flag==0:
            modelEnsmb.load_state_dict(model.state_dict())
            # running mean for ensemble model
            modelRunningMean = running_mean_model(modelEnsmb)
            running_flag = 1
            ensmb_errval = np.mean(train_loss_history[epoch_avgls[0]:epoch_avgls[1]])
        if epoch>epoch_mdlens_start and train_loss_history[-1]<=ensmb_errval:
            modelRunningMean.update_parameters(model.state_dict())
    
    # forward pass ensemble model so that if batch normalization is required
    # modelTemp = modelRunningMean.ensemble_model()
    # mse, _, _ = evaluate_model(val_dl, modelEnsmb, device_name)
    # print(f'validation loss (ensemble model before) = {mse:1.3e}')
    # modelEnsmb = copy.deepcopy(modelTemp)
    # mse, _, _ = evaluate_model(val_dl, modelEnsmb, device_name)
    # print(f'validation loss (ensemble model after) = {mse:1.3e}')
    # mse, _, _ = evaluate_model(val_dl, modelTemp, device_name)
    # print(f'validation loss (ensemble model true) = {mse:1.3e}')
    
    modelEnsmb = copy.deepcopy(modelRunningMean.ensemble_model())
    for i, (inputs, targets) in enumerate(train_dl):
        inputs  = inputs.to(device_name, non_blocking=False)
        modelEnsmb(inputs)
        
    print(f'Init epoch: {epoch_mdlens_start}; Window epoch: {epoch_avgls}; avg. loss: {ensmb_errval}')
    return train_loss_history, val_loss_history, optimizer, epoch, loss, modelRunningMean.count

# ## Function: Evaluate the model
def evaluate_model(test_dl, model, device_name):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # ============cuda settings============
        inputs = inputs.to(device_name, non_blocking=False)
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse, actuals, predictions

# ## Function: Train a NODE model
def train_NODEmodel(datatrain_IP, datatrain_OP, train_dydt, train_time, n_val, batch_size, model, NODE_method, NODE_rtol, NODE_atol, 
                    nepoch, lrate, gamma, device_name, 
                    plot_itr=False, plot_epoch=100, data_ip_varnames=[], data_dy_varnames=[], 
                    epoch_in=1, train_loss_history=[], val_loss_history=[], epoch_history=[], lrate_history=[]):
    
    # define the loss_func & optimizer
    loss_func = nn.MSELoss()
    # optimizer = optim.NAdam(model.parameters(), lr=lrate)
    optimizer = optim.Adamax(model.parameters(), lrate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma) # learning rate (exploration) exponentially decays over time
    
    # time sequence used for training
    # train_t used for training the model has the same limits ([0,t_seq]) for all batches
    # this can be seen as showing the model different initial conditions (at different regimes) of the same system
    train_t = torch.from_numpy(train_time[0,:]).to(device_name, non_blocking=False)

    # validation data for each epoch
    # Note: Just use the first sequence as the data will not be strictly increasing or decreasing for mixed cases
    val_t = torch.from_numpy(train_time[0,:]).to(device_name, non_blocking=False)
    val_y0 = datatrain_IP[0,:].to(device_name, non_blocking=False)
    # ODE solver solution for comparison
    val_true_y = datatrain_OP[0,:,:].to(device_name, non_blocking=False)
    val_true_dy = torch.from_numpy(train_dydt[0,:,:]).to(device_name, non_blocking=False)

    # testing data for plotting
    # Note: Just use the first sequence as the data will not be strictly increasing or decreasing for mixed cases
    test_t = torch.from_numpy(train_time[0,:]).to(device_name, non_blocking=False)
    test_y0 = datatrain_IP[0,:].to(device_name, non_blocking=False)
    # ODE solver solution for comparison
    test_true_y = datatrain_OP[0,:,:].to(device_name, non_blocking=False)
    test_true_dy = torch.from_numpy(train_dydt[0,:,:]).to(device_name, non_blocking=False)

    ii = 0  # keep track of epoch-1
    
    train_dl, val_dl = prepare_data(datatrain_IP, datatrain_OP, n_val, batch_size)
    if plot_itr:
        import plotting as plots

    for ep in range(epoch_in, nepoch + 1):
        t0 = time.time()       
        for inputs, targets in train_dl:  # inputs are batch_y0; targets are batch_y
            inputs  = inputs.to(device_name, non_blocking=False)
            targets = targets.to(device_name, non_blocking=False)
            optimizer.zero_grad()
            yhat = odeint(model, inputs, train_t, method=NODE_method, rtol=NODE_rtol, atol=NODE_atol).to(device_name, non_blocking=False)
            # need to swap the dimensions of targets as the first dim should be the seq_length (this is how NODE outputs the predictions)
            # Do not swap axis of full dataset as this will make the workflow inconsistent with the standard ML trianing code
            targets = np.swapaxes(targets, 0, 1)
            loss = loss_func(yhat, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()    
        train_loss_history.append(loss.item())
        t1 = time.time()

        with torch.no_grad():
            val_loss, pred_y, pred_dy = evaluate_NODE_model(model, NODE_method, NODE_rtol, NODE_atol, val_y0, val_t, val_true_y, loss_func)
            lrate_history.append(scheduler.get_last_lr())
            val_loss_history.append(val_loss.item())
            t2 = time.time()

            if ep%10==0:
                print(f'Iteration: {ep} | Train Loss {loss.item():1.3e} | Validation Loss {val_loss.item():1.3e} | Time-train = {t1-t0:4.3f}s |  Time-val = {t2-t1:4.3f}s')
            epoch_history.append(ii)
            ii += 1
        # plot training curves and full testing
        if plot_itr==True and ep%plot_epoch==0:
            with torch.no_grad():
                _, pred_y, pred_dy = evaluate_NODE_model(model, NODE_method, NODE_rtol, NODE_atol, test_y0, test_t, test_true_y, loss_func)
                args = test_t.cpu(), test_t.cpu(), test_true_y.cpu(), pred_y.cpu(), data_ip_varnames, \
                        test_true_dy.cpu(), pred_dy.cpu(), data_dy_varnames, \
                        epoch_history, train_loss_history, val_loss_history, lrate_history
                plots.plot_train('RANS',args)
        else:
            pass
        # early stopping if loss target is met
        if loss.item() < 1e-10: # 0.005 for undamped, # 0.0002 for damped
                break
    
    return train_loss_history, val_loss_history, optimizer, epoch_history, lrate_history, loss, loss_func

# ## Function: Evaluate NODE model
def evaluate_NODE_model(func, NODE_method, NODE_rtol, NODE_atol, test_y0, test_t, test_y, loss_func):
    pred_y = odeint(func, test_y0, test_t, method=NODE_method, rtol=NODE_rtol, atol=NODE_atol) # model prediction
    # predict dydt
    pred_dy = func(test_t, pred_y).cpu()
    val_loss = loss_func(pred_y, test_y)
    return val_loss, pred_y, pred_dy

# ## Save model to ONNX format
#Function to Convert to ONNX 
def convert_ONNX(model, dummy_input, modelsavename, input_names, output_names, verbose=True): 
    # Export the model   
    torch.onnx.export(   model,         # model being run 
                         dummy_input,       # model input (or a tuple for multiple inputs) 
                         modelsavename,       # where to save the model 
                         verbose=verbose,        # output verbose
                         export_params=True,  # store the trained parameter weights inside the model file 
                         opset_version=10,    # the ONNX version to export the model to 
                         do_constant_folding=True,  # whether to execute constant folding for optimization 
                         input_names = input_names,   # the model's input names 
                         output_names = output_names) # the model's output names 
                         # dynamic_axes={input_names  : {0 : 'batch_size'},    # variable length axes 
                         #               output_names : {0 : 'batch_size'}}) 
    print('Model has been converted to ONNX') 


# ## Funciton: ODE solver

# ### Convert E <-> q
# function to convert E -> q for Eh, Ev & Ep
def convert_E_q(E, Nfreq):
    q = E.copy()
    # if normEnergy:
    #     q[0] = q[0] / totalEt0     # convert Eh -> Enorm(uH2)
    #     q[1] = q[1] / totalEt0     # convert Ev -> Enorm(ww)
    #     q[2] = q[2] / totalEt0     # convert Ep -> Enorm(b2)
    #     q[3] = q[3] / (2 * Nfreq * totalEt0)   # convert bw -> Enorm(bw)
    # else:
    q[0] = 2 * q[0]            # convert Eh -> uH2
    q[1] = 2 * q[1]            # convert Ev -> ww
    q[2] = 2 * Nfreq**2 * q[2]     # convert Ep -> b2
    return q

# function to convert E -> q for Eh, Ev & Ep
def convert_E_normq(E):
    q = E.copy()
    # if normEnergy:
    q[0] = q[0] / totalEt0     # convert Eh -> Enorm(uH2)
    q[1] = q[1] / totalEt0     # convert Ev -> Enorm(ww)
    q[2] = q[2] / totalEt0     # convert Ep -> Enorm(b2)
    q[3] = q[3] / (2 * Nfreq * totalEt0)   # convert bw -> Enorm(bw)
    return q

# function to convert q -> E for q_H, q_v & q_b2
def convert_q_E(q, Nfreq):
    E = q.copy()
    # if normEnergy:
    #     E[0] = E[0] * totalEt0      # convert Enorm(uH2) -> Eh
    #     E[1] = E[1] * totalEt0      # convert Enorm(ww) -> Ev
    #     E[2] = E[2] * totalEt0      # convert Enorm(b2) -> Ep
    #     E[3] = E[3] * 2 * Nfreq * totalEt0 # convert Enorm(bw) -> bw
    # else:
    E[0] = 0.5 * E[0]             # convert uH2 -> Eh
    E[1] = 0.5 * E[1]             # convert ww -> Ev
    E[2] = E[2] / (2 * Nfreq**2)  # convert b2 -> Ep
    return E

# function to convert q -> E for q_H, q_v & q_b2
def convert_normq_E(q):
    E = q.copy()
    # if normEnergy:
    E[0] = E[0] * totalEt0      # convert Enorm(uH2) -> Eh
    E[1] = E[1] * totalEt0      # convert Enorm(ww) -> Ev
    E[2] = E[2] * totalEt0      # convert Enorm(b2) -> Ep
    E[3] = E[3] * 2 * Nfreq * totalEt0 # convert Enorm(bw) -> bw
    # else:
    #     E[0] = 0.5 * E[0]             # convert uH2 -> Eh
    #     E[1] = 0.5 * E[1]             # convert ww -> Ev
    #     E[2] = E[2] / (2 * Nfreq**2)  # convert b2 -> Ep
    return E


# ### MLP ODE
# true rhs of dqdt using data (from gradient of data)
def rhsRANStrue(q, t, interpmodel):
    # interpolate at t
    dqdt = interpmodel(t)
    return dqdt   # output in vector format

# rhs of dqdt using NN model
def rhsRANSnn(q, t, nnmodel):
    # q is actuall E. so need to convert Eh, Ev & Ep to q_H, q_v & q_b2
    # NOTE: Shouldn't need conversion if input variables to the model are in energy dimensions.
    # q = convert_E_q(q)
    # nninput = torch.unsqueeze(torch.from_numpy( np.array(q[:].reshape(-1).astype('float32')) ), 0)
    temp = np.array(q.reshape(-1).astype('float32'))
    temp = np.tile(temp, (seq_len,1))
    nninput = torch.unsqueeze(torch.from_numpy( temp ), 0)
    dqdt = nnmodel(nninput).detach().numpy()  # ML output in numpy array format
    return dqdt.reshape(-1)   # output in vector format

# Time integrate q using time data and initial value q0 using NN model as RHS of dqdt
def ODE_RK4(rhs, q0, t, rhsmodel):
    n = len(t)
    # initialize output matrix
    q = np.zeros( (n, len(q0)) )
    q[0,:] = q0[:]
    for i in range(n-1):
        y = q[i,:]
        h = t[i+1] - t[i]
        # Apply Runge Kutta Formula to find next value of q
        k1 = rhs(y             ,  t[i]        , rhsmodel)
        k2 = rhs(y + (0.5*h*k1),  t[i] + 0.5*h, rhsmodel)
        k3 = rhs(y + (0.5*h*k2),  t[i] + 0.5*h, rhsmodel)
        k4 = rhs(y + (h*k3)    ,  t[i] + h    , rhsmodel)
        # Update next value of y
        q[i+1,:] = y + ( (h/6.0) * (k1 + 2*k2 + 2*k3 + k4) )
        
    return q


# ### LSTM ODE
# true rhs of dqdt using data (from gradient of data)
def rhsRANStrue_LSTM(q, t, interpmodel, qlag, **modelargs):
    # interpolate at t
    dqdt = interpmodel(t)
    return dqdt   # output in vector format

# rhs of dqdt using NN model
def rhsRANSnn_LSTM(q, t, nnmodel, qlag, **modelargs):
    # q is actually E. So, need to convert Eh, Ev & Ep to q_H, q_v & q_b2
    # NOTE: Shouldn't need conversion if input variables to the LSTM are in energy dimensions.
    if not modelargs['normEnergy']:
        for i in range(qlag.shape[0]):
            qlag[i,:] = convert_E_q(qlag[i,:], modelargs['Nfreq'])
    nninput = torch.unsqueeze(torch.from_numpy( np.array(qlag.astype('float32')) ), 0)
    
    dqdt = nnmodel(nninput).detach().numpy()  # ML output in numpy array format
    return dqdt.reshape(-1)   # output in vector format

# Time integrate q using sequenced time data (for LSTM) at initial condition, q0, using NN model as RHS of dqdt
def ODE_RK4_LSTM(rhs, q0, t, **modelargs):
    rhsmodel = modelargs['rhsmodel']
    n = len(t)
    # initialize output matrix
    q = np.zeros( (n, q0.shape[-1]) )
    q[0,:] = q0[-1,:]     # the last element in the i/p sequence q0 is the q at first time instant
    for i in range(n-1):
        y = q[i,:]
        h = t[i+1] - t[i]
        # Apply Runge Kutta Formula to find next value of q
        qlag = q0.copy()
        
        k1 = rhs(y             ,  t[i]        , rhsmodel, qlag, **modelargs)
        qlag = np.concatenate( (q0[1:,:], [y + (0.5*h*k1)]), axis=0 )
        
        k2 = rhs(y + (0.5*h*k1),  t[i] + 0.5*h, rhsmodel, qlag, **modelargs)
        qlag = np.concatenate( (q0[1:,:], [y + (0.5*h*k2)]), axis=0 )
        
        k3 = rhs(y + (0.5*h*k2),  t[i] + 0.5*h, rhsmodel, qlag, **modelargs)
        qlag = np.concatenate( (q0[1:,:], [y + (h*k3)]), axis=0 )
        
        k4 = rhs(y + (h*k3)    ,  t[i] + h    , rhsmodel, qlag, **modelargs)
        
        # Update next value of y
        q[i+1,:] = y + ( (h/6.0) * (k1 + 2*k2 + 2*k3 + k4) )
        # Update i/p sequence q0 with new q at end
        q0 = np.concatenate( (q0[1:,:], [q[i+1,:]]), axis=0 )
        
    return q

'''
Same as ODE_RK4_LSTM() but q0 has additional inputs.
Works for data without additional inputs as well
'''
# Get additional inputs based on case type
def get_addIP(addIPs):
    if addIPs['casenum'] == 0: # no additional input
        return []
    if addIPs['casenum'] == 1: # only time
        return addIPs['t']
    if addIPs['casenum'] == 2: # time, ke, pe
        return [addIPs['t'], addIPs['ke'], addIPs['pe']]
    else:
        raise Exception(f"Case number {addIPs['casenum']} is invalid.")

# Time integrate q using sequenced time data (for LSTM) at initial condition, q0, using NN model as RHS of dqdt
def ODE_RK4_LSTM_addIP(rhs, q0, t, **modelargs):
    rhsmodel = modelargs['rhsmodel']
    num_op   = modelargs['num_op']
    casenum  = modelargs['addIP_case']
    n        = len(t)
    # initialize output matrix
    q        = np.zeros( (n, num_op) )
    q[0,:]   = q0[-1,:num_op]     # the last element in the i/p sequence q0 is the q at first time instant
    for i in range(n-1):
        y         = q[i,:]
        h         = t[i+1] - t[i]
        # Apply Runge Kutta Formula to find next value of q
        qlag      = q0.copy()
        
        k1        = rhs(y             ,  t[i]        , rhsmodel, qlag, **modelargs)
        yhat      = y + (0.5*h*k1)
        addIPs    = {'casenum': casenum, 't': t[i] + 0.5*h, 'ke': 0.5*sum(yhat[0:2]), 'pe': yhat[2]}
        new_input = np.append( yhat, get_addIP(addIPs) )
        qlag      = np.concatenate( (q0[1:,:], [new_input]), axis=0 )
        
        k2        = rhs(yhat          ,  t[i] + 0.5*h, rhsmodel, qlag, **modelargs)
        yhat      = y + (0.5*h*k2)
        addIPs    = {'casenum': casenum, 't': t[i] + 0.5*h, 'ke': 0.5*sum(yhat[0:2]), 'pe': yhat[2]}
        new_input = np.append( yhat, get_addIP(addIPs) )
        qlag      = np.concatenate( (q0[1:,:], [new_input]), axis=0 )
        
        k3        = rhs(yhat          ,  t[i] + 0.5*h, rhsmodel, qlag, **modelargs)
        yhat      = y + (h*k3)
        addIPs    = {'casenum': casenum, 't': t[i] + h, 'ke': 0.5*sum(yhat[0:2]), 'pe': yhat[2]}
        new_input = np.append( yhat, get_addIP(addIPs) )
        qlag      = np.concatenate( (q0[1:,:], [new_input]), axis=0 )
        
        k4        = rhs(yhat          ,  t[i] + h    , rhsmodel, qlag, **modelargs)
        
        # Update next value of y
        q[i+1,:]  = y + ( (h/6.0) * (k1 + 2*k2 + 2*k3 + k4) )
        
        # Update i/p sequence q0 with new q at end
        yhat      = q[i+1,:]
        addIPs    = {'casenum': casenum, 't': t[i+1], 'ke': 0.5*sum(yhat[0:2]), 'pe': yhat[2]}
        new_input = np.append( yhat, get_addIP(addIPs) )
        q0        = np.concatenate( (q0[1:,:], [new_input]), axis=0 )
        
    return q