# # Introduction
""" 
* The following notebook is a simple implementation of a neural ordinary differential equation. This model is referred to a time-stepping model since the neural network is used in conjuction with a standard ODE solver. 

**Ensemble ML model for directly modeling the rhs of the URANS equations for homogeneous SST**
* Input:  y0, t
* Output: y(t)
* **Flow field**: Various SST cases
* **NN framework**: PyTorch
* **Data script**: [DataAcqu_NODE.ipynb](DataAcqu_NODE.ipynb)
Sample run script:
(base)[user]$ python NODE_nnTraining.py --casenum 1 --target_T 1.0 --seq_len_T 64 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --nepoch 5 --batch_size 100 --saveNN --HDdir .
"""

# # Import Packages
# import all python, pytorch, and ML libraries, classes, and functions
from NN_funcs import *
import argparse

# # Start main

parser = argparse.ArgumentParser()
# Data type
parser.add_argument("--casenum", nargs="+", type=int, default=[1], required=True, help="Data case. See `data_funcs.py`: case 1 - F4R32; case 12 - F4R64; case 13 - F2R32")
# parameters used for data acquisition
parser.add_argument("--target_T", type=float, default=1.0, required=False, help="target time period for sampling frequency")
parser.add_argument("--seq_len_T", type=int, default=64, required=True, help="target sequence length (sampling frequency) of LSTM input for target_T")
parser.add_argument("--normEnergy", default=False, action='store_true', help="normalize inputs to dimension of energy and non-dimensionalize both i/p & o/p using total energy at initial state")
parser.add_argument("--interpIO", default=False, action='store_true', help="interpolate data so that seq dt are same")
parser.add_argument("--set_dt_seq", default=False, action='store_true', help="set dt for `interpIO` such that time length of `seq_len` data points = 1 time period")
parser.add_argument("--dt_T", type=float, default=1.0, required=False, help="number of training data (%)")
parser.add_argument("--add_IP_time", default=False, action='store_true', help="add time info as an additional input")
parser.add_argument("--add_IP_ke_pe_T", default=False, action='store_true', help="add ke and pe decay time scales as additional inputs (make sure the IPs are normalized)")
parser.add_argument("--add_IP_Fr_Gn", default=False, action='store_true', help="add Frh and Gn as additional inputs")
parser.add_argument("--Ntrainper", type=float, default=0.9, required=False, help="number of training data (%)")
# Hyperparameters
# data
parser.add_argument("--n_val", type=float, default=0.1, required=False, help="validation data (ratio)")
parser.add_argument("--n_inputs", type=int, default=4, required=False, help="number of inputs")
parser.add_argument("--n_outputs", type=int, default=4, required=False, help="number of outputs")
# network architecture
parser.add_argument("--model_name", default="NODE_MLP", required=False, help="model name: 'SingleMLP' 'ResNet' 'DenseNet' 'LSTM' 'NODE_MLP'")
parser.add_argument("--NODE_method", default="rk4", required=False, help="ODE solver in NODE model")
parser.add_argument("--NODE_rtol", type=float, default=1e-3, required=False, help="relative tolerance in NODE")
parser.add_argument("--NODE_atol", type=float, default=1e-4, required=False, help="absolute tolerance in NODE")
# dense layers
parser.add_argument("--n_per_layer", type=int, default=40, required=False, help="number of neurons on all hidden dense layers")
parser.add_argument("--n_layers", type=int, default=10, required=False, help="number of dense layers - ResNet, DenseNet")
parser.add_argument("--drp_in", type=float, default=0, required=False, help="probability value for input dropout")
parser.add_argument("--drp_hd", type=float, default=0, required=False, help="probability value for hidden layer dropout")
# other settings
parser.add_argument("--lkyReLU_alpha", type=float, default=0.1, required=False, help="slope of leakyReLU")
parser.add_argument("--lrate", type=float, default=5e-2, required=False, help="learning rate")
parser.add_argument("--gamma", type=float, default=0.99, required=False, help="for learning rate scheduler")
# training parameters
parser.add_argument("--nepoch", type=int, default=100, required=True, help="number of training epochs (additional for restart runs)")
parser.add_argument("--batch_size", type=int, default=100, required=True, help="batch size")
parser.add_argument("--scaleData", default=False, action='store_true', help="scale data (e.g.: naive min-max scaling of I/O)")
parser.add_argument("--plot_itr", default=False, action='store_true', help="plot training curves and testing")
parser.add_argument("--plot_epoch", type=int, default=200, required=False, help="epoch interval for plotting")
# Save NN model
parser.add_argument("--HDdir", type=str, default=".", required=True, help="root dir of all data and models")
parser.add_argument("--saveNN", default=False, action='store_true', help="save the model, learning history")
parser.add_argument("--restartTrain", default=False, action='store_true', help="restart training")
parser.add_argument("--ensmbleTrain", default=False, action='store_true', help="restart training with ensemble (first restart training for a few epoch and figure out ensm_errval)")
parser.add_argument("--ensmb_errval", default=1.7e-6, required=False, help="models with error below this will be used for ensemble. Choose manually based on original training curve")
parser.add_argument("--nepoch_in", default=4000, required=False, help="nepoch of original (old) model")
parser.add_argument("--lrate_restart", default=5e-4, required=False, help="new lrate")

# Device settings
parser.add_argument("--disable_device", default=False, action='store_true', help="disable device (NVIDIA or AMD GPU; Mac Metal)")

# set parser arguments
args = parser.parse_args()

# ## Set parameters
# Data type
casenum         = args.casenum         # dataset case number(s) (see DataAcqu_LSTM.ipynb or data_funcs.py)
# parameters used for data acquisition
target_T        = args.target_T        # target time period for sampling frequency
seq_len_T       = args.seq_len_T       # target sequence length (sampling frequency) of LSTM input for target_T
dt_target       = target_T/seq_len_T

normEnergy      = args.normEnergy      # normalize inputs to dimension of energy and non-dimensionalize both i/p & o/p using total energy at initial state
interpIO        = args.interpIO        # interpolate data so that seq dt are same
set_dt_seq      = args.set_dt_seq      # set dt for `interpIO` such that time length of `seq_len` data points = dt_T time period
dt_T            = args.dt_T            # dt for interpIO
if set_dt_seq:
    seq_len     = np.max([int(dt_T/dt_target), 1])  # sequence length (lag time) for LSTM input
else:
    seq_len     = seq_len_T            # sequence length (lag time) for LSTM input

if seq_len==1:
    raise Exception(f'Need more than 1 elements in the sequence for integration. '\
                    f'Current: seq_len_T={seq_len_T}, dt_T={dt_T}, seq_len={seq_len}. Exiting...')

add_IP_time     = args.add_IP_time     # add time info as an additional input
add_IP_ke_pe_T  = args.add_IP_ke_pe_T  # add ke and pe decay time scales as additional inputs (make sure the IPs are normalized)
add_IP_Fr_Gn    = args.add_IP_Fr_Gn    # add Frh and Gn as additional inputs

Ntrainper       = args.Ntrainper       # number of training data (%)

# Hyperparameters
# data
n_val           = args.n_val           # validation data (ratio)
n_inputs        = args.n_inputs        # number of inputs
n_outputs       = args.n_outputs       # number of outputs
# network architecture
model_name      = args.model_name      # model name: 'SingleMLP' 'ResNet' 'DenseNet' 'LSTM' 'NODE_MLP'
NODE_method     = args.NODE_method     # ODE solver in NODE model
NODE_rtol       = args.NODE_rtol       # relative tolerance in NODE
NODE_atol       = args.NODE_atol       # absolute tolerance in NODE
# dense layers
n_per_layer     = args.n_per_layer     # number of neurons on all hidden dense layers (manually update model if needed)
n_layers        = args.n_layers        # number of dense layers - ResNet, DenseNet
drp_in          = args.drp_in          # probability value for input dropout 
drp_hd          = args.drp_hd          # probability value for hidden layer dropout
# other settings
lkyReLU_alpha   = args.lkyReLU_alpha   # slope of leakyReLU
lrate           = args.lrate           # learning rate
gamma           = args.gamma           # for learning rate scheduler

# training parameters
nepoch          = args.nepoch          # number of training epochs (additional for restart runs)
batch_size      = args.batch_size      # batch size
scaleData       = args.scaleData       # scale data (e.g.: naive min-max scaling of I/O)
plot_itr        = args.plot_itr        # plot training curves and testing
plot_epoch      = args.plot_epoch      # epoch interval for plotting

# Save NN model
HDdir           = args.HDdir           # root dir of all data and models
saveNN          = args.saveNN          # save the model, learning history
restartTrain    = args.restartTrain    # restart training
ensmbleTrain    = args.ensmbleTrain    # restart training with ensemble (first restart training for a few epoch and figure out ensm_errval)
ensmb_errval    = args.ensmb_errval    # models with error below this will be used for ensemble. Choose manually based on original training curve
nepoch_in       = args.nepoch_in       # nepoch of original (old) model
lrate_restart   = args.lrate_restart   # new lrate

# Device settings
disable_device    = args.disable_device    # disable GPU?

# get casenames
casename     = [get_casename(i) for i in casenum]

# model filename
savefilename = f'{HDdir}/Data_models/SSTRANS_EnergyEqn_normEnergy{int(normEnergy)}_interpIO{int(interpIO)}'\
                f'_setdt{int(set_dt_seq)}-T{dt_T}_'\
                f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
for i in range(len(casenum)):              # casename(s)
    savefilename = f'{savefilename}_{casename[i]}'
savefilename = f'{savefilename}_PyTModel_{model_name}_seqlen{seq_len}_'\
                f'nMLPLayers{n_layers}_nNeurons{n_per_layer}_batch{batch_size}_lr{lrate}_'\
                f'scaleData{scaleData}_nin{n_inputs}_Ntrain{Ntrainper}'
if ensmbleTrain or restartTrain:           # restart filename
    savefilename = f'{savefilename}_nepoch{nepoch_in}'
    lrate        = lrate_restart
    saveFnameRes =f'_restartEpoch{nepoch_in}_lr{lrate}'
else:
    savefilename = f'{savefilename}_nepoch{nepoch}'

# data filename
shuffledata  = False             # randomly shuffle training data or not
loaddatapath = f'{HDdir}/Data_training/RANSdata_shuffle{shuffledata}_in-Energy_NODE_seqlen{seq_len}_'\
            f'normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
            f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
for i in range(len(casenum)):    # casename(s)
    loaddatapath = f'{loaddatapath}_{casename[i]}'
loaddatapath = f'{loaddatapath}_Ntrain{Ntrainper}.npz'

# ============cuda settings============
if not disable_device and (torch.cuda.is_available() or torch.backends.mps.is_available()):
    if torch.cuda.is_available():
        device_name = torch.device('cuda')
    else:
        device_name = torch.device("mps")
else:
    device_name = torch.device('cpu')
print(f'Using the device: {device_name}')


# # Import Data

npzfile = np.load(loaddatapath)

# !!!!!!!Make sure the datatype is float!!!!!!
train_IP   = npzfile['datatrain_IP'].astype('float32')
train_OP   = npzfile['datatrain_OP'].astype('float32')
train_dydt = npzfile['datatrain_dydt'].astype('float32')
train_time = npzfile['datatrain_time'].astype('float32')
t          = npzfile['data_time'].astype('float32')

# Test data is 30% of full data (to change use `DataAcqu*` code)
test_IP    = npzfile['datatest_IP'].astype('float32')
test_OP    = npzfile['datatest_OP'].astype('float32')
test_time  = npzfile['datatest_time']
# varnames and time data for plotting
data_ip_varnames  = npzfile['data_ip_varnames']
data_dy_varnames  = npzfile['data_dy_varnames']

print(f'Number of samples for training:\t{train_IP.shape[0]}')
print(f'Number of samples for testing:\t{test_IP.shape[0]}')

if n_inputs != train_IP.shape[-1]:
    raise Exception(f'n_inputs ({n_inputs:,}) ~= # input of training samples from data ({train_IP.shape[-1]:,})')

# Send data to device
train_IP_d = torch.from_numpy(train_IP).to(device_name, non_blocking=False)
train_OP_d = torch.from_numpy(train_OP).to(device_name, non_blocking=False)


# # Define the neural network model within NODE

model = defNNmodel(model_name, n_inputs=n_inputs, n_outputs=n_outputs, n_layers=n_layers, 
                  n_per_layer=n_per_layer, device_name=device_name)

# # Train the NODE model

tin = time.time()
train_loss_history, val_loss_history, optimizer, epoch_history, lrate_history, loss, loss_func = train_NODEmodel(train_IP_d, train_OP_d, train_dydt, train_time, 
                    n_val, batch_size, model, NODE_method, NODE_rtol, NODE_atol, nepoch, lrate, gamma, device_name, 
                    plot_itr, plot_epoch, data_ip_varnames, data_dy_varnames)
elpsdt = time.time() - tin
print(f'Time elapsed for training model: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')


# ## Save model

# send model and everything to cpu for saving
if device_name.type == 'cuda' or device_name.type == 'mps':
    device_name = torch.device('cpu')
    model.to(device_name)   # for saving the model and plotting results
    model.device_name = device_name

if saveNN:
    modelpath = savefilename+'_restart.tar'
    torch.save({
                'epoch_history': epoch_history,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'NODE_method': NODE_method, 
                'NODE_rtol': NODE_rtol, 
                'NODE_atol': NODE_atol,
                'loss_func': loss_func,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'lrate_history': lrate_history,
                }, modelpath)

if saveNN:
    print("========Finished saving========")
else:
    print("========Finished training========")
    
    