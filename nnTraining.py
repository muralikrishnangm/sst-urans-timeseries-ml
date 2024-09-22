# # Introduction
"""
**Ensemble ML model for directly modeling the URANS non-linear closure terms
* Input:  sequence of flow variables
* Output: time derivate of variables
* **Flow field**: Various SST cases
* **NN framework**: PyTorch
* **Data script**: `DataAcqu_LSTM.ipynb`

Sample run script:
(base)[user]$ python nnTraining.py --casenum 1 --target_T 1.0 --seq_len_T 16 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --nepoch 5 --batch_size 10 --saveNN --HDdir .
(base)[user]$ python nnTraining.py --casenum 1 --target_T 1.0 --seq_len_T 16 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --nepoch 100 --batch_size 100 --ensmbleTrain --ensmbleEpochs 50 80 80 --HDdir .
"""

# # Import libraries
# import all python, pytorch, and ML libraries, classes and functions
from NN_funcs import *
import argparse

# Uncomment the following for deterministic training
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# np.random.seed(0)
# torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)

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
parser.add_argument("--model_name", default="LSTM", required=False, help="model name: 'SingleMLP' 'ResNet' 'DenseNet' 'LSTM'")
# dense layers
parser.add_argument("--n_per_layer", type=int, default=15, required=False, help="number of neurons on all hidden dense layers")
parser.add_argument("--n_layers", type=int, default=1, required=False, help="number of dense layers - ResNet, DenseNet")
parser.add_argument("--drp_in", type=float, default=0, required=False, help="probability value for input dropout")
parser.add_argument("--drp_hd", type=float, default=0, required=False, help="probability value for hidden layer dropout")
# LSTM layers
parser.add_argument("--n_lstm_layers", type=int, default=4, required=False, help="number of layers - LSTM blocks/cells")
parser.add_argument("--hidden_size", type=int, default=10, required=False, help="hidden state size of LSTM cell")
# other settings
parser.add_argument("--lkyReLU_alpha", type=float, default=0.1, required=False, help="slope of leakyReLU")
parser.add_argument("--lrate", type=float, default=1e-3, required=False, help="learning rate")
parser.add_argument("--reg_type", default='None', required=False, help="manual weight regularization 'L1' 'L2' 'None'")
parser.add_argument("--reg_factor", type=float, default=1e-6, required=False, help="regularization factor")
# training parameters
parser.add_argument("--nepoch", type=int, default=100, required=True, help="number of training epochs (additional for restart runs)")
parser.add_argument("--batch_size", type=int, default=10, required=True, help="batch size")
parser.add_argument("--scaleData", default=False, action='store_true', help="scale data (e.g.: naive min-max scaling of I/O)")

# Save NN model
parser.add_argument("--HDdir", type=str, default=".", required=True, help="root dir of all data and models")
parser.add_argument("--saveNN", default=False, action='store_true', help="save the model, learning history")
parser.add_argument("--ensmbleTrain", default=False, action='store_true', help="Training with ensemble (first run a couple of cases to see after which epoch the model conferges)")
parser.add_argument("--ensmbleEpochs", type=int, default=[2500, 3500, 3500], required=False, nargs='+', help="Epochs for calculating avg loss and starting ensemble: [epoch_avg_start, epoch_avg_top, epoch_ensemble_start]. Choose manually based on sample training curves.")
parser.add_argument("--restartTrain", default=False, action='store_true', help="restart training")
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
model_name      = args.model_name      # model name: 'SingleMLP' 'ResNet' 'DenseNet' 'LSTM'
# dense layers
n_per_layer     = args.n_per_layer     # number of neurons on all hidden dense layers (manually update model if needed)
n_layers        = args.n_layers        # number of dense layers - ResNet, DenseNet
drp_in          = args.drp_in          # probability value for input dropout 
drp_hd          = args.drp_hd          # probability value for hidden layer dropout
# RNN/LSTM layers
n_lstm_layers   = args.n_lstm_layers   # number of layers - LSTM blocks/cells
hidden_size     = args.hidden_size     # hidden state size of LSTM cell
# other settings
lkyReLU_alpha   = args.lkyReLU_alpha   # slope of leakyReLU
lrate           = args.lrate           # learning rate
reg_type        = args.reg_type        # manual weight regularization 'L1' 'L2' 'None'
reg_factor      = args.reg_factor      # regularization factor
# training parameters
nepoch          = args.nepoch          # number of training epochs (additional for restart runs)
batch_size      = args.batch_size      # batch size
scaleData       = args.scaleData       # scale data (e.g.: naive min-max scaling of I/O)

# Save NN model
HDdir           = args.HDdir           # root dir of all data and models
saveNN          = args.saveNN          # save the model, learning history
ensmbleTrain    = args.ensmbleTrain    # restart training with ensemble (first restart training for a few epoch and figure out ensm_errval)
ensmbleEpochs   = args.ensmbleEpochs   # [epoch_avg_start, epoch_avg_top, epoch_ensemble_start]
epoch_avgls     = ensmbleEpochs[0:2]
epoch_ens_start = ensmbleEpochs[2]
restartTrain    = args.restartTrain    # restart training
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
savefilename = f'{savefilename}_PyTModel_{model_name}_seqlen{seq_len}_nLSTMLayers{n_lstm_layers}_hiddensize{hidden_size}_'\
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
loaddatapath = f'{HDdir}/Data_training/RANSdata_shuffle{shuffledata}_in-Energy_LSTM_seqlen{seq_len}_'\
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


# ## Prepare the data: split training & testing

# ### Load full data and scale
npzfile = np.load(loaddatapath)
# !!!!!!!Make sure the datatype is float!!!!!!
train_IP = npzfile['datatrain_IP'].astype('float32')
train_OP = npzfile['datatrain_OP'].astype('float32')
# Test data is 30% of full data (to change use `DataAcqu*` code)
test_IP = npzfile['datatest_IP'].astype('float32')
test_OP = npzfile['datatest_OP'].astype('float32')
# varnames and time data for plotting
data_ip_varnames  = npzfile['data_ip_varnames']
data_op_varnames  = npzfile['data_op_varnames']
datatrain_time    = npzfile['datatrain_time']
datatest_time     = npzfile['datatest_time']

print(f'Number of samples for training:\t{train_IP.shape[0]}')
print(f'Number of samples for testing:\t{test_IP.shape[0]}')

if n_inputs != train_IP.shape[-1]:
    raise Exception(f'n_inputs ({n_inputs:,}) ~= # input of training samples from data ({train_IP.shape[-1]:,})')

# Scale data
if scaleData:
    train_IP, train_OP, test_IP, test_OP, input_scalar, output_scalar = get_scaleddataset(train_IP, train_OP, 
                                                                                          test_IP, test_OP, 
                                                                                          MinMaxScaler(), MinMaxScaler())

    # Save scaling measures (min-max)
    if saveNN:
        sclrArray = np.append(input_scalar.data_min_.reshape(n_inputs,1),input_scalar.data_max_.reshape(n_inputs,1),axis=1)
        sclrFname = loaddatapath[:-4]+'_input.txt'
        np.savetxt(sclrFname, sclrArray, fmt="%s")
        sclrArray = np.append(output_scalar.data_min_.reshape(n_outputs,1),output_scalar.data_max_.reshape(n_outputs,1),axis=1)
        sclrFname = loaddatapath[:-4]+'_output.txt'
        np.savetxt(sclrFname, sclrArray, fmt="%s")

# Send data to device
train_IP_d = torch.from_numpy(train_IP).to(device_name, non_blocking=False)
train_OP_d = torch.from_numpy(train_OP).to(device_name, non_blocking=False)
test_IP_d = torch.from_numpy(test_IP).to(device_name, non_blocking=False)
test_OP_d = torch.from_numpy(test_OP).to(device_name, non_blocking=False)


# ## Define the network
# original model
model = defNNmodel(model_name, n_inputs, n_outputs, seq_len, hidden_size, n_lstm_layers, n_layers, n_per_layer, drp_in, drp_hd, lkyReLU_alpha, device_name)


# ## Train the model
# ============cuda settings============
if (not ensmbleTrain) and (not restartTrain):
    model.train()
    print(f"Training from begining...")
    t = time.time()
    train_loss_history, val_loss_history, optimizer, epoch, loss = train_model(train_IP_d, train_OP_d, n_val, 
                                                                               batch_size, model, nepoch, lrate,
                                                                               reg_factor, reg_type, device_name)
    elpsdt = time.time() - t
elif ensmbleTrain and (not restartTrain):
    model.train()
    print(f"Training from begining (ensemble-auto: epochs_avg = {epoch_avgls}; epoch_ensb_start = {epoch_ens_start})...")
    t = time.time()
    # run automated ensemble training
    modelEnsmb = defNNmodel(model_name, n_inputs, n_outputs, seq_len, hidden_size, n_lstm_layers, n_layers, n_per_layer, drp_in, drp_hd, lkyReLU_alpha, device_name)
    train_loss_history, val_loss_history, optimizer, epoch, loss, ensmb_count = train_model_ensemble_auto(train_IP_d, train_OP_d, n_val,
                                                                                        batch_size, model, modelEnsmb, nepoch, lrate,
                                                                                        reg_factor, reg_type, device_name,
                                                                                        epoch_ens_start, epoch_avgls)
    model = copy.deepcopy(modelEnsmb)
    print(f'Number of ensembles = {ensmb_count:,}')
    elpsdt = time.time() - t
elif restartTrain:
    # model
    optimizer = qhoptim.QHAdam(model.parameters(), eps=1e-08, weight_decay=0, **qhoptim.QHAdam.from_nadam())
    checkpoint = torch.load(savefilename+'_restart.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    nepoch += epoch
    loss = checkpoint['loss']
    model.train()
    # loss history
    loss_history = np.load(savefilename+'_epoch'+str(nepoch_in)+'_losshistory.npy')
    train_loss_history = tuple(loss_history[:,0].reshape(1, -1)[0])
    val_loss_history = tuple(loss_history[:,1].reshape(1, -1)[0])
    # re-start training
    if not ensmbleTrain:    # restart the training WITHOUT ensemble of model
        print(f"Restart training from {epoch} epoch...")
        t = time.time()
        train_loss_history, val_loss_history, optimizer, epoch, loss = train_model(train_IP_d, train_OP_d, n_val, 
                                                                                   batch_size, model, nepoch, lrate,
                                                                                   reg_factor, reg_type, device_name,
                                                                                   epoch, train_loss_history, 
                                                                                   val_loss_history)    
    else:              # restart training WITH ensemble of model
        # re-start training
        print(f"Restart training from {epoch} epoch and computing model ensemble (ensemble-auto: epochs_avg = {epoch_avgls}; epoch_ensb_start = {epoch_ens_start})...")
        model.train()
        t = time.time()
        # run automated ensemble training
        modelEnsmb = defNNmodel(model_name, n_inputs, n_outputs, seq_len, hidden_size, n_lstm_layers, n_layers, n_per_layer, drp_in, drp_hd, lkyReLU_alpha, device_name)
        modelEnsmb.load_state_dict(checkpoint['model_state_dict'])
        train_loss_history, val_loss_history, optimizer, epoch, loss, ensmb_count = train_model_ensemble_auto(train_IP_d, train_OP_d, n_val,
                                                                                            batch_size, model, modelEnsmb, nepoch, lrate,
                                                                                            reg_factor, reg_type, device_name,
                                                                                            epoch_ens_start, epoch_avgls)
        model = copy.deepcopy(modelEnsmb)
        print(f'Number of ensembles = {ensmb_count:,}')
    
    elpsdt = time.time() - t
    savefilename = f'{savefilename}{saveFnameRes}'
print(f'Time elapsed for training model: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')
print(f'Final training MSE: {train_loss_history[-1]:4.3e} (RMSE: {math.sqrt(train_loss_history[-1]):4.3e})')
print(f'Final validation MSE: {val_loss_history[-1]:4.3e} (RMSE: {math.sqrt(val_loss_history[-1]):4.3e})')


# ## Evaluate the model & prediction

# * Make a separate dataloader for test dataset
# send model and everything to cpu for inference
if device_name.type == 'cuda' or device_name.type == 'mps':
    device_name = torch.device('cpu')
    model.to(device_name)   # for saving the model and plotting results
    model.device_name = device_name
    # # ensemble model
    # modelEnsmb.to(device_name)   # for saving the model and plotting results
    # modelEnsmb.device_name = device_name
    
model.eval()
n_testsamples = test_IP.shape[0]
temp_MLop = model( torch.from_numpy( np.array(test_IP) ) )
test_OP_ML = temp_MLop.detach().numpy()
mse = mean_squared_error(test_OP,test_OP_ML, multioutput='uniform_average')
msefull = mean_squared_error(test_OP,test_OP_ML, multioutput='raw_values')
print('Test MSE: %4.3e (RMSE: %4.3e)' % (mse, math.sqrt(mse)))

# # ensemble model
# modelEnsmb.eval()
# n_testsamples = test_IP.shape[0]
# temp_MLop = modelEnsmb( torch.from_numpy( np.array(test_IP) ) )
# test_OP_ML = temp_MLop.detach().numpy()
# mse = mean_squared_error(test_OP,test_OP_ML, multioutput='uniform_average')
# msefull = mean_squared_error(test_OP,test_OP_ML, multioutput='raw_values')
# print('Test MSE (ensemble): %4.3e (RMSE: %4.3e)' % (mse, math.sqrt(mse)))

# ## Save model
# send model and everything to cpu for saving
if device_name.type == 'cuda' or device_name.type == 'mps':
    device_name = torch.device('cpu')
    model.to(device_name)   # for saving the model and plotting results
    model.device_name = device_name

# * Save Python version of model
if saveNN:
    modelpath = savefilename+'.pkl'
    torch.save(model.state_dict(), modelpath)

if saveNN:
    print("========Finished saving========")
else:
    print("========Finished training========")