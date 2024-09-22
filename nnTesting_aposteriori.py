# # Introduction
"""
* Testing 1D ML-URANS model with ODE solver and comparing with true data
* **NOTE**: For modeling energies
Sample run script:
(base)[user]$ python nnTesting_aposteriori.py --casenum 1 --casenum_test 1 --target_T 1.0 --seq_len_T 16 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --nepoch 5 --batch_size 10 --eval_regime training --tin_offset 0.0 --tin_add 29.0 --dt 0.0 --saveData --HDdir . --disable_cuda
"""

# import all python, pytorch, and ML libraries, classes and functions
from NN_funcs import *
import argparse

# # Start main

parser = argparse.ArgumentParser()
# Data type
parser.add_argument("--casenum", nargs="+", type=int, default=[1], required=True, help="Data case. See `data_funcs.py`: case 1 - F4R32; case 12 - F4R64; case 13 - F2R32")
parser.add_argument("--casenum_test", nargs="+", type=int, default=[1], required=True, help="case to be tested")
# parameters used for data acquisition
parser.add_argument("--target_T", type=float, default=1.0, required=False, help="target time period for sampling frequency")
parser.add_argument("--seq_len_T", type=int, default=64, required=True, help="target sequence length (sampling frequency) of LSTM input for target_T")
parser.add_argument("--normEnergy", default=False, action='store_true', help="normalize inputs to dimension of energy and non-dimensionalize both i/p & o/p using total energy at initial state")
parser.add_argument("--interpIO", default=False, action='store_true', help="interpolate data so that seq dt are same")
parser.add_argument("--set_dt_seq", default=False, action='store_true', help="set dt for `interpIO` such that time length of `seq_len` data points = 1 time period")
parser.add_argument("--dt_T", type=float, default=1.0, required=False, help="number of training data (%)")
parser.add_argument("--add_IP_case", type=int, default=0, required=False, help="cases determining the additional inputs case 0: no additional inputs; case 1: time; case 2: time, ke, pe")
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
# restart NN model
parser.add_argument("--restartTrain", default=False, action='store_true', help="restart training")
parser.add_argument("--ensmbleTrain", default=False, action='store_true', help="restart training with ensemble (first restart training for a few epoch and figure out ensm_errval)")
parser.add_argument("--ensmb_errval", default=1.7e-6, required=False, help="models with error below this will be used for ensemble. Choose manually based on original training curve")
parser.add_argument("--nepoch_in", default=4000, required=False, help="nepoch of original (old) model")
parser.add_argument("--lrate_restart", default=5e-4, required=False, help="new lrate")

# Device settings
parser.add_argument("--disable_cuda", default=False, action='store_true', help="disable GPU")

# Testing parameters
parser.add_argument("--eval_regime", type=str, default="training", required=True, help="model evaluation regime: in NN training or testing data regimes")
parser.add_argument("--tin_offset", type=float, default=0.0, required=True, help="time step for ODE solver. Use dt = 0 for time data from training samples")
parser.add_argument("--tin_add", type=float, default=5.0, required=True, help="amount of time after tin: tfi = tin + tin_add")
parser.add_argument("--dt", type=float, default=0.0, required=True, help="time step for ODE solver. Use dt = 0 for time data from training samples")

# Save data
parser.add_argument("--HDdir", type=str, default=".", required=True, help="root dir of all data and models")
parser.add_argument("--saveData", default=False, action='store_true', help="save data of plots for full plotting in `RANS_nnTesting_CompareResults.ipynb`")

# set parser arguments
args = parser.parse_args()

# ## Set parameters
# Data type
casenum         = args.casenum           # dataset case number(s) (see DataAcqu_LSTM.ipynb or data_funcs.py)
casenum_test    = args.casenum_test      # case to be tested
# parameters used for data acquisition
target_T        = args.target_T        # target time period for sampling frequency
seq_len_T       = args.seq_len_T       # target sequence length (sampling frequency) of LSTM input for target_T
dt_target       = target_T/seq_len_T

normEnergy      = args.normEnergy      # normalize inputs to dimension of energy and non-dimensionalize both i/p & o/p using total energy at initial state
interpIO        = args.interpIO        # interpolate data so that seq dt are same
set_dt_seq      = args.set_dt_seq      # set dt for `interpIO` such that time length of `seq_len` data points = 1 time period
dt_T            = args.dt_T            # dt for interpIO
if set_dt_seq:
    seq_len     = np.max([int(dt_T/dt_target), 1])  # sequence length (lag time) for LSTM input
else:
    seq_len     = seq_len_T            # sequence length (lag time) for LSTM input

add_IP_case     = args.add_IP_case     # cases determining the additional inputs case 0: no additional inputs; case 1: time; case 2: time, ke, pe
add_IP_time     = args.add_IP_time     # add time info as an additional input
add_IP_ke_pe_T  = args.add_IP_ke_pe_T  # add ke and pe decay time scales as additional inputs (make sure the IPs are normalized)
add_IP_Fr_Gn    = args.add_IP_Fr_Gn    # add Frh and Gn as additional inputs

Ntrainper       = args.Ntrainper       # number of training data (%)

# Hyperparameters (from training)
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
# restart NN model
restartTrain    = args.restartTrain    # restart training
ensmbleTrain    = args.ensmbleTrain    # restart training with ensemble (first restart training for a few epoch and figure out ensm_errval)
ensmb_errval    = args.ensmb_errval    # models with error below this will be used for ensemble. Choose manually based on original training curve
nepoch_in       = args.nepoch_in       # nepoch of original (old) model
lrate_restart   = args.lrate_restart   # new lrate
# Device settings
disable_cuda    = args.disable_cuda    # disable GPU?
# Model evaluation regime (in NN training or testing data regimes)
eval_regime     = args.eval_regime     # training testing
tin_offset      = args.tin_offset      # time offset from initial time of the evaluation regime
# settings for time integration
tin_add         = args.tin_add         # amount of time after tin: tfi = tin + tin_add   (29.4,  20)
dt              = args.dt              # time step for ODE solver. Use dt = 0 for time data from training samples. (0.064, 0.016)
# Save figures & output data
saveData        = args.saveData        # save data of plots for full plotting



# get casenames
casename     = [get_casename(i) for i in casenum]

# model filename
modelfname = f'SSTRANS_Phydt_normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
                f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
for i in range(len(casenum)):              # casename(s)
    modelfname = f'{modelfname}_{casename[i]}'
modelfname_full = f'{modelfname}_PyTModel_{model_name}_seqlen{seq_len}_nLSTMLayers{n_lstm_layers}_hiddensize{hidden_size}_'\
                    f'nMLPLayers{n_layers}_nNeurons{n_per_layer}_batch{batch_size}_lr{lrate}_'\
                    f'scaleData{scaleData}_nin{n_inputs}_Ntrain{Ntrainper}'
if ensmbleTrain or restartTrain:           # restart filename
    modelfname_full = f'{modelfname_full}_nepoch{nepoch_in}'
    lrate        = lrate_restart
    saveFnameRes =f'_restartEpoch{nepoch_in}_lr{lrate}'
else:
    modelfname_full = f'{modelfname_full}_nepoch{nepoch}'
savefilename = f'Data_models/{modelfname_full}'

# testing data filename
# get casenames
casename     = [get_casename(i) for i in casenum_test]
shuffledata  = False             # randomly shuffle training data or not
loaddatapath = f'Data_training/RANSdata_shuffle{shuffledata}_in-Energy_LSTM_seqlen{seq_len}_'\
                f'normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
                f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
for i in range(len(casenum_test)):    # casename(s)
    loaddatapath = f'{loaddatapath}_{casename[i]}'
loaddatapath = f'{loaddatapath}_Ntrain{Ntrainper}.npz'

# figure & output data filenames
fnameOP      = f'Out_Phydt{modelfname_full[17:]}_ODE_testdata'
for i in range(len(casenum_test)):    # casename(s)
    fnameOP = f'{fnameOP}_{casename[i]}'
fnameOP      = f'{fnameOP}_tout{tin_add}_toff{tin_offset}'
fnameData    = f'Data_models/{fnameOP}.npz'

# ============cuda settings============
if not disable_cuda and torch.cuda.is_available():
    device_name = torch.device('cuda')
else:
    device_name = torch.device('cpu')
print(f'Using the device: {device_name}')


# ## Load data

# ### Load unshuffled data from training (& testing) - scale data if necessary
npzfile = np.load(loaddatapath)
# !!!!!!!Make sure the datatype is float!!!!!!
nntrain_IP     = npzfile['datatrain_IP'].astype('float32')
nntest_IP      = npzfile['datatest_IP'].astype('float32')
nntrain_OP     = npzfile['datatrain_OP'].astype('float32')
nntest_OP      = npzfile['datatest_OP'].astype('float32')
nu             = npzfile['nu']
drhobardz      = npzfile['drhobardz']
accel          = npzfile['accel']
rho0           = npzfile['rho0']
Nfreq          = np.sqrt( -(accel/rho0) * drhobardz )

if scaleData:
    npzfile = np.load(loaddatapath)
    # !!!!!!!Make sure the datatype is float!!!!!!
    train_IP_temp = npzfile['datatrain_IP'].T.astype('float32')
    train_OP_temp = npzfile['datatrain_OP'].T.astype('float32')
    test_IP_temp = npzfile['datatest_IP'].T.astype('float32')
    test_OP_temp = npzfile['datatest_OP'].T.astype('float32')
    nntrain_IP, _, nntest_IP, _, input_scalar, output_scalar = get_scaleddataset(nntrain_IP, nntrain_OP, 
                                                                                                  nntest_IP, nntest_OP,
                                                                                                  MinMaxScaler(), MinMaxScaler())

data_ip_varnames  = npzfile['data_ip_varnames']
data_op_varnames  = npzfile['data_op_varnames']
nntrain_time      = npzfile['datatrain_time']
nntest_time       = npzfile['datatest_time']
if normEnergy:
    totalEt0 = npzfile['totalE'][0]


# ### Select which regime to evaluate model (training or testing regime)
# NOTE: Use scaled data for ODE solver, else, each variable might need separate dt for stability
if eval_regime == 'testing':
    test_data = nntest_IP
    test_OP   = nntest_OP
    test_time = nntest_time
elif eval_regime == 'training':
    test_data = np.append(nntrain_IP, nntest_IP, axis=0)
    test_OP   = np.append(nntrain_OP, nntest_OP, axis=0)
    test_time = np.append(nntrain_time, nntest_time, axis=0)

# if evaluation is done after an offset from starting time of each regime
if tin_offset>0:
    tstart = test_time[0] + tin_offset
    tempID = np.where( test_time>tstart)[0]
    if len(tempID)>2:
        print(f'Adding {tin_offset} offset to initial time for evaluation.\nOld inital time:\t{test_time[0]}\nNew initial time:\t{tstart}')
        test_time = test_time[tempID]
        test_data = test_data[tempID,:,:]
        test_OP   = test_OP[tempID,:]
    else:
        print(f'Offset ({tstart}) is above last time entry of data available [{test_time[0]}, {test_time[-1]}] OR not enough data left after offset ({len(tempID)}). Reduce offset time!')


# ## Converting from q to E (if needed)
if normEnergy:
    E_true = test_data.copy()
else:
    E_true = test_data.copy()
    for i in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            E_true[i,j,:] = convert_q_E(test_data[i,j,:], Nfreq)
    print(f'Energy converted from q')
dEdt_true = np.gradient(E_true[:,-1,:], test_time, edge_order=2, axis=0)


# ## Define & Load the network
# original model
model = defNNmodel(model_name, n_inputs, n_outputs, seq_len, hidden_size, n_lstm_layers, n_layers, n_per_layer, drp_in, drp_hd, lkyReLU_alpha, device_name)

# load model
modelpath = f'{savefilename}.pkl'
model.load_state_dict(torch.load(modelpath));
model.eval();


# ## Time integrate using NN model - pass through ODE solver
# Initial condition
q0   = E_true[0,:]
# time step for ODE
if dt==0:
    dt   = np.min(np.diff(test_time))
else:
    print(f"NOTE: Data dt = {np.min(np.diff(test_time))}. Chosen dt = {dt}")
# time data
tin  = test_time[0]
tfi  = tin + tin_add
t    = np.arange(tin, tfi, dt)

# q from true RHS - to validate ODE solver
finterp = interpolate.interp1d(test_time, test_OP, axis=0, kind='cubic', fill_value="extrapolate")
test_OP_interp = finterp(t)
tic = time.time()
modelargs = {'rhsmodel'   : finterp}
q_true = ODE_RK4_LSTM(rhsRANStrue_LSTM, q0, t, **modelargs)
# modelargs = {'rhsmodel'   : finterp,
#              'num_op'     : n_outputs,
#              'addIP_case' : add_IP_case}
# q_true = ODE_RK4_LSTM_addIP(rhsRANStrue_LSTM, q0, t, **modelargs)
elpsdt = time.time() - tic
print(f'Time elapsed for true RHS: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')

# q from NN model - time integrate using RHS = NN model
tic = time.time()
modelargs = {'rhsmodel'   : model,
             'num_op'     : n_outputs,
             'addIP_case' : add_IP_case,
             'normEnergy' : normEnergy,
             'Nfreq'      : Nfreq
            }
q_ML = ODE_RK4_LSTM_addIP(rhsRANSnn_LSTM, q0, t, **modelargs)
elpsdt = time.time() - tic
print(f'Time elapsed for NN RHS: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')

# Save data
if saveData:
    np.savez(fnameData,
             test_time=test_time, q_true=E_true[:,-1,:],
             t=t, q_ODE=q_true, q_ML=q_ML, totalEt0 = totalEt0, Nfreq=Nfreq,
             data_op_varnames=data_op_varnames)
    print('=============Save data=============')
else:
    print('=============Data not saved=============')


