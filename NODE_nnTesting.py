# # Introduction
"""
* Testing 1D NODE-URANS model and comparing with true data
* **NOTE**: For modeling energies
Sample run script:
(base)[user]$ python NODE_nnTesting.py --casenum 1 --casenum_test 1 --target_T 1.0 --seq_len_T 64 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --nepoch 5 --batch_size 100 --eval_regime training --tin_offset 0.0 --tin_add 30.0 --dt 0.0 --saveData --HDdir . --disable_cuda
"""

# import all python, pytorch, and ML libraries, classes and functions
from NN_funcs import *
from scipy.integrate import odeint

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

if seq_len==1:
    raise Exception(f'Need more than 1 elements in the sequence for integration. '\
                    f'Current: seq_len_T={seq_len_T}, dt_T={dt_T}, seq_len={seq_len}. Exiting...')

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
modelfname = f'SSTRANS_EnergyEqn_normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
                f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
for i in range(len(casenum)):              # casename(s)
    modelfname = f'{modelfname}_{casename[i]}'
modelfname_full = f'{modelfname}_PyTModel_{model_name}_seqlen{seq_len}_'\
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
loaddatapath = f'Data_training/RANSdata_shuffle{shuffledata}_in-Energy_NODE_seqlen{seq_len}_'\
            f'normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
            f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
for i in range(len(casenum_test)):    # casename(s)
    loaddatapath = f'{loaddatapath}_{casename[i]}'
loaddatapath = f'{loaddatapath}_Ntrain{Ntrainper}.npz'

# figure & output data filenames
fnameOP      = f'Out_{modelfname_full[18:]}_ODE_testdata'
for i in range(len(casenum_test)):    # casename(s)
    fnameOP = f'{fnameOP}_{casename[i]}'
fnameOP      = f'{fnameOP}_tout{tin_add}_toff{tin_offset}'
fnameFig     = f'Figs/{fnameOP}.png'
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
nntrain_IP   = npzfile['datatrain_IP'].astype('float32')
nntrain_OP   = npzfile['datatrain_OP'].astype('float32')
nntrain_dydt   = npzfile['datatrain_dydt'].astype('float32')
nntrain_time = npzfile['datatrain_time'].astype('float32')
t            = npzfile['data_time'].astype('float32')
nu           = npzfile['nu']
drhobardz    = npzfile['drhobardz']
accel        = npzfile['accel']
rho0         = npzfile['rho0']
Nfreq        = np.sqrt( -(accel/rho0) * drhobardz )

# Test data is 30% of full data (to change use `DataAcqu*` code)
nntest_IP    = npzfile['datatest_IP'].astype('float32')
nntest_OP    = npzfile['datatest_OP'].astype('float32')
nntest_dydt   = npzfile['datatest_dydt'].astype('float32')
nntest_time  = npzfile['datatest_time']

# varnames and time data for plotting
data_ip_varnames  = npzfile['data_ip_varnames']
data_dy_varnames  = npzfile['data_dy_varnames']

print(f'Number of samples for training:\t{nntrain_IP.shape[0]}')
print(f'Number of samples for testing:\t{nntest_IP.shape[0]}')

if n_inputs != nntrain_IP.shape[-1]:
    raise Exception(f'n_inputs ({n_inputs:,}) ~= # input of training samples from data ({nntrain_IP.shape[-1]:,})')

if normEnergy:
    totalEt0 = npzfile['totalE'][0]


# ### Select which regime to evaluate model (training or testing regime)

# NOTE: Use scaled data for ODE solver, else, each variable might need separate dt for stability
if eval_regime == 'testing':
    test_data = nntest_IP 
    test_OP   = nntest_dydt[:,0,:]
    test_time = nntest_time[:,0]
elif eval_regime == 'training':
    test_data = np.append(nntrain_IP, nntest_IP, axis=0) # nntrain_IP
    test_OP   = np.append(nntrain_dydt[:,0,:], nntest_dydt[:,0,:], axis=0) # nntrain_dydt[:,0,:]
    test_time = np.append(nntrain_time[:,0], nntest_time[:,0], axis=0) # nntrain_time[:,0]

# if evaluation is done after an offset from starting time of each regime
if tin_offset>0:
    tstart = test_time[0] + tin_offset
    tempID = np.where( test_time>tstart)[0]
    if len(tempID)>2:
        print(f'Adding {tin_offset} offset to initial time for evaluation.\nOld inital time:\t{test_time[0]}\nNew initial time:\t{tstart}')
        test_time = test_time[tempID]
        test_data = test_data[tempID,:]
        test_OP   = test_OP[tempID,:]
    else:
        print(f'Offset ({tstart}) is above last time entry of data available [{test_time[0]}, {test_time[-1]}] OR not enough data left after offset ({len(tempID)}). Reduce offset time!')


# ## Converting from q to E (if needed)

if normEnergy:
    E_true = test_data.copy()
else:
    E_true = test_data.copy()
    for i in range(test_data.shape[0]):
        E_true[i,:] = convert_q_E(test_data[i,:], Nfreq)
    print(f'Energy converted from q')
dEdt_true = np.gradient(E_true[:,:], test_time, edge_order=2, axis=0)


# ## Define & Load the network

# original model
model = defNNmodel(model_name, n_inputs=n_inputs, n_outputs=n_outputs, n_layers=n_layers, 
                  n_per_layer=n_per_layer, device_name=device_name)
# load model
modelpath = savefilename+'_restart.tar'
checkpoint = torch.load(modelpath)
epoch_history = checkpoint['epoch_history']
train_loss_history = checkpoint['train_loss_history']
val_loss_history = checkpoint['val_loss_history']
lrate_history = checkpoint['lrate_history']
NODE_method = checkpoint['NODE_method']
NODE_rtol = checkpoint['NODE_rtol']
NODE_atol = checkpoint['NODE_atol']
loss_func = checkpoint['loss_func']
model.load_state_dict(checkpoint['model_state_dict'])
model.eval();

# ## Time integrate using NODE

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
dq_true = finterp(t)
# true rhs of dqdt using data (from gradient of data)
def rhsRANStrue(q, t, interpmodel):
    # interpolate at t
    dqdt = interpmodel(t)
    return dqdt   # output in vector format
tic = time.time()
q_true = odeint(rhsRANStrue, q0, t, args=(finterp,))
elpsdt = time.time() - tic
print(f'Time elapsed for true RHS: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')

# q from NN model - time integrate using RHS = NN model
tic = time.time()

with torch.no_grad():
    _, q_ML, dq_ML = evaluate_NODE_model(model, NODE_method, NODE_rtol, NODE_atol, 
                                         torch.from_numpy(q0), torch.from_numpy(t), torch.from_numpy(q_true), loss_func)
elpsdt = time.time() - tic
print(f'Time elapsed for NN RHS: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')
q_ML = q_ML.detach().cpu().numpy()
dq_ML = dq_ML.detach().cpu().numpy()


# Save data

if saveData:
    np.savez(fnameData,
             test_time=test_time, q_true=E_true[:,:],
             t=t, q_ODE=q_true, q_ML=q_ML, dq_ODE=dq_true, dq_ML=dq_ML, totalEt0=totalEt0, Nfreq=Nfreq,
             data_ip_varnames=data_ip_varnames, data_dy_varnames=data_dy_varnames)
    print('=============Save data=============')
else:
    print('=============Data not saved=============')
