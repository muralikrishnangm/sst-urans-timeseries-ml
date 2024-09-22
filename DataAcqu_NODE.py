# # Introduction
""" 
**Python script for transforming training data for LSTM into that for NODE - URANS modeling of homogeneous SST**

* **NOTE:** For NODE model. y0 is input and sequential data is output.

* See [DataAcqu_LSTM.ipynb](DataAcqu_LSTM.ipynb) for details of generating data

* This script loads the LSTM training data and converts into the format that NODE model uses

Sample run script:
(base)[user]$ python DataAcqu_NODE.py --Datadir Data_raw --casenum 2 3 --target_T 1.0 --seq_len_T 36 --Ntrainper 0.7 --normEnergy --interpIO --set_dt_seq --dt_T 0.9 --savedata --HDdir .
"""

from data_funcs import *
import argparse


# # Start main
parser = argparse.ArgumentParser()

# Data type
parser.add_argument("--Datadir", type=str, default='Data_raw', required=True, help="Machine directory for the raw data.")
parser.add_argument("--casenum", nargs="+", type=int, default=[1], required=True, help="Data case. See `data_funcs.py`: case 1 - F4R32; case 12 - F4R64; case 13 - F2R32; case 101 - Pendulum")
parser.add_argument("--target_T", type=float, default=1.0, required=False, help="target time period for sampling frequency")
parser.add_argument("--seq_len_T", type=int, default=64, required=True, help="target sequence length (sampling frequency) of LSTM input for target_T")
# Pre-processing parameters
parser.add_argument("--Ntrainper", type=float, default=0.9, required=False, help="number of training data (%)")
parser.add_argument("--normEnergy", default=False, action='store_true', help="normalize inputs to dimension of energy and non-dimensionalize both i/p & o/p using total energy at initial state")
parser.add_argument("--interpIO", default=False, action='store_true', help="interpolate data so that seq dt are same")
parser.add_argument("--set_dt_seq", default=False, action='store_true', help="set dt for `interpIO` such that time length of `seq_len` data points = `dt_T` time period")
parser.add_argument("--dt_T", type=float, default=1.0, required=False, help="dt for interpIO")
# Additional inputs
parser.add_argument("--add_IP_time", default=False, action='store_true', help="add time info as an additional input")
parser.add_argument("--add_IP_ke_pe_T", default=False, action='store_true', help="add ke and pe decay time scales as additional inputs (make sure the IPs are normalized)")
parser.add_argument("--add_IP_Fr_Gn", default=False, action='store_true', help="add Frh and Gn as additional inputs")
# Miscellaneous
parser.add_argument("--normIO", default=False, action='store_true', help="normalize input or output [Only for debugging data/model. Use proper nomralization for actual model]")
parser.add_argument("--shuffledata", default=False, action='store_true', help="randomly shuffle training data or not")
# Save data
parser.add_argument("--HDdir", type=str, default=".", required=True, help="root dir of all data and models")
parser.add_argument("--savedata", default=False, action='store_true', help="Save data at `Data_training/<filename>` with `<filename>` based on parameters.")

# set parser arguments
args = parser.parse_args()

HDdir           = args.HDdir           # root dir of all data and models
machine_dir     = args.Datadir         
cases_train     = args.casenum               
# case 1 - F4R32, case 12 - F4R64, case 13 - F2R32; NOTE: see data_funcs.py
Ntrainper       = args.Ntrainper       # number of training data (%)

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

normIO          = args.normIO          # normalize input or output [Only for debugging data/model. Use proper nomralization for actual model]
shuffledata     = args.shuffledata     # randomly shuffle training data or not
  
savedata        = args.savedata


# # Get case info and set savefilename
numcases = len(cases_train)
savefilename = f'{HDdir}/Data_training/RANSdata_shuffle{shuffledata}_in-Energy_NODE_seqlen{seq_len}_'\
            f'normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
            f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
case_info = []
for i in range(numcases):
    case_info += [get_case_info(cases_train[i], machine_dir),]
    savefilename += '_'+case_info[i].casename
savefilename += f'_Ntrain{Ntrainper}.npz'


# # Load LSTM data
def get_gradient(data, time):
    '''
    # data is of dimnesion [batch, sequence, variables]
    # time is of dimension [batch, sequence]
    # output is of dimension [batch, sequence, variables]
    '''
    data_dt = np.array([np.gradient(data[0,:,:], time[0,:], axis=0, edge_order=2)])
    
    # Loop over each depth dimension
    for i in range(1, data.shape[0]):
        # Compute the gradient along the time axis for the current depth
        data_gradient = np.gradient(data[i,:,:], time[i,:], axis=0, edge_order=2)
        # Append the gradient data for the current depth to the list
        data_dt = np.append(data_dt, np.array([data_gradient]), axis=0)
    return data_dt

loaddatapath = f'Data_training/RANSdata_shuffle{shuffledata}_in-Energy_LSTM_seqlen{seq_len}_'\
            f'normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
            f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}'
Ntrain_list = np.zeros((numcases)).astype(int)

# Initialize data with first case
# =====================
j = 0
loaddatafname = f'{loaddatapath}_{case_info[j].casename}_Ntrain{Ntrainper}.npz'
print(f"Loading data {loaddatafname}")
npzfile = np.load(loaddatafname)

data_ip_varnames = npzfile['data_ip_varnames']
data_dy_varnames = npzfile['data_op_varnames']

# Extract data for NODE model
ninputs = npzfile['datatrain_IP'].shape[2]
ndydt = npzfile['datatrain_OP'].shape[1]
Ntrain = npzfile['datatrain_IP'].shape[0]
Ntest = npzfile['datatest_IP'].shape[0]
data_time = npzfile['data_time']   # full time data

# batched time data
batch_t = np.zeros([len(data_time)-seq_len+1, seq_len])
for i in range(len(data_time)-seq_len+1):
    batch_t[i,:] = data_time[i:i+seq_len]

# extract training data: state variables from the LSTM data
datatrain_IP = npzfile['datatrain_IP'][:,0,:]    # inputs: initial conditions of states (first value of inputs of LSTM data)
datatrain_OP = npzfile['datatrain_IP'][:,:,:]   # outputs: time series of states/inputs (full input seq of LSTM data)
datatrain_time = batch_t[0:Ntrain,:]
datatrain_dydt = get_gradient(npzfile['datatrain_IP'], datatrain_time) # np.gradient(npzfile['datatrain_IP'][:,:,:], datatrain_time[:,0], edge_order=2, axis=0)   # dydt

# extract testing data: state variables from the LSTM data
datatest_IP = npzfile['datatest_IP'][:,0,:]    # inputs: initial conditions of states (first value of inputs of LSTM data)
datatest_OP = npzfile['datatest_IP'][:,:,:]   # outputs: time series of states/inputs (full input seq of LSTM data)
datatest_time = batch_t[Ntrain:Ntrain+Ntest,:]
datatest_dydt = get_gradient(npzfile['datatest_IP'], datatest_time) # np.gradient(datatest_IP[:,:], datatest_time[:,0], edge_order=2, axis=0)   # dydt

Ntrain_list[j]    = Ntrain
print(f"Length of {case_info[j].casename} dataset: {Ntrain_list[j]}")

# Loop over all cases
# =====================
for j in range(1,numcases):
    loaddatafname = f'{loaddatapath}_{case_info[j].casename}_Ntrain{Ntrainper}.npz'
    print(f"Loading data {loaddatafname}")
    npzfile = np.load(loaddatafname)
    
    data_ip_varnames = npzfile['data_ip_varnames']
    data_dy_varnames = npzfile['data_op_varnames']
    
    # Extract data for NODE model
    ninputs = npzfile['datatrain_IP'].shape[2]
    ndydt = npzfile['datatrain_OP'].shape[1]
    Ntrain = npzfile['datatrain_IP'].shape[0]
    Ntest = npzfile['datatest_IP'].shape[0]
    data_time_case = npzfile['data_time']   # full time data
    data_time = np.append(data_time, data_time_case, axis=0)   # dydt
    
    # batched time data
    batch_t = np.zeros([len(data_time_case)-seq_len+1, seq_len])
    for i in range(len(data_time_case)-seq_len+1):
        batch_t[i,:] = data_time_case[i:i+seq_len]
    
    # extract training data: state variables from the LSTM data
    datatrain_IP_case = npzfile['datatrain_IP'][:,0,:]    # inputs: initial conditions of states (first value of inputs of LSTM data)
    datatrain_OP_case = npzfile['datatrain_IP'][:,:,:]   # outputs: time series of states/inputs (full input seq of LSTM data)
    datatrain_time_case = batch_t[0:Ntrain,:]
    datatrain_dydt_case = get_gradient(npzfile['datatrain_IP'], datatrain_time_case) # np.gradient(datatrain_IP_case[:,:], datatrain_time_case[:,0], edge_order=2, axis=0)   # dydt
    datatrain_IP = np.append(datatrain_IP, datatrain_IP_case, axis=0)    # inputs: initial conditions of states (first value of inputs of LSTM data)
    datatrain_OP = np.append(datatrain_OP, datatrain_OP_case, axis=0)   # outputs: time series of states/inputs (full input seq of LSTM data)
    datatrain_time = np.append(datatrain_time, datatrain_time_case, axis=0)
    datatrain_dydt = np.append(datatrain_dydt, datatrain_dydt_case, axis=0)   # dydt
    
    # extract testing data: state variables from the LSTM data
    datatest_IP_case = npzfile['datatest_IP'][:,0,:]    # inputs: initial conditions of states (first value of inputs of LSTM data)
    datatest_OP_case = npzfile['datatest_IP'][:,:,:]   # outputs: time series of states/inputs (full input seq of LSTM data)
    datatest_time_case = batch_t[Ntrain:Ntrain+Ntest,:]
    datatest_dydt_case = get_gradient(npzfile['datatest_IP'], datatest_time_case) # np.gradient(datatest_IP_case[:,:], datatest_time_case[:,0], edge_order=2, axis=0)   # dydt
    datatest_IP = np.append(datatest_IP, datatest_IP_case, axis=0)    # inputs: initial conditions of states (first value of inputs of LSTM data)
    datatest_OP = np.append(datatest_OP, datatest_OP_case, axis=0)   # outputs: time series of states/inputs (full input seq of LSTM data)
    datatest_time = np.append(datatest_time, datatest_time_case, axis=0)
    datatest_dydt = np.append(datatest_dydt, datatest_dydt_case, axis=0)   # dydt
    
    Ntrain_list[j]    = Ntrain
    print(f"Length of {case_info[j].casename} dataset: {Ntrain_list[j]}")

Ntrain = len(datatrain_time)
print(f"Length of full dataset: {Ntrain}")


# # Save data to npz file
if savedata == True:
    if numcases==1:
        np.savez(savefilename, 
             datatrain_IP=datatrain_IP, datatrain_OP=datatrain_OP, datatrain_dydt=datatrain_dydt, datatrain_time=datatrain_time,
             datatest_IP=datatest_IP, datatest_OP=datatest_OP, datatest_dydt=datatest_dydt, datatest_time=datatest_time,
             data_time=data_time, data_ip_varnames=data_ip_varnames, data_dy_varnames=data_dy_varnames,
             nu=case_info[0].nu, drhobardz=case_info[0].drhobardz, accel=case_info[0].accel, rho0=case_info[0].rho0, 
             totalE=npzfile['totalE'], Frh=npzfile['Frh'], Gn=npzfile['Gn'])
    else:
        np.savez(savefilename, 
                 datatrain_IP=datatrain_IP, datatrain_OP=datatrain_OP, datatrain_dydt=datatrain_dydt, datatrain_time=datatrain_time,
                 datatest_IP=datatest_IP, datatest_OP=datatest_OP, datatest_dydt=datatest_dydt, datatest_time=datatest_time,
                 data_time=data_time, data_ip_varnames=data_ip_varnames, data_dy_varnames=data_dy_varnames)
    print("===========Data saved===========")
else: 
    print("===========Data not saved===========")


