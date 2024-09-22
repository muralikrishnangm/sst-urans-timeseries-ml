# # Introduction
""" 
**Python script for data acquisition for modeling pendulum**

* **NOTE:** For LSTM model. Sequential data is input.

* **Input data**: KE, PE, gw
    - Size of a single input to NN model: $L_{\text{seq}} \times 3$, where $L_{\text{seq}}$ is the length of the time sequence
* **Output data**: Time derivative of energy terms
    - Size of corresponding output from NN model: $3$
* **Training data size**:
    - Input:  $L_{\text{seq}} \times 3 \times N_{\text{train}}$
    - Output: $3 \times N_{\text{train}}$
## Data information
See variable `cases` in `data_funcs.py` for metadata info.

Sample run script:
(base)[user]$ python DataAcqu_LSTM_pendulum.py --Datadir Data_raw --case 101 --target_T 1.0 --seq_len_T 128 --Ntrainper 0.7 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --savedata --HDdir .
"""

# import all libraries, classes and functions
from data_funcs import *
import subprocess

# import matplotlib.pyplot as plt

import argparse

# # Start main
parser = argparse.ArgumentParser()

# Data type
parser.add_argument("--Datadir", type=str, default='Data_raw', required=True, help="Machine directory for the raw data.")
parser.add_argument("--case", type=int, default=1, required=True, help="Data case. See `data_funcs.py`: case 1 - F4R32; case 12 - F4R64; case 13 - F2R32; case 101 - Pendulum")
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

# # User input
HDdir           = args.HDdir           # root dir of all data and models
machine_dir     = args.Datadir         
case            = args.case               
# case 101 - Pendulum; NOTE: see data_funcs.py
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
add_IP_time     = args.add_IP_time     # add time info as an additional input
add_IP_ke_pe_T  = args.add_IP_ke_pe_T  # add ke and pe decay time scales as additional inputs (make sure the IPs are normalized)
add_IP_Fr_Gn    = args.add_IP_Fr_Gn    # add Frh and Gn as additional inputs

normIO          = args.normIO          # normalize input or output [Only for debugging data/model. Use proper nomralization for actual model]
shuffledata     = args.shuffledata     # randomly shuffle training data or not
  
savedata        = args.savedata


# # Load data
# ## Get case info and compute constants
case_info    = get_case_info(case, machine_dir)
savefilename = f'{HDdir}/Data_training/Pendulum_shuffle{shuffledata}_in-Energy_LSTM_seqlen{seq_len}_'\
                f'normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
                f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}_'\
                f'{case_info.casename}_Ntrain{Ntrainper}.npz'

# ## Generate pendulum data & plot
# Construct the command string with variables
command = f"python -u datasets.py {case_info.casename} --tperiod {dt_T} --win {seq_len}"
# Execute the command
result = subprocess.run(command, shell=True, stderr=subprocess.PIPE)
# Check if there was an error
if result.returncode != 0:
    print(f"Error occurred: {result.stderr.decode()}")
    # break  # Exit the loop if there was an error

file = f'pendulum_b0.5_m1_g25_l1_theta0.7854_omega0_t0-30_npts1000_solRK45_tp{dt_T}_win{seq_len}.npz'
data = np.load(f"{machine_dir}/Pend/{file}")

data_ip_lstm = data['labels']
data_op_lstm = data['outputs']
data_time_lstm = data['seq_time'][:,-1]
data_time = data['times_full']
Nsampls = data_ip_lstm.shape[0]
seq_len = data_ip_lstm.shape[1]
nin = data_ip_lstm.shape[2]
nop = data_op_lstm.shape[1]
data_ip_varnames = ['Ek','Ep',r'$g\omega$'] 
data_op_varnames = [r'$\frac{d}{dt}(Ek)$',r'$\frac{d}{dt}(Ep)$',r'$\frac{d}{dt}(g\omega)$']

print(f'# samples \t= {Nsampls}\nSequence len \t= {seq_len}\n# i/p \t= {nin}\n# o/p \t= {nop}')

# # Validating data with original data
# fig, axs = plt.subplots(1,2,figsize=(15,5))
# axs = axs.ravel()
# axs[0].plot(data_time_lstm, data_ip_lstm[:,-1,0])
# # axs[0].plot(data_time_lstm, data_ip[seq_len-1:,0],'--')
# # axs[0].set_xlim([-1,5])
# axs[1].plot(data_time_lstm, data_op_lstm[:,0])
# # axs[1].plot(data_time_lstm, data_op[seq_len-1:,0],'--');
# # axs[1].set_xlim([-1,5]);

# ## Split data train & testing
# compute number of samples for training & testing
Ntrain = int(Nsampls * Ntrainper)
Ntest = Nsampls - Ntrain
if shuffledata: samplList = np.random.permutation(np.arange(0,Nsampls - Ntest))
else: samplList = np.arange(0,Nsampls - Ntest)

if Ntrain>len(samplList):
    print("Need more samples!!!!!")
else:
    samplList = samplList[0:Ntrain]
    # Training data
    datatrain_IP = data_ip_lstm[samplList, :]
    datatrain_OP = data_op_lstm[samplList, :]
    # Testing data
    datatest_IP = data_ip_lstm[Nsampls-Ntest:, :]
    datatest_OP = data_op_lstm[Nsampls-Ntest:, :]
    # time stamps
    datatrain_time = data_time_lstm[samplList]
    datatest_time  = data_time_lstm[Nsampls-Ntest:]


# # ## Validate data
# if ~shuffledata:
#     fig, axs = plt.subplots(2,2,figsize=(20,10))
#     axs = axs.ravel()
#     for i in range(nin):
#         axs[i].plot(datatrain_time, datatrain_IP[:,-1,i], 'b-')
#         axs[i].plot(datatest_time, datatest_IP[:,-1,i], 'r-')
#         axs[i].set_title(data_ip_varnames[i])
#         # axs[i].set_xlim([1,50])
#     axs[-1].plot(datatrain_time, np.sum(datatrain_IP[:,-1,0:2], axis=1), 'b-')
#     axs[-1].set_title(f'Sum of {data_ip_varnames[0:2]}')

# if ~shuffledata:
#     fig, axs = plt.subplots(2,2,figsize=(20,10))
#     axs = axs.ravel()
#     for i in range(nop):
#         axs[i].plot(datatrain_time, datatrain_OP[:,i], 'b-')
#         axs[i].plot(datatest_time, datatest_OP[:,i], 'r-')
#         axs[i].set_title(data_op_varnames[i])
#         # axs[i].set_xlim([1,50])


# ## Save to npz file
if savedata:
    np.savez(savefilename,
             Ntrain=Ntrain, Ntest=Ntest, 
             datatrain_IP=datatrain_IP, datatrain_OP=datatrain_OP, 
             datatest_IP=datatest_IP, datatest_OP=datatest_OP,
             data_ip_varnames=data_ip_varnames, data_op_varnames=data_op_varnames,
             datatrain_time=datatrain_time, datatest_time=datatest_time, data_time=data_time)

