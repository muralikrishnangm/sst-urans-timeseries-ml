 # Introduction
"""
**Python script for data acquisition for RANS modeling of homogeneous SST**

* **NOTE:** For LSTM model. Sequential data is input.

* **Input data**:  Horizontal velocity ($u^2 + v^2$), vertical velocity ($w^2$), buoyancy ($b^2$), and buoyancy flux ($bw$)
    - Size of a single input to NN model: $L_{\text{seq}} \times 4$, where $L_{\text{seq}}$ is the length of the time sequence
* **Output data**: Time derivative of energy terms $\frac{\partial(E_i)}{\partial t}$ (horizontal kinetic energy, vertical kinetic energy, and potential energy) and buoyancy flux $\frac{\partial(bw)}{\partial t}$
    - Size of corresponding output from NN model: $4$
* **Training data size**:
    - Input:  $L_{\text{seq}} \times 4 \times N_{\text{train}}$
    - Output: $4 \times N_{\text{train}}$

## Data information
See variable `cases` in `data_funcs.py` for metadata info.

Sample run script:
(base)[user]$ python DataAcqu_LSTM.py --Datadir Data_raw --case 1 --target_T 1.0 --seq_len_T 16 --Ntrainper 0.7 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --savedata --HDdir .
"""

# import all libraries, classes and functions
from data_funcs import *
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

HDdir           = args.HDdir           # root dir of all data and models
machine_dir     = args.Datadir         
case            = args.case               
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
add_IP_time     = args.add_IP_time     # add time info as an additional input
add_IP_ke_pe_T  = args.add_IP_ke_pe_T  # add ke and pe decay time scales as additional inputs (make sure the IPs are normalized)
add_IP_Fr_Gn    = args.add_IP_Fr_Gn    # add Frh and Gn as additional inputs

normIO          = args.normIO          # normalize input or output [Only for debugging data/model. Use proper nomralization for actual model]
shuffledata     = args.shuffledata     # randomly shuffle training data or not
  
savedata        = args.savedata


# # Load data

# ## Get case info and compute constants
case_info    = get_case_info(case, machine_dir)
savefilename = f'{HDdir}/Data_training/RANSdata_shuffle{shuffledata}_in-Energy_LSTM_seqlen{seq_len}_'\
                f'normEnergy{int(normEnergy)}_interpIO{int(interpIO)}_setdt{int(set_dt_seq)}-T{dt_T}_'\
                f'IPtime{int(add_IP_time)}_IPKEPEtime{int(add_IP_ke_pe_T)}_IPFrGn{int(add_IP_Fr_Gn)}_'\
                f'{case_info.casename}_Ntrain{Ntrainper}.npz'

# ## Load data & compute time derivatives
filename = case_info.HDdir+'rstats'
rstats = statsData(filename)
if case_info.casename[-1]=='a':
    filename = case_info.HDdir+'vstats'
else:
    filename = case_info.HDdir+'vstats2'
vstats = statsData(filename)
# get time conversion files if it is isotropic turb cases (deBK JFM19)
if int(str(case)[0])>1:
    filename = f'{case_info.HDdir}itime_{case_info.casename}'
    tstats = statsData(filename)

# ## Compute variables
# Compute B (bw) and b2
if case_info.casename[-1]=='a':  # gravity in y-direction
    B = rstats.values[:,rstats.ID('rv')]  * (case_info.accel/case_info.rho0)
else:
    B = rstats.values[:,rstats.ID('rw')]  * (case_info.accel/case_info.rho0)
bb = rstats.values[:,rstats.ID('rr')]  * (case_info.accel/case_info.rho0)**2

# Compute energies
uu = rstats.values[:,rstats.ID('uu')]
if case_info.casename[-1]=='a':  # gravity in y-direction
    vv = rstats.values[:,rstats.ID('ww')]
    ww = rstats.values[:,rstats.ID('vv')]
else:
    vv = rstats.values[:,rstats.ID('vv')]
    ww = rstats.values[:,rstats.ID('ww')]
uH = uu + vv
ke = 0.5 * (uH + ww)
pe = bb / (2 * case_info.N**2)
totalE = ke + pe

# Compute Frh and Gn
# eps = mestats.values[:,mestats.ID('eps')]
# k = 0.5 * (uH + ww)
# Frt = eps / (N * uH)
# Frt = 2 * np.pi * U / (N * (U**3/eps))
# Gn  = eps / (nu * N**2)
Frh = vstats.values[:,vstats.ID('Fh')]
dkedt = np.gradient(ke, rstats.values[:,rstats.IDt], edge_order=2, axis=0)
epsilon = -(B) - dkedt
Frt = 2*np.pi*epsilon / (case_info.N * np.sqrt(uu+vv+ww))
if case_info.casename[-1]=='a':  # use Gn
    Gn = vstats.values[:,vstats.ID('Gn')]
else:
    Gn = vstats.values[:,vstats.ID('Rb')]


# # Save data

# ## Choose input & output data
# time data
data_time = rstats.values[:,rstats.IDt]
    
# =========Input=========
# pwork, eh, ev, ep (dissipation in hori., ver., pot. energy equations??), B (buoyancy flux)
# uH = vstats.values[:,vstats.ID('u')]**2 + vstats.values[:,vstats.ID('v')]**2
# ww = vstats.values[:,vstats.ID('w')]**2
if normEnergy:
    data_ip = np.array( [ uH/(2*totalE[0]), 
                         ww/(2*totalE[0]), 
                         bb/(2*case_info.N**2 * totalE[0]), 
                         B/(2*case_info.N*totalE[0]) ] ).T
    data_ip_varnames = [ 'uH2/2*Et0', 'ww/2*Et0', 'b2/2*N2*Et0', 'bw/2*N*Et0' ]
else:
    data_ip = np.array( [ uH, ww,
                         bb,
                         B ] ).T
    data_ip_varnames = [ 'uH2', 'ww' , 'b2', 'bw' ]
# add id's for variables for future reference
id_uH = 0; id_ww = 1; id_bb = 2; id_bw = 3

# =========Output=========
# time derivative of Energy terms
# NOTE: !!!Need to take time derivative using re-scaled time variable!!!
# select the o/p (normalize as necessary) and compute time derivative
if normEnergy:
    data_op   = np.gradient(data_ip[:,:], data_time, edge_order=2, axis=0)
    data_op_varnames = [ 'd(Eh/Et0)/dt', 'd(Ev/Et0)/dt', 'd(Ep/Et0)/dt', 'd(bw/2*N*Et0)/dt' ]
else:
    data_op = np.array( [ uH/2, ww/2,
                         pe,
                         B ] ).T
    data_op   = np.gradient(data_op[:,:], data_time, edge_order=2, axis=0)
    data_op_varnames = [ 'd(Eh)/dt', 'd(Ev)/dt', 'd(Ep)/dt', 'd(bw)/dt' ]

# =======================
# interpolate data to finer time instants from time conversion files
if int(str(case)[0])>1:
    # interp i/p
    finterp = interpolate.interp1d(data_time, data_ip, axis=0, kind='cubic', fill_value="extrapolate")
    data_time_dim = tstats.values[:,tstats.ID('dim_t')]
    data_ip = finterp(data_time_dim)
    # interp o/p
    finterp = interpolate.interp1d(data_time, data_op, axis=0, kind='cubic', fill_value="extrapolate")
    data_op = finterp(data_time_dim)
    # interp Frh & Gn
    finterp = interpolate.interp1d(data_time, Frh, axis=0, kind='linear', fill_value="extrapolate")
    Frh = finterp(data_time_dim)
    finterp = interpolate.interp1d(data_time, Gn, axis=0, kind='linear', fill_value="extrapolate")
    Gn = finterp(data_time_dim)
    data_time_t = tstats.values[:,tstats.ID('t')]
    data_time_T = tstats.values[:,tstats.ID('T')]

# =======================
# choose appropriate time data
if case_info.normTime:
    # # use dimensionless time, t
    # if int(str(case)[0])>1:
    #     data_time = data_time_t
    # else:
    #     data_time = (data_time - case_info.t0) / (case_info.tLe)
    # time_varname = ['t']
    # use buoyancy time period, T
    if int(str(case)[0])>1:
        data_time = data_time_T
    else:
        data_time = ( ((data_time - case_info.t0) / case_info.tLe) - 1 ) / case_info.Fr
    time_varname = ['T']
else: # use dimensional time, \hat{t}
    if int(str(case)[0])>1:
        data_time = data_time_dim
    time_varname = ['\hat{t}']

# =======================
# interpolate data to have constant dt
if interpIO:
    if set_dt_seq:    # set dt such that time length of seq_len data = dt_T time period
        dt = dt_T/seq_len
    else:
        dt = np.mean(np.diff(data_time))  # min or mean??
    data_time_temp = data_time.copy()
    data_time = np.arange(data_time_temp[0], data_time_temp[-1], dt)
    # interp i/p
    finterp = interpolate.interp1d(data_time_temp, data_ip, axis=0, kind='cubic', fill_value="extrapolate")
    data_ip = finterp(data_time)
    # interp o/p
    if normEnergy:
        data_op = np.gradient(data_ip[:,:], data_time, edge_order=2, axis=0)
    else:
        finterp = interpolate.interp1d(data_time_temp, data_op, axis=0, kind='cubic', fill_value="extrapolate")
        data_op = finterp(data_time)
    # filter gradient data due to discontinuities from numerical differentiation
    data_op_temp = np.copy(data_op)
    if case_info.filtgrad:
        data_op = gaussian_filter1d(data_op, sigma=case_info.filtgrad_sigma, axis=0)
        if np.max(np.abs(data_op_temp - data_op)) > 1e0:
            raise Exception(f"High error in filtering!!!! {np.max(np.abs(data_op_temp - data_op))}")
    # interp Frh & Gn
    finterp = interpolate.interp1d(data_time_temp, Frh, axis=0, kind='linear', fill_value="extrapolate")
    Frhinterp = finterp(data_time)
    finterp = interpolate.interp1d(data_time_temp, Gn, axis=0, kind='linear', fill_value="extrapolate")
    Gninterp = finterp(data_time)

# =======================
# skip transient data
if case_info.Tskip>0:
    ID = np.where(data_time > case_info.Tskip)[0]
    data_time = data_time[ID]
    data_ip   = data_ip  [ID,:]
    data_op   = data_op  [ID,:]
    Frhinterp = Frhinterp[ID]
    Gninterp  = Gninterp [ID]

# =======================
# skip end of data
if case_info.Gnfinal>0 and case_info.Gnfinal>Gninterp[-1]:
    ID = np.where(Gninterp > case_info.Gnfinal)[0]
    data_time = data_time[ID]
    data_ip   = data_ip  [ID,:]
    data_op   = data_op  [ID,:]
    Frhinterp = Frhinterp[ID]
    Gninterp  = Gninterp [ID]

# =======================
# add additional inputs
if add_IP_ke_pe_T:
    dkedt_interp = data_op[:,id_uH] + data_op[:,id_ww]
    dpedt_interp = data_op[:,id_bb]
    epsilon_interp = (-data_ip[:,id_bw]) - dkedt_interp  # -bw - d(ke)/dt
    chi_interp     =  data_ip[:,id_bw]   - dpedt_interp  # bw - d(pe)/dt
    t_ke = (data_ip[:,id_uH]+data_ip[:,id_ww]) / epsilon_interp # ke / epsilon
    t_pe = data_ip[:,id_bb] / chi_interp # pe / chi
    data_ip = np.append( data_ip, np.array([epsilon_interp, chi_interp]).T, axis=1 )
    data_ip_varnames += ['epsilon', 'chi']

if add_IP_Fr_Gn:
    data_ip = np.append( data_ip, np.array([Frhinterp, Gninterp]).T, axis=1 )
    data_ip_varnames += ['Frh', 'Gn']

if add_IP_time:
    data_ip = np.append( data_ip, np.array([data_time]).T, axis=1)
    data_ip_varnames += time_varname

# force normalize input/output
if normIO:
    data_op = ( data_op - np.min(data_op, axis=0) ) / (np.max(data_op, axis=0) - np.min(data_op, axis=0))

Nsampls   = len(data_time)

print(f'# input variables:\t{data_ip.shape[1]}\n# output variables:\t{data_op.shape[1]}\n# samples:\t\t{Nsampls:,}')

print(f'seq len for 1T length of data = {round(1/np.diff(data_time)[0])}')

# ## Generate LSTM data
data_ip_lstm = np.zeros( (Nsampls-seq_len+1, seq_len, data_ip.shape[1]) )
data_op_lstm = np.zeros( (Nsampls-seq_len+1, data_op.shape[1]) )

for i in range(Nsampls-seq_len+1):
    data_ip_lstm[i,:,:] = data_ip[i:i+seq_len,:]
    data_op_lstm[i,:]   = data_op[i+seq_len-1,:]

data_time_lstm = data_time[seq_len-1:]
Nsampls = len(data_time_lstm)

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

print(f'Shape of arrays:\ndatatrain_IP = {datatrain_IP.shape}\ndatatrain_OP = {datatrain_OP.shape}\ndatatrain_time = {datatrain_time.shape}')
# ## Save to npz file
if savedata:
    np.savez(savefilename,
             Ntrain=Ntrain, Ntest=Ntest, 
             datatrain_IP=datatrain_IP, datatrain_OP=datatrain_OP, 
             datatest_IP=datatest_IP, datatest_OP=datatest_OP,
             data_ip_varnames=data_ip_varnames, data_op_varnames=data_op_varnames,
             datatrain_time=datatrain_time, datatest_time=datatest_time,
             nu=case_info.nu, drhobardz=case_info.drhobardz, accel=case_info.accel, rho0=case_info.rho0, totalE=totalE,
             Frh=Frhinterp, Gn=Gninterp, data_time=data_time)
    print("===========Data saved===========")
else: 
    print("===========Data not saved===========")

