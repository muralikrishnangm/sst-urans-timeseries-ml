# Python script for data acquisition functions for RANS data for homogeneous SST**

import numpy as np
import csv
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# ## Loading data
def isfloat(value):
    try :
        float(value)
        return True
    except :
        return False

# function to load file and extract values & headers
def load_data(filename):
    # load data
    with open (filename, 'r') as f:
        data = [row for row in csv.reader(f,delimiter=' ')]
        f.close

    # read lines into headers & variable values
    data_header = []
    data_values = [['-1'],]
    flag = 0 # flag to only read first header
    for i in range(len(data)):
        if len(data[i]) > 0:  # if it is not an empty row
            # see if it is a string or not
            if not isfloat(data[i][0]):
                if len(data[i][0]) == 1 and data[i][0] == '%' and flag == 0: # normal header starting with %
                    data_header += [data[i][1:],]
                elif len(data[i][0]) > 1 and data[i][0][0] == '%' and flag == 0: # header with % combined with a string
                    data_header += [data[i][1:],]
                elif flag == 0: # still first header
                    if isfloat(data[i][1]): # first value row, which starts with the string ''
                        # print(data[i])
                        data[i] = data[i][1:]
                        flag = 1 # finished reading first header
                    else:
                        print(f'This is a wierd header: {data[i][0]}')
                else : # it is value row but starting with the string, ''
                    # print(data[i])
                    data[i] = data[i][1:]
                    flag = 1 # finished reading first header
            if isfloat(data[i][0]): # if the current row only has values
                # flag = 1 # finished reading first header
                # print(data[i])
                # print('These are values')
                if np.double(data[i][0]) > np.double(data_values[-1][0]): # skip duplicate data based on time value
                    data_values += [data[i][0:],]
                
        # =======================================
        # if data[i][0] == '%' and flag == 0:
        #     # print('This is a header')
        #     data_header += [data[i][1:],]
        # elif data[i][0] != '%':
        #     # print(data[i][0][0])
        #     if not isfloat(data[i][0]):  # rstats has a `` as the first element of the row
        #         data[i] = data[i][1:]
        #     flag = 1 # finished reading first header
        #     # print(data[i])
        #     # print('These are values')
        #     if np.double(data[i][0]) > np.double(data_values[-1][0]): # skip duplicate data based on time value
        #         data_values += [data[i][0:],]
    data_values = data_values[1:]

    # save variable values
    ntime = len(data_values)  # num of time instants
    nvars = 0                 # num of variables
    for i in range(len(data_values[0])):
        if data_values[0][i] != '':
            nvars += 1
    values = np.ndarray((ntime,nvars))
    for i in range(ntime):
        k = 0
        # if data_values[i][0] > data_values[i-1][0]: # skip duplicate data based on time value
        for j in range(len(data_values[i])):
            if data_values[i][j] != '':
                values[i,k] = np.double(data_values[i][j])
                k += 1
    
    # save header
    nhead = len(data_header)
    header = []
    # save all individual strings from all the headers in the full file
    for i in range(nhead):
        for j in range(len(data_header[i])):
            if data_header[i][j] != '':
                header += [data_header[i][j],]
    # save variable names: the last `nvars` strings from all the headers in the full file
    varnames = header[len(header)-nvars:]
    # in some files, a comma is followed by the var names. Take the comma out
    for i in range(len(varnames)): 
        if len(varnames[i])>1 and varnames[i][-1]==',':
            varnames[i] = varnames[i][:-1]
    # print(header)
    
    # return data
    return values, ntime, nvars, varnames


# ## Data class
class statsData():
    def __init__(self, filename):
        
        # print('Loading data...')
        self.values, self.ntime, self.nvars, self.varnames = load_data(filename)
        
        # ID variables
        dtype = [('Name', (np.str_, 10)), ('ID', np.int32)]
        self.IDs=np.empty(self.nvars,dtype=dtype)        
        for i in range(self.nvars):
            self.IDs[i]['Name'] = self.varnames[i]
            self.IDs[i]['ID']   = i

        # time derivatives
        self.IDt = self.IDs['ID'][np.where( (self.IDs['Name']=='t') )[0][0]]
        # print('Computing time derivatives...')
        self.values_dt = np.gradient(self.values[:,np.where(self.IDs['Name']!='t')[0]], self.values[:,self.IDt], edge_order=2, axis=0)
        
        # # print data summary
        # print(f'Loaded {filename} data\n# time instants = {self.ntime}\n# variables =  {self.nvars}\nvariable names = {self.varnames}')
        # for i in range(self.nvars):
        #     print('Variable %d is %s' % (self.IDs[i]['ID'], self.IDs[i]['Name']))
            
    def ID(self, var):
        return self.IDs['ID'][np.where(self.IDs['Name']==var)[0][0]]

    # number of elements in the dataset
    def __len__(self):
        return len(self.ntime)

# ## Get information/metadata of case
class get_case_info():
    def __init__(self, case, machine_dir):
        if case == 1:  # F4R32
            # NOTE: Duplicate vstats to vstats2 and change Gn -> Rb throughout the file
            self.casename  = 'F4R32'
            self.HDdir     = machine_dir+'/F4R32/RANSdata/'
            self.Fr        = 4
            self.Re        = 3200
            self.nu        = 3.125000000000000e-04
            self.drhobardz = -2.467400000000000e+00
            self.accel     = 1
            self.rho0      = 1
            self.Tskip     = 0.0        # skip initial transient data till time T
            self.Gnfinal   = -1.0       # skip final decaying data till time T
            self.normTime  = False     # normalize/re-scale time to dimensional time
            self.filtgrad  = False     # filtering of time derivatives due to discontinuities from numerical differentiation 
        elif case == 12:  # F4R64
            # NOTE: Duplicate vstats to vstats2 and change Gn -> Rb throughout the file
            # Delete last line of rstats
            self.casename  = 'F4R64'
            self.HDdir     = machine_dir+'/F4R64/RANSdata/'
            self.Fr        = 4
            self.Re        = 6400
            self.nu        = 1.562500000000000e-04
            self.drhobardz = -2.467400000000000e+00
            self.accel     = 1
            self.rho0      = 1
            self.Tskip     = 0.2        # skip initial transient data till time T
            self.Gnfinal   = -1.0       # skip final decaying data till time T
            self.normTime  = False     # normalize/re-scale time to dimensional time
            self.filtgrad  = False
        elif case == 13:  # F2R32
            # NOTE: Duplicate vstats to vstats2 and change Gn -> Rb throughout the file
            self.casename  = 'F2R32'
            self.HDdir     = machine_dir+'/F2R32/RANSdata/'
            self.Fr        = 2
            self.Re        = 3200
            self.nu        = 3.125000000000000e-04
            self.drhobardz = -2.467400000000000e+00
            self.accel     = 4
            self.rho0      = 1
            self.Tskip     = 0.0        # skip initial transient data till time T
            self.Gnfinal   = -1.0       # skip final decaying data till time T
            self.normTime  = False     # normalize/re-scale time to dimensional time
            self.filtgrad  = False
        elif case == 101: # pendulum
            self.casename  = 'pendulum'
            self.HDdir     = machine_dir+'/Pend'
        else:
            raise Exception(f"Case number {case} is invalid.")
        if case != 101:
            self.N = np.sqrt( -(self.accel/self.rho0) * self.drhobardz )
            self.U_L = self.Fr*self.N/(2*np.pi)
            self.UL = self.Re*self.nu
            self.U = np.sqrt( self.U_L * self.UL )
            self.L = self.UL / self.U