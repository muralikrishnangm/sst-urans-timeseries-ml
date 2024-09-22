import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import interpolate as itp
import argparse

parser = argparse.ArgumentParser(description='Select dataset to generate ')
parser.add_argument('dataset', type=str ,help='RANS or Pendulum Dataset')

# Pendulum arguments
parser.add_argument('-b', type=float, default = 0.5 ,help='damping coeff (b)')
parser.add_argument('-m', type=float, default = 1 ,help='mass (m)')
parser.add_argument('-g', type=float, default = 25 ,help='gravity (g)')
parser.add_argument('-l', type=float, default = 1 ,help='length (l)')
parser.add_argument('--theta_zero', type=float, default = np.pi/4 ,help='initial condition for theta')
parser.add_argument('--omega_zero', type=float, default = 0 ,help='initial condition for omega')
parser.add_argument('--tmin', type=float, default = 0 ,help='initial time')
parser.add_argument('--tmax', type=float, default = 30 ,help='final time')
parser.add_argument('--npoints', type=int, default = 1000 ,help='number of time points')
parser.add_argument('--method', type=str, default = 'RK45' ,help='ode solver')

# Sequencing arguments
parser.add_argument('--tperiod', type=float, default = 1 ,help='time period held constant for interpolation')
parser.add_argument('--win', type=int, default = 34 ,help='number of points per time period')
    
args = parser.parse_args()
    
class pendulum():
    
    def model_unforced(b=1,m=1,g=36,l=1,y0=[np.pi/4,0],tmin=0,tmax=30,npoints=18000,method='RK45'):
        
        # pendulum differential equation
        
        def pend(t, y, b, m, g, l):
            #print('i')
            #print(t,y)
            theta, omega = y
            dydt = [omega, -(b/m)*np.sqrt(l/g)*omega - np.sin(theta)]
            return dydt
        
        t_span = [tmin, tmax]
        t = np.linspace(tmin, tmax, npoints)
        
        # integrate the ode using solve_ivp
        sol = solve_ivp(pend, t_span, y0, method=method, args = (b,m,g,l), t_eval = t, rtol = 1e-8)
        
        # state variables
        theta = sol.y[0,:]
        omega = sol.y[1,:]
        t = sol.t
        
        energy = np.zeros([len(t),3])
        energy[:,0] = (1/2)*omega**2 # KE
        energy[:,1] = 1-np.cos(theta) # PE
        energy[:,2] = np.sin(theta)*omega # gw
        
        d_energy = np.zeros([len(t),3])
        d_energy[:,0] = -(b/m)*np.sqrt(l/g)*omega**2-np.sin(theta)*omega # d_KE
        d_energy[:,1] = np.sin(theta)*omega # d_PE
        d_energy[:,2] = (np.cos(theta)*omega**2-(b/m)*np.sqrt(l/g)*np.sin(theta)*omega-np.sin(theta)**2) # d_gw
        
        return y0, t_span, t, theta, omega, energy, d_energy
    
    def model_forced(b=0.25,c=1,om_d=2,y0=[0, 0.1],tmin=0,tmax=30,npoints=200,method='RK45'):
        
        # pendulum differential equation
        
        def pend(t, y, b, c, om_d):
            #print('i')
            #print(t,y)
            theta, omega = y
            #dydt = [omega, k*omega - m*g*l*np.sin(theta)]
            dydt = [omega, -np.sin(theta) - b*omega + c/np.exp(0.23*t)*np.cos(om_d*t)]
            return dydt
        
        t_span = [tmin, tmax]
        t = np.linspace(tmin, tmax, npoints)
        
        # integrate the ode using solve_ivp
        sol = solve_ivp(pend, t_span, y0, method=method, args = (b, c, om_d), t_eval = t, rtol = 1e-8)
        
        # state variables
        theta = sol.y[0,:]
        omega = sol.y[1,:]
        
        t = sol.t

        return y0, t_span, t, theta, omega
        
# class RANS():
    
#     print('needs to be implemented: RANS')
    
#################### Scaling Dataset ###############################

class normalize():

    def scale_training(data_x, data_y, experimentID=None, message='None'):

        scaler = MinMaxScaler()
        x_scaler = scaler.fit(data_x)
        x_scaled = x_scaler.transform(data_x)

        #y0 = true_y[0,:]

        scaler = MinMaxScaler()
        y_scaler = scaler.fit(np.expand_dims(data_y,1))
        y_scaled = np.squeeze(y_scaler.transform(np.expand_dims(data_y,1)))
        
        filex=open('./models/x_scaler_{}.pkl'.format(experimentID), 'wb')
        dump(x_scaler, filex)
        filex.close()
        filey=open('./models/y_scaler_{}.pkl'.format(experimentID), 'wb')
        dump(y_scaler, filey)
        filey.close()
        
        return message, print('Scaler Object Saved')
    
    def scale(data_x, data_y, experimentID=None):
        
        # open data
        filex = open('./models/x_scaler_{}.pkl'.format(experimentID), 'rb')
        x_scaler = pickle.load(filex)
        filex.close()
                    
        filey = open('./models/y_scaler_{}.pkl'.format(experimentID), 'rb')
        y_scaler = pickle.load(filey)
        filey.close()
                     
        data_x_scaled = x_scaler.transform(data_x)
        data_y_scaled = np.squeeze(y_scaler.transform(np.expand_dims(data_y,1)))
                     
        return data_x_scaled, data_y_scaled
    
    def unscale(data_x, data_y, experimentID=None):
        
        # open data
        filex = open('./models/x_scaler_{}.pkl'.format(experimentID), 'rb')
        x_scaler = pickle.load(filex)
        filex.close()
                    
        filey = open('./models/y_scaler_{}.pkl'.format(experimentID), 'rb')
        y_scaler = pickle.load(filey)
        filey.close()
                     
        data_x_scaled = x_scaler.inverse_transform(data_x)
        data_y_scaled = np.squeeze(y_scaler.inverse_transform(np.expand_dims(data_y,1)))
                    
        return data_x_scaled, data_y_scaled
    
class prepare():
    
    def sequence(input_data, output_data, t, tperiod=1, win=32):
        
        energy = input_data
        d_energy = output_data  # dy/dt (for LSTM model)
        tperiod = tperiod
        dt = tperiod/win # dt # where to start sequencing in the dataset
         
        # interpolate full dataset for resolution
        f = itp.interp1d(t,energy.T)
        f_d = itp.interp1d(t,d_energy.T)

        # first lets split up into time intervals of length t_period
        t_max = t[-1] # max time point for data
        num_obs = int((t_max-tperiod)/dt)+1 # total number of sequence splits
        times_full = np.arange(0, t_max, dt)
        
        # populate matrices
        labels = np.zeros((num_obs, win, energy.shape[1]))
        inputs = np.zeros((num_obs, energy.shape[1]))
        outputs = np.zeros((num_obs, d_energy.shape[1]))
        seq_time = np.zeros((num_obs, win))

        plt.figure()
        for i in range(num_obs):
            times = times_full[i:i+win]
            true_y = f(times).T
            true_dy = f_d(times).T
            inputs[i,:]   = true_y[0,:]
            labels[i,:,:] = true_y[:,:]
            outputs[i,:]  = true_dy[-1,:]
            seq_time[i,:] = times[:]
            plt.plot(times,true_y[:,0])

        print('done')
        print(np.shape(inputs))
        print(np.shape(labels))
        print(np.shape(outputs))
        
        return inputs, labels, outputs, times_full, seq_time

if __name__ == '__main__':
    
    datafolder = './Data_raw/'
    
    if args.dataset == 'pendulum':
    
        y0,t_span,t,theta,omega,energy,d_energy = pendulum.model_unforced(b=args.b,
                                                                          m=args.m,
                                                                          g=args.g,
                                                                          l=args.l,
                                                                          y0=[args.theta_zero,args.omega_zero],
                                                                          tmin=args.tmin,
                                                                          tmax=args.tmax,
                                                                          npoints=args.npoints,
                                                                          method=args.method)
        
        inputs, labels, outputs, times_full, seq_time = prepare.sequence(energy, d_energy, t, tperiod=args.tperiod, win=args.win)
        y0 = energy[0,:].T
        true_y = energy
        
        data = 'Pend/'
        filename = '{}_b{}_m{}_g{}_l{}_theta{}_omega{}_t{}_npts{}_sol{}_tp{}_win{}.npz'.format(
            args.dataset,
            args.b,
            args.m,
            args.g,
            args.l,
            np.round_(args.theta_zero,4),
            np.round_(args.omega_zero,4),
            str(args.tmin)+'-'+str(args.tmax),
            args.npoints,
            args.method,
            args.tperiod,
            args.win)
        # print(np.shape(labels))
        # print(np.shape(inputs))
                                                                                        
        path = datafolder+data+filename
    
        np.savez(path, y0=y0, t=t, theta=theta, omega=omega, energy=energy, d_energy=d_energy, 
                 inputs=inputs, labels=labels, outputs=outputs, times_full=times_full, seq_time=seq_time)
        
    else:
        raise Exception('Needs to be implemented')
    


