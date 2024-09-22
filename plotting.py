"""
Plotting routines within NODE training
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_train(dataset,args):
    
    
    t, test_t, true_y, pred_y, y_varnames, energy_vars, pred_der, dy_varnames, epoch, l_train, l_val, learn_rate = args
    
    if dataset == 'pendulum':
        
        #idx1 = [0,2]
        #idx2 = [1,3]
        #fig, axs = plt.subplots(2,2,figsize=(10,10))
        #fig.suptitle('State Variables Taylor-Green')
        #axs = axs.ravel()
        #for i in range(2):
        #    axs[idx1[i]].plot(t,true_y[:,i])
        #    axs[idx1[i]].plot(test_t,pred_y[:,i].cpu())
        #    axs[idx2[i]].plot(t,energy_vars[:,i])
        #    axs[idx2[i]].plot(t,pred_der[:,i])
        #    #axs[i].set_xlim([0,10])
        #    axs[i].set_xlabel('T')
        
        pred_y = pred_y.cpu()
        
        idx1 = [0,2,4]
        idx2 = [1,3,5]
        
        rvals = np.zeros(7)
        
        for i in range(3):

            rvals[idx1[i]] = r2_score(true_y[:,i],pred_y[:,i])
            rvals[idx2[i]] = r2_score(energy_vars[:,i],pred_der[:,i])
            rvals[6] = r2_score(true_y[:,0]+true_y[:,1],pred_y[:,0]+pred_y[:,1])

        rvals = np.round_(rvals, decimals = 4)
            
        plt.figure()
        ls = y_varnames
        le = dy_varnames
        fig, axs = plt.subplots(4,2,figsize=(12,10))
        axs = axs.ravel()
        for i in range(3):
            axs[idx1[i]].plot(t,true_y[:,i],'-r',label='True')
            axs[idx1[i]].plot(test_t,pred_y[:,i],'--b',label='Neural ODE, $R^2$ = {}'.format(rvals[idx1[i]]))
            axs[idx1[i]].legend(frameon=False)
            axs[idx2[i]].plot(t,energy_vars[:,i],'-r',label='True')
            axs[idx2[i]].plot(test_t,pred_der[:,i],'--b',label='Neural ODE, $R^2$ = {}'.format(rvals[idx1[i]]))
            axs[idx2[i]].legend(frameon=False)
            axs[idx1[i]].set_ylabel(ls[i])
            axs[idx2[i]].set_ylabel(le[i])
        axs[5].set_xlabel('T')
        axs[6].set_xlabel('T')
        axs[6].plot(t,true_y[:,0]+true_y[:,1],'-r',label='True')
        axs[6].plot(test_t,pred_y[:,0]+pred_y[:,1],'--b',label='Neural ODE, $R^2$ = {}'.format(rvals[6]))
        axs[6].legend(frameon=False)
        axs[6].set_ylabel('Total Energy')
        axs[7].axis('off')
 
        
    if dataset == 'RANS':

        fig, axs = plt.subplots(5,2,figsize=(10,10))
        axs = axs.ravel()
        for i in range(4):
            axs[i].plot(t,true_y[:,i])
            axs[i].plot(test_t,pred_y[:,i].cpu())
            axs[i].set_ylabel(y_varnames[i])
            axs[i+4].plot(t,energy_vars[:,i])
            axs[i+4].plot(t,pred_der[:,i])
            axs[i+4].set_ylabel(dy_varnames[i])
        axs[6].set_xlabel('T'); axs[7].set_xlabel('T');

        axs[8].plot(np.array(epoch), np.array(l_train), color = 'tab:blue',linestyle = 'solid', label = 'Train')
        axs[8].plot(np.array(epoch), np.array(l_val), color = 'tab:red',linestyle = 'dashed', label = 'Validation')
        axs[8].set_xlabel('Epoch')
        axs[8].set_ylabel('Loss')
        axs[8].set_yscale('log')
        axs[8].legend(ncol=2,loc='upper center', frameon=False)

        axs[9].plot(np.array(epoch), np.array(learn_rate), color = 'tab:blue',linestyle = 'solid', label = 'Exponential')
        axs[9].set_xlabel('Epoch')
        axs[9].set_ylabel('Learning Rate')
        axs[9].set_yscale('log')
        axs[9].legend(ncol=1,loc='upper center', frameon=False)

        plt.show()