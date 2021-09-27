# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:38:10 2020

@author: XiaopengHuang
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import signal
from scipy.fft import fft,ifft,irfft,fftshift,fftfreq
from scipy.special import gamma
import matplotlib.pyplot as plt
import math
import control as ctrl
import RC_tf

###========================= Calculate impedance matrix ======================
def imped(time_data,y_data_train,pwr_train):
    
    ## get the ambient temperature = initial temperature
    Ta = min(y_data_train[0][0,:]) 
    
    n_sources = len(pwr_train)
    n_sensors = y_data_train[0].shape[1]

    ## get thermal impedance matrix Zia from Ti (including both Tj and Ts) to Ta
    ## dimension (i,j,k), i for Ti (Tj or Ts), j for Pj, k for time
    Zall = [] # include all the impedance curves
    for idx_target in range(n_sensors):
        tmp = []
        for idx_source in range(n_sources):
            Z = (y_data_train[idx_source][:,idx_target] - Ta)/pwr_train[idx_source]
            tmp.append(Z)
        Zall.append(tmp)

    
    # ## get thermal impedance matrix Zjs from Tj to Ts
    # ## dimension (l,i,j,k) or (i,j,k), l for interested Ts, i for Tj, j for Pj, k for time 
    # idx_Tj = [0,1,2]
    # idx_Ts = [3,4,5,6,7,8] # [3,4] for Ts_bot, [5,6,7,8] for Ts_top
    # for idx_Ts_m in idx_Ts:
    #     Zjs = []
    #     for idx_target in idx_Tj:
    #         tmp = []
    #         for idx_source in range(0,n_sources):
    #             Z = (y_data_train[idx_source][:,idx_target] - \
    #                  y_data_train[idx_source][:,idx_Ts_m])/pwr_train[idx_source]
    #             tmp.append(Z)
    #         Zjs.append(tmp)        
    #     Zall.append(Zjs)
        
    return Zall, Ta

###====== Fitting each thermal impedance curve with a Foster RC ladder =======
def fitFo(time_data,Zall,n_stage_range):
    ## take a list of arguments and break it down into two lists 
    ## for the fit function to understand
    def wrapper(t, *args): 
        N = int(len(args)/2)
        R = list(args[0:N])
        tau = list(args[N:2*N])
        return func(t, R, tau)

    ## define the Foster RC ladder function for fitting
    def func(t, R, tau):

        if isinstance(R,float) & isinstance(tau,float): # 1-stage RC ladder
            imped = R*(1-np.exp(-t/tau))
        else:  # multi-stage RC ladder
            if not (isinstance(R,np.ndarray) | isinstance(R,list)):
                raise("Error: 2nd argument R has to be a numpy array or a list")
            if not (isinstance(tau,np.ndarray) | isinstance(tau,list)):
                raise("Error: 3rd argument tau has to be a numpy array or a list")
                
            ## convert R & tau into array if they aren't 
            ## assume 1D array or list
            R = np.array(R)
            tau = np.array(tau)
            
            if (len(R.shape) > 1) | (len(tau.shape) > 1):
                raise("Error: R and tau should be one dimension array or list")
            else:
                if not R.shape[0] == tau.shape[0]:
                    raise("Error: R and tau must have the same length")
                
            n_stage = R.shape[0]
            imped = np.zeros(len(t))
            for idx_stage in range(n_stage):            
                imped += R[idx_stage]*(1-np.exp(-t/tau[idx_stage]))
            
        return imped
    
    
    ## fitting

    RC_fitFo_all = []
    RC_fitFo_cov_all = []
    for Z in Zall:
        RC_fitFo = []
        RC_fitFo_cov = []        
        for idx_n_stage in n_stage_range:
            RC_fitFo_src = []
            RC_fitFo_cov_src = []
            for idx_source in range(len(Z)):
                xdata = time_data
                ydata = Z[idx_source]
                R_init = [10*x/x for x in range(1,idx_n_stage+1)]
                tau_init = [100*x/x for x in range(1,idx_n_stage+1)]
                R_tau_init = R_init + tau_init
                R_lo = [0.05*x/x for x in range(1,idx_n_stage+1)]
                R_up = [1000*x/x for x in range(1,idx_n_stage+1)]
                tau_lo = [0.001*x/x for x in range(1,idx_n_stage+1)]
                tau_up = [10000*x/x for x in range(1,idx_n_stage+1)]
                R_tau_bounds = (tuple(R_lo+tau_lo),tuple(R_up+tau_up))              
                
                popt, pcov = curve_fit(lambda t,*p0:wrapper(t,*p0), xdata, ydata,\
                                       p0=R_tau_init,bounds=R_tau_bounds,maxfev=100000)
                tmp = popt.reshape(2,-1).T
                if idx_n_stage > 1:
                    popt_sort = tmp[tmp[:,1].argsort()]  # sort RC pair according to time constant
                else:
                    popt_sort = tmp
                    
                # print(popt)
                RC_fitFo_src.append(popt_sort)
                RC_fitFo_cov_src.append(pcov)
                
                # plt.plot(xdata, ydata, 'b-', label='data')
                # plt.plot(xdata, func(xdata, popt[0:idx_n_stage],popt[idx_n_stage:2*idx_n_stage]), 'r-',
                #           label='%d-stage RC' % idx_n_stage)
                # plt.xscale('log')
                # plt.xlabel('Time (s)')
                # plt.ylabel('Thermal impedance (K/W)')
                # plt.legend()
                # plt.show()
                
            RC_fitFo.append(RC_fitFo_src)
            RC_fitFo_cov.append(RC_fitFo_cov_src)
            
        RC_fitFo_all.append(RC_fitFo)
        RC_fitFo_cov_all.append(RC_fitFo_cov)
        
    return RC_fitFo_all, RC_fitFo_cov_all

###== Obtain time-constant spectrum and R-tau pairs from thermal impedance curve
def nid(time_data,Z):
    
    ## 1. convert linear time scale to equidistant natural log time scale
    time_data_log10_start = math.log10(max(time_data[0],1e-9))
    time_data_log10_end = math.log10(time_data[-1])
    n_per_decade = 600
    time_data_log10_num = math.floor(n_per_decade*(time_data_log10_end - time_data_log10_start))+1
    time_data_log10_end = (time_data_log10_num-1)/n_per_decade + time_data_log10_start
    time_data_log10 = np.logspace(time_data_log10_start,time_data_log10_end,time_data_log10_num)
    logtime = np.log(10)*np.linspace(time_data_log10_start,time_data_log10_end,time_data_log10_num)
    logtime_int = logtime[1]-logtime[0]
    freqs = fftfreq(len(logtime),logtime_int)
    
    sig_data = np.asarray(Z, dtype=np.float64).T
    fun_sig = interpolate.interp1d(time_data,sig_data,kind='linear',axis=0)
    Z_new = fun_sig(time_data_log10)
    
    Z_grad = np.divide(np.gradient(Z_new,axis=0),np.gradient(logtime))
    wt = np.exp(logtime - np.exp(logtime))
    F_wt = fft(wt)  ## !!! this didn't give a correct result
    
    wr = 2*np.divide(np.exp(2*logtime),(1+np.exp(2*logtime))**2)
    
    wi = np.divide(np.exp(logtime),1+np.exp(2*logtime))
    
    ## 1. direct signal.deconvolve
    ## !!! don't know how to use this function
    R_tau = []
    for idx_source in range(len(Z)):
        quotient, remainder = signal.deconvolve(Z_grad[:,idx_source], wt)
        R_tau.append(quotient)


    
    deconv, remainder = scipy.signal.deconvolve(filtered, gauss)
    #the deconvolution has n = len(signal) - len(gauss) + 1 points
    n = len(signal)-len(gauss)+1
    # so we need to expand it by 
    s = int(np.floor((len(signal)-n)/2))
    #on both sides.
    deconv_res = np.zeros(len(signal))
    deconv_res[s:len(signal)-s-1] = deconv

    
    # ## 2. use NID method, Touzelbaev's paper Eqn. 11
    # F_Z_grad = fft(Z_grad)

    # # freq = np.linspace(0,1.0/(2.0*logtime_int),len(logtime)//2)

    # freq0 = 0.5
    # F_Gauss = np.exp(-(freqs/freq0)**2)
    # F_wt_theo = gamma(1-2j*math.pi*freqs)
    # f_F_wt_theo = ifft(F_wt_theo)
     
       
    # R_tau = ifft([F_Gauss[x]/(F_wt_theo[x]+1e-1)*F_Z_grad[x] for x in range(len(freqs))])
    # R_tau = np.abs(ifft([Gauss_k[x]/W_k[x]*F_Z_grad[x] for x in range(len(freqs))]))
    # R_tau = np.abs(ifft([1/W_k[x]*F_Z_grad[x] for x in range(len(freqs))]))
    # R_tau = np.abs(ifft([1/F_wz[x]*F_Z_grad[x] for x in range(len(freqs))]))
    
    # R_tau = ifft(np.divide(F_Z_grad,F_wz))
    # R_tau = ifft(np.divide(F_Z_grad,W_k))


    ## plot the curves
    plt.plot(logtime,Z_grad)
    plt.legend(['Z_grad'])
    plt.show()
    
    plt.plot(logtime,wt,logtime,wr,logtime,wi)
    plt.legend(['wt','wr','wi'])
    plt.show()
    
    # plt.plot(freqs,F_wt_theo)
    # plt.yscale('log')
    # plt.xlim([0,6])
    # # plt.ylim([1e-12,1])
    # # plt.legend(['wt','F(wt)'])
    # plt.show()
    
    # plt.plot(freqs, F_Gauss,'o')
    # plt.show()
    
    
    # plt.plot(freqs,F_wt,'k',linestyle='dashed',linewidth=1)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Thermal impedance (K/W)')
    # plt.xlim([-1,1])
    # plt.legend()

    
    return logtime,R_tau,Z_grad