# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:44:42 2020
1.21.2021 Add leakage power update in each time step

@author: XiaopengHuang
"""

import numpy as np

### Estimate the sensor temperatures at different power scenarios

## for step power input, use direct vector method
def infer_step(time_data,Zall,pwr_infer):
    
    ## convert Zall & pwr_infer into numpy array if it is not one
    if not isinstance(Zall,np.ndarray):
        Zall = np.array(Zall)
    
    if not isinstance(pwr_infer,np.ndarray):
        pwr_infer_arr = np.array(pwr_infer)
    else:
        pwr_infer_arr = pwr_infer
    
    Zia = Zall  # select the impedance curves for sensor temperature calculation
    ## get sensor temperature from Ti = Zia X P + Ta
    dTia = np.einsum('ijk,mj->mki', Zia, pwr_infer_arr)
        
    return dTia

## for dynamic power input, use discretized convolution method (conv(impulse response*p))
def infer_dyn(time_data,Zall,pwr_infer,y_data_valid):
    
    ## convert Zall & pwr_infer into numpy array if it is not one
    if not isinstance(Zall,np.ndarray):
        Zall = np.array(Zall)
    
    if not isinstance(pwr_infer,np.ndarray):
        pwr_infer_arr = np.array(pwr_infer)
    else:
        pwr_infer_arr = pwr_infer
        
    ## loop through the validation dataset
    Zia = Zall  # select the impedance curves for sensor temperature calculation
    dTia_allcases = []
    for idx_infer in range(pwr_infer_arr.shape[0]):
        pwr_infer = pwr_infer_arr[idx_infer]

        #= use linear-scale time instead of log-scale time
        
        time_diff = np.insert(np.diff(time_data),0,time_data[0])
        Zia_diff = np.insert(np.diff(Zia),0,Zia[0])
        
        # tmp_trapezoidal = convolve(pwr_infer,np.gradient(Zia,time_data),time_diff)
        # y_data_infer_trapezoidal = tmp_trapezoidal + Ta
        
        conv_mode = 'full' # 'full', 'valid','same'
        
        tmp = np.convolve(pwr_infer,Zia_diff,mode=conv_mode)
        dTia = tmp[0:len(pwr_infer)]
        dTia_allcases.append(dTia)
                
    return dTia_allcases

## for dynamic temperature-dependent power input (consider leakage), use recursive method
## Tips: a timestep of 0.5-5s almost has the same infer results.
## To-Do: 
## 1. T-P convergence within a timestep
## 2. non-linear effect caused by piece-wise linear approximation
## 3. adapative timestep, critical for dynamic simulation
## 4. improve the speed by using elegant algorithms for convolution summation 
def infer_lkg(time_data,Zall,pwr_infer,list_fun_pwr_total_scale,idx_T_src):
    
    ## convert Zall & pwr_infer into numpy array if it is not one
    if not isinstance(Zall,np.ndarray):
        Zall = np.array(Zall)
    
    if not isinstance(pwr_infer,np.ndarray):
        pwr_infer_arr = np.array(pwr_infer)
    else:
        pwr_infer_arr = pwr_infer
        
    ## loop through the validation dataset
    Zia = Zall  # select the impedance curves for sensor temperature calculation
    Zia_diff = np.diff(Zia,axis=2) # improve 30% speed comparing to put it in k-loop
    dTia_allcases = []
    output_steps = 10
    output_step_gap = int((len(time_data) - 1)/output_steps)
    for idx_infer in range(pwr_infer_arr.shape[0]):
        pwr_infer = pwr_infer_arr[idx_infer]
        print(f'========= Working on power case {idx_infer:d}: =========')
        print('[' + ', '.join('%.4f' % v for v in pwr_infer) + ']')
        print('--------------------------------------------------------')
        pwr_series = [pwr_infer]   
        dTia = np.zeros((len(time_data),Zia.shape[0]))
        ## get sensor temperature by equation 2
        for timestep in range(2,len(time_data)+1): # calculate temperature at each timestep
            #= use linear-scale time instead of log-scale time
            #= forward time discretization
            summrand = np.zeros((Zia.shape[0]))
            for k in range(1,timestep):
                summrand += np.einsum('ij,j->i', Zia_diff[:,:,timestep - k - 1], pwr_series[k-1])
            dTia[timestep-1,:] = summrand
            dTia_max = max(summrand)
            
            ## update the power of each source based on new temperature
            pwr_new_step = np.zeros(pwr_infer.shape)
            pwr_scale_all = []
            for idx_src in range(len(idx_T_src)):
                fun_pwr_total_scale = list_fun_pwr_total_scale[idx_infer][idx_src]
                pwr_scale = fun_pwr_total_scale(summrand[idx_T_src[idx_src]])
                pwr_new_step[idx_src] = pwr_scale*pwr_infer[idx_src]
                pwr_scale_all.append(pwr_scale)
            pwr_series.append(pwr_new_step)
            if timestep%output_step_gap == 0:
                print('Timestep %d, time %g s, dTmax = %.3f C, power updated with scale factors below' \
                          %(timestep,time_data[timestep-1],dTia_max))
                print('[' + ', '.join('%.2f' % v for v in pwr_scale_all) + ']')
            
        dTia_allcases.append(dTia)   
    return dTia_allcases