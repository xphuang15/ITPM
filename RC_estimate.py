# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:44:42 2020

@author: XiaopengHuang
"""

import numpy as np

### Estimate the sensor temperatures at different power scenarios

## for step power input, use direct vector method
def infer_step(time_data,Zall,pwr_valid):
    
    ## convert Zall & pwr_valid into numpy array if it is not one
    if not isinstance(Zall,np.ndarray):
        Zall = np.array(Zall)
    
    if not isinstance(pwr_valid,np.ndarray):
        pwr_valid_arr = np.array(pwr_valid)
    else:
        pwr_valid_arr = pwr_valid
    
    Zia = Zall  # select the impedance curves for sensor temperature calculation
    ## get sensor temperature from Ti = Zia X P + Ta
    dTia = np.einsum('ijk,mj->mki', Zia, pwr_valid_arr)
        
    return dTia

## for dynamic power input, use discretized convolution method (conv(impulse response*p))
def infer_dyn(time_data,Zall,pwr_valid,y_data_valid):
    
    ## convert Zall & pwr_valid into numpy array if it is not one
    if not isinstance(Zall,np.ndarray):
        Zall = np.array(Zall)
    
    if not isinstance(pwr_valid,np.ndarray):
        pwr_valid_arr = np.array(pwr_valid)
        
    ## loop through the validation dataset
    Zia = Zall  # select the impedance curves for sensor temperature calculation
    for idx_valid in range(pwr_valid_arr.shape[0]):
        pwr_valid = pwr_valid_arr[idx_valid]

        #= use linear-scale time instead of log-scale time
        
        time_diff = np.insert(np.diff(time_data),0,time_data[0])
        Zia_diff = np.insert(np.diff(Zia),0,Zia[0])
        
        # tmp_trapezoidal = convolve(pwr_valid,np.gradient(Zia,time_data),time_diff)
        # y_data_infer_trapezoidal = tmp_trapezoidal + Ta
        
        conv_mode = 'full' # 'full', 'valid','same'
        
        tmp = np.convolve(pwr_valid,Zia_diff,mode=conv_mode)
        dTia = tmp[0:len(pwr_valid)]
                
    return dTia