# -*- coding: utf-8 -*-
"""
Created on Sun Aug  18 10:41:17 2020

@author: XiaopengHuang
"""

import os, sys
import numpy as np
from scipy import interpolate

def resample(time_data,sig_data,kind='linear'):
    ### Resample in higher frequency for power and temperature with a constant 
    ### sampling rate. Assuming the power input always does step change

    # make the start time as 0 if the time is in log scale with a small positive start time
    if (time_data[0] > 1e-8) & (time_data[0] < 1):
        time_data = np.insert(time_data,0,0)
        sig_data_init = min(sig_data[0])
        sig_data = np.insert(sig_data,0,sig_data_init,axis=0)

    start_time = time_data[0]
    end_time = time_data[-1]
        
    ts_min = np.round(min(time_data[1:-1] - time_data[0:-2]),2) # only round to x.xx s
    # assuming unit of time data is second
    ts_thold_max = 5
    ts_thold_min = 0.5
    ts_min = min(ts_min, ts_thold_max)
    ts_min = max(ts_min, ts_thold_min)
    time_data_new = np.round(np.arange(start_time,end_time + ts_min,ts_min),2)
    if time_data_new[-1] > end_time:
        time_data_new = np.delete(time_data_new,-1)
    fun_sig = interpolate.interp1d(time_data,sig_data,kind=kind,axis=0)
    sig_data_new = fun_sig(time_data_new)

    return [time_data_new,sig_data_new]
