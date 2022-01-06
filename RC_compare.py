# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:40:53 2020

@author: XiaopengHuang
"""

import numpy as np

def compare(y_data_ref,y_data):
    err = y_data - y_data_ref
    data_RMSE = np.sqrt(np.mean(np.square(err),axis=0))
    err_max = np.amax(abs(err),axis=0)
    
    return [data_RMSE,err_max]

def compare_timeseries(time_data,y_data_ref,y_data,time_tholds = [300,600,1200]):
    err = y_data - y_data_ref
    data_RMSE = np.sqrt(np.mean(np.square(err),axis=0))
    err_end = err[-1,:]
    err_max = np.amax(err,axis=0)
    err_min = np.amin(err,axis=0)
    for i in range(np.shape(err)[1]):
        if abs(err_max[i]) < abs(err_min[i]):
            err_max[i] = err_min[i]
    
    ## assuming time_data unit is second, has the same size as y_data
    ## assuming time_data is already resampled (1s interval at least)
    err_all = []
    for time_thold in time_tholds:
        idx_time = np.abs(time_data - time_thold).argmin(axis=0)
        err_time_tholds = err[idx_time,:]
        err_all.append(err_time_tholds)
    err_all.append(err_end)
    err_all.append(err_max)
    err_all.append(data_RMSE)
    
    return np.asarray(err_all)