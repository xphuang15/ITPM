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
