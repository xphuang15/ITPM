# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:16:40 2020

@author: XiaopengHuang
"""

import os, sys, scipy
import csv
import numpy as np
from scipy import signal
from scipy import interpolate
from matplotlib import pyplot as plt
import math

### ===================== Setting up the case ================================= 
import RC_network_setup
[path_to_files,filenames_y,filenames_u] = RC_network_setup.setup()
if len(filenames_u) == 1:
    output_filename_ToT = os.path.splitext(filenames_u[0])[0] + "_ToT_results.csv"
elif len(filenames_u) > 1:
    output_filename_ToT = os.path.commonprefix(filenames_u) + "_ToT_results.csv"

### ===================== Import and clean up the data ======================== 
import RC_data_import

uDATA_series = RC_data_import.u_traces_import(path_to_files,filenames_u,'csv',0)
[case_labels, u_data, u_labels] = uDATA_series[0]
n_sources = len(u_labels) # the files include at least n_sources step response file + one validation file
pwr_train = u_data[0,:] ## power of each active source for step response with one active source
pwr_valid = u_data[1:,:] ## power of each active source in each scenario
                        ## FindX2ProCC 4K30UD (900s), 4K30SN (2000s) and 4K60 (900s) 
n_pwr_cases = pwr_valid.shape[0] if len(pwr_valid.shape) > 1 else 1

cleanup_flag = 1  # do data resampling or not
yDATA_series = RC_data_import.y_traces_import(path_to_files,filenames_y,'csv',cleanup_flag)
yDATA_num = len(yDATA_series)

y_data_train = []
[time_data, tmp, y_labels] = yDATA_series[0]
n_sensors = len(y_labels)
## select the temperature sensors need estimation
idx_col = [*range(0,n_sensors)]
# idx_Tj = [10,11,12]
# idx_Ts = [0,4,14,17,19,22]
# idx_col = idx_Tj + idx_Ts

y_data_train.append(tmp[:,idx_col])
y_labels = y_labels[idx_col]
## assume time_data and y_labels are the same from step response files for each source 
for idx_source in range(1,n_sources):
    tmp1 = yDATA_series[idx_source][1][:,idx_col]
    y_data_train.append(tmp1)

## loop through each validation dataset
n_valid = yDATA_num - n_sources
time_data_valid = []
y_data_valid = []
y_labels_valid = []
n_sensors_valid = len(yDATA_series[16][2])
## select the temperature sensors need validation
idx_col_valid = [*range(0,n_sensors_valid)]
for idx_valid in range(n_sources,yDATA_num):
    time_data_valid.append(yDATA_series[idx_valid][0])
    tmp2 = yDATA_series[idx_valid][1][:,idx_col_valid]   
    y_data_valid.append(tmp2)
    y_labels_valid.append(yDATA_series[idx_valid][2])


### ===================== Thermal RC network from impedance curves ============ 
import RC_imped
Zall, Ta = RC_imped.imped(time_data,y_data_train,pwr_train)


### ===================== Estimate the model =================================
import RC_estimate
from finite_convolve import convolve

## estimate the response for step power input, use direct vector method
dTia = RC_estimate.infer_step(time_data,Zall,pwr_valid)
y_data_infer = dTia + Ta

## estimate the time to threshold (ToT)
idx_Tj = [1,4,9,10,11,14,19,22,26,27,28,29,30,31,32,37,40,45]
idx_Tmid = [25,42]
idx_Ts = [x for x in range(n_sensors) if x not in (idx_Tj+idx_Tmid)]
Ts_labels = y_labels[idx_Ts]
Ts_tholds = [39.0,40.0,42.0,43.0,44.0,45.0] # unit Celsius
Tj_labels = y_labels[idx_Tj]
# Tj_thold = 90.0  # unit Celsius
Zsa = [Zall[x] for x in idx_Ts]
Zja = [Zall[x] for x in idx_Tj]
## for step power input and skin temperature 
dTsa = RC_estimate.infer_step(time_data,Zsa,pwr_valid)
y_data_infer_Ts = dTsa + Ta
## for step power input and skin temperature 
dTja = RC_estimate.infer_step(time_data,Zja,pwr_valid)
y_data_infer_Tj = dTja + Ta

ToT_Ts = []
ToT_Ts_labels = []
Tj_at_ToT_Ts = []
for idx_pwr_case in range(n_pwr_cases):
    ToT_Ts_per_case = []
    ToT_Ts_per_case_labels = []
    Tj_at_ToT_Ts_per_case = []
    for Ts_thold in Ts_tholds:
        ## assume time_data is already resampled (1s interval at least)
        idx_time = np.abs(y_data_infer_Ts[idx_pwr_case] - Ts_thold).argmin(axis=0)
        ToT_Ts_every_sensor = time_data[idx_time]
        idx_sensor = ToT_Ts_every_sensor.argmin()
        ToT_Ts_tmp = ToT_Ts_every_sensor[idx_sensor]
        ToT_Ts_per_case.append(ToT_Ts_tmp)
        ToT_Ts_per_case_labels.append(Ts_labels[idx_sensor])
        
        idx_time_Tj = np.abs(time_data - ToT_Ts_tmp).argmin(axis=0)
        Tj_at_ToT_Ts_tmp = y_data_infer_Tj[idx_pwr_case][idx_time_Tj]
        Tj_at_ToT_Ts_per_case.append(Tj_at_ToT_Ts_tmp)
    ToT_Ts.append(ToT_Ts_per_case)
    ToT_Ts_labels.append(ToT_Ts_per_case_labels)
    Tj_at_ToT_Ts.append(Tj_at_ToT_Ts_per_case)

ToT_Ts = np.array(ToT_Ts).astype(str)
ToT_Ts_labels = np.array(ToT_Ts_labels)


print("Writing result file ...")

my_comment = "# time to threshold (s) of each power case for different thresholds"
header_list = ["scenarios"]
time_header = ["T_thold="+str(x)+"C" for x in Ts_tholds] # ignore time=0
time_header = time_header + [" "] + time_header
header_list.extend(time_header)
my_delim = ","
my_header = my_delim.join(np.array(header_list))
my_header = '\n'.join([my_comment,my_header])
output = []
for i in range(n_pwr_cases):
    output.append([case_labels[i+1]] + ToT_Ts[i,:].tolist() + [" "] + ToT_Ts_labels[i,:].tolist())
np.savetxt(output_filename_ToT,output,fmt='%s',delimiter=my_delim,\
           header=my_header,comments='')
