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
    output_filename_cmp = os.path.splitext(filenames_u[0])[0] + "_compare_results.csv"
elif len(filenames_u) > 1:
    output_filename_ToT = os.path.commonprefix(filenames_u) + "_ToT_results.csv"
    output_filename_cmp = os.path.commonprefix(filenames_u) + "_compare_results.csv"

### ===================== Import and clean up the data ======================== 
import RC_data_import

uDATA_series = RC_data_import.u_traces_import(path_to_files,filenames_u,'csv',0)
[case_labels, u_data, u_labels] = uDATA_series[0]
n_sources = len(u_labels) # the files include at least n_sources step response file + one validation file
idx_T_src = u_data[0,:].astype(int) # index of the temperature of each active source
pwr_train = u_data[1,:] # power of each active source for step response with one active source
## the format of the rest of u_data (power) depends on the y_data (temperature)

cleanup_flag = 1  # do data resampling or not
yDATA_series = RC_data_import.y_traces_import(path_to_files,filenames_y,'csv',cleanup_flag)
yDATA_num = len(yDATA_series)
n_cases_valid = yDATA_num - n_sources

## reading the rest of u_data
pwr_infer = u_data[2::2,:] # power of each active source in each scenario for inference
pwr_valid = pwr_infer[0:n_cases_valid,:] # power of each active source in the scenario for validation (has CFD yData) 
pwr_est = pwr_infer[n_cases_valid:,:] # power of each active source in the scenario for estimation (no CFD yData) 
lkg_ratios = u_data[3::2,:] # leakage ratio of each active source in each scenario

n_cases_infer = pwr_infer.shape[0] if len(pwr_infer.shape) > 1 else 1
n_cases_est = n_cases_infer - n_cases_valid

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
if n_cases_valid > 0:
    time_data_valid = []
    y_data_valid = []
    y_labels_valid = []
    n_sensors_valid = len(yDATA_series[n_sources][2])
    ## select the temperature sensors need validation
    idx_col_valid = [*range(0,n_sensors_valid)]
    for idx_valid in range(n_sources,yDATA_num):
        time_data_valid.append(yDATA_series[idx_valid][0])
        tmp2 = yDATA_series[idx_valid][1][:,idx_col_valid]   
        y_data_valid.append(tmp2)
        y_labels_valid.append(yDATA_series[idx_valid][2])


### ============== Thermal RC network model from impedance curves ============= 
import RC_imped
Zall, Ta = RC_imped.imped(time_data,y_data_train,pwr_train)


### ======== Inference by the model for validation and estimation =============
import time,RC_estimate,RC_lkg_model
# from finite_convolve import convolve

t_curr = time.time()

lkg_flag = 1
if lkg_flag == 0:
    ## estimate the response for step power input, use direct vector method
    dTia = RC_estimate.infer_step(time_data,Zall,pwr_infer)
    y_data_infer = dTia + Ta

else:

    ## for dynamic temperature-dependent power input (consider leakage), use recursive method
    fun_lkg_scale = RC_lkg_model.PWL # RC_lkg_model.curvefit is another option
    
    #= create a total power scaling factor function for each active source and form a list
    list_fun_pwr_total_scale = []
    for idx_infer in range(n_cases_infer):
        tmp = []
        for i in range(n_sources):
            fun_pwr_total_scale = lambda x,lkg_ratio = lkg_ratios[idx_infer][i]: \
                (1-lkg_ratio) + lkg_ratio*fun_lkg_scale(x)
            tmp.append(fun_pwr_total_scale)
        list_fun_pwr_total_scale.append(tmp)
        
    # the temperature index of each active source, its size should be n_sources 
    dTia = RC_estimate.infer_lkg(time_data,Zall,pwr_infer,list_fun_pwr_total_scale,idx_T_src)
    y_data_infer = dTia + Ta # All the estimated temperature series are in this data structure

print('It takes ',time.time() - t_curr, ' seconds.')

### ===================== Post-analysis =================================
## estimate the time to threshold (ToT)
idx_Tj = [5,3,88,89,6,\
          8,9,10,69,70,\
          71,72,73,74,75,\
          76,77,78,90,91,\
          94,93,97,98,99,\
          100,101,107,108,109,\
          68,110,92,113]
idx_Ts = [0,1,2] + list(range(11,68)) + [106,111,112]
idx_Tmid = [x for x in range(n_sensors) if x not in (idx_Tj+idx_Ts)]
Ts_labels = y_labels[idx_Ts]
Ts_tholds = [39.0,40.0,42.0,43.0,44.0,45.0] # unit Celsius
Tj_labels = y_labels[idx_Tj]
# Tj_thold = 90.0  # unit Celsius
y_data_infer_Ts = y_data_infer[:,:,idx_Ts]
y_data_infer_Tj = y_data_infer[:,:,idx_Tj]

ToT_Ts = []
ToT_Ts_labels = []
Tj_at_ToT_Ts = []
for idx_case in range(n_cases_infer):
    ToT_Ts_per_case = []
    ToT_Ts_per_case_labels = []
    Tj_at_ToT_Ts_per_case = []
    for Ts_thold in Ts_tholds:
        ## assume time_data is already resampled (1s interval at least)
        idx_time = np.abs(y_data_infer_Ts[idx_case] - Ts_thold).argmin(axis=0)
        ToT_Ts_every_sensor = time_data[idx_time]
        idx_sensor = ToT_Ts_every_sensor.argmin()
        ToT_Ts_tmp = ToT_Ts_every_sensor[idx_sensor]
        ToT_Ts_per_case.append(ToT_Ts_tmp/60)  # convert ToT unit from seconds to mins
        ToT_Ts_per_case_labels.append(Ts_labels[idx_sensor])
        
        idx_time_Tj = np.abs(time_data - ToT_Ts_tmp).argmin(axis=0)
        Tj_at_ToT_Ts_tmp = y_data_infer_Tj[idx_case][idx_time_Tj]
        Tj_at_ToT_Ts_per_case.append(Tj_at_ToT_Ts_tmp)
    ToT_Ts.append(ToT_Ts_per_case)
    ToT_Ts_labels.append(ToT_Ts_per_case_labels)
    Tj_at_ToT_Ts.append(Tj_at_ToT_Ts_per_case)

ToT_Ts = np.array(ToT_Ts).astype(str)
ToT_Ts_labels = np.array(ToT_Ts_labels)

print("Writing ToT result file ...")

my_comment = "# time to threshold (mins) of each power case for different thresholds"
header_list = ["scenarios"]
time_header = ["T_thold="+str(x)+"C" for x in Ts_tholds] # ignore time=0
time_header = time_header + [" "] + time_header
header_list.extend(time_header)
my_delim = ","
my_header = my_delim.join(np.array(header_list))
my_header = '\n'.join([my_comment,my_header])
output = []
for i in range(n_cases_infer):
    output.append([case_labels[i]] + ToT_Ts[i,:].tolist() + [" "] + ToT_Ts_labels[i,:].tolist())
np.savetxt(output_filename_ToT,output,fmt='%s',delimiter=my_delim,\
           header=my_header,comments='')

### ===================== Compare the estimation and original data ============ 
if n_cases_valid > 0:
    from RC_compare import compare_timeseries
    time_tholds = [300,600,1200] # compare at these time points (seconds)
    f = open(output_filename_cmp, "w")
    for idx_valid in range(n_cases_valid):    
        print("Writing compare result for Case " + filenames_y[idx_valid+n_sources] + ":\n")
        my_comment = "# Comparison error for Case " + filenames_y[idx_valid+n_sources]
        my_comment1 = "Row labels: Tsensor err_5mins err_10mins err_20mins err_end err_max RMSE"
        my_delim = ","
        my_header = my_delim.join(np.array(y_labels))
        my_header = '\n'.join([my_comment,my_comment1,my_header])
        err_all = \
                compare_timeseries(time_data,y_data_valid[idx_valid],y_data_infer[idx_valid],time_tholds)
        np.savetxt(f,err_all,'%.3e',delimiter=my_delim,\
                    header=my_header,comments='')
    f.close()
    
### ===================== Plot and save the comparison results ============ 
import thermal_data_plot

## plot the temperature or thermal impedance curves from step response CFD simulations

idx_sel_sensors_train_Ts = [0,15,106]
for idx_sel_sources in range(n_sources):
    idx_sel_sensors_train = idx_sel_sensors_train_Ts + [idx_Tj[idx_sel_sources]]
    plot_xdata = time_data
    plot_ydata = y_data_train[idx_sel_sources][:,idx_sel_sensors_train]
    case="Temperature traces of step power ("+ str(pwr_train[idx_sel_sources]) + " W) on " + u_labels[idx_sel_sources]
    plot_ylabels = [x for x in y_labels[idx_sel_sensors_train]]
    fig, axes = thermal_data_plot.tjts_traces(plot_xdata,\
                            plot_ydata,plot_ylabels,case=case)
    fig.savefig(path_to_files+'\\'+os.path.commonprefix(filenames_y)+str(idx_sel_sources)+\
                            '_tjts.png',dpi=600,bbox_inches='tight')
        
## loop through the validation dataset
idx_sel_sensors_valid = [0,1,2,15,106,111,112] # Tskin
idx_sel_sensors_infer = idx_sel_sensors_valid # sometimes validation and inferrence have different sensor sets 
save_file_suffix = '_ts'
for idx_valid in range(n_cases_valid):
    plot_xdata_orig = time_data_valid[idx_valid]
    plot_ydata_orig = y_data_valid[idx_valid][:,idx_sel_sensors_valid]
    plot_xdata_infer = time_data
    plot_ydata_infer = y_data_infer[idx_valid][:,idx_sel_sensors_infer]
    y_labels_sel = y_labels_valid[idx_valid][idx_sel_sensors_valid]
    plot_ylabels_orig = ['CFD_' + x for x in y_labels_sel]
    plot_ylabels_infer = ['RC_' + x for x in y_labels_sel]
    fig, axes = thermal_data_plot.tjts_compare_traces(plot_xdata_orig,plot_ydata_orig,\
                            plot_ylabels_orig,plot_xdata_infer,\
                            plot_ydata_infer,plot_ylabels_infer,case=path_to_files)
    fig.savefig(path_to_files+'\\'+os.path.splitext(filenames_y[idx_valid+n_sources])[0]+\
                            save_file_suffix + '.png',dpi=600,bbox_inches='tight')