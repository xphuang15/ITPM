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
output_filename = os.path.commonprefix(filenames_y) + "compare_results.txt"
pwr_budget_filename = os.path.commonprefix(filenames_y) + "power_budget.csv"

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


# ### ===================== System identification from step response ============ 
# import sid_step

# ### ===================== System identification from dynamic response ========= 
# from sid_dynamic import sid_dynamic
# method = 'MOESP' # 'N4SID', 'MOESP', 'CVA', 'PARSIM-P', 'PARSIM-S' AND 'PARSIM-K'
# order = 10
# [sys_id,xid,yid] = sid_dynamic(y_data, u_data, method, order)

# ### ===================== Simulated iDCTM training from dynamic response ====== 

# ### ===================== Experimental iDCTM training from dynamic response === 
# from DCTM import iDCTM


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
Ts_tholds = [39.0,40.0,42.0,43.0] # unit Celsius
# Tj_thold = 90.0  # unit Celsius
Zsa = [Zall[x] for x in idx_Ts]
Zja = [Zall[x] for x in idx_Tj]
## for step power input and skin temperature 
dTsa = RC_estimate.infer_step(time_data,Zsa,pwr_valid)
y_data_infer_Ts = dTsa + Ta
Ts_ToT = []
Ts_ToT_labels = []
for idx_pwr_case in range(n_pwr_cases):
    Ts_ToT_per_case = []
    Ts_ToT_per_case_labels = []
    for Ts_thold in Ts_tholds:
        ## assume time_data is already resampled (1s interval at least)
        idx_time = np.abs(y_data_infer_Ts[idx_pwr_case] - Ts_thold).argmin(axis=0)
        Ts_ToT_every_sensor = time_data[idx_time]
        idx_sensor = Ts_ToT_every_sensor.argmin()
        Ts_ToT_per_case.append(Ts_ToT_every_sensor[idx_sensor])
        Ts_ToT_per_case_labels.append(Ts_labels[idx_sensor])
    Ts_ToT.append(Ts_ToT_per_case)
    Ts_ToT_labels.append(Ts_ToT_per_case_labels)

# ### ===================== Power budgeting with the extract RC network =========
# import RC_budget

# # pwr_base = [1,0.049,0.33,0.219,0.094,0.5,0.221,1,0.096,0.142,0.1839,3.5,0.02,1,0.04,0.006]
# # pwr_opt_ptg = [0.5,1,1,1,1,0.2,1,0.5,1,1,1,0.6,1,0.5,1,1] # ultimate power optimization target for each source

# pwr_base = [0.8,0.049,0.33,0.219,0.094,0.5,0.221,0.8,0.096,0.142,0.1839,3,0.02,0.8,0.04,0.006]
# pwr_opt_ptg = [0.5,1,1,1,1,0.2,1,0.5,1,1,1,0.6,1,0.5,1,1] # ultimate power optimization target for each source
# pwr_steps = [12,1,1,1,1,5,1,12,1,1,1,10,1,12,1,1]
# # pwr_steps = [6,1,1,1,1,5,1,6,1,1,1,5,1,6,1,1]
# # pwr_steps = [2,1,1,1,1,2,1,2,1,1,1,2,1,2,1,1]
# tmp = []
# for idx_source in range(n_sources):
#     arr = pwr_base[idx_source]*np.linspace(1,pwr_opt_ptg[idx_source],pwr_steps[idx_source])
#     tmp.append(np.round(arr,3))

# pwr_cases_grid = np.meshgrid(*tmp)
# pwr_cases = np.array([np.ndarray.flatten(x) for x in pwr_cases_grid]).T

# idx_Tj = [1,4,9,10,11,14,19,22,26,27,28,29,30,31,32,37,40,45]
# idx_Tmid = [25,42]
# idx_Ts = [x for x in range(n_sensors) if x not in (idx_Tj+idx_Tmid)]
# ToT = 600 # unit s
# Ts_thold = 44 # unit Celsius
# Tj_thold = 90  # unit Celsius
# Zsa = [Zall[x] for x in idx_Ts]
# Zja = [Zall[x] for x in idx_Tj]
# pwr_budget,idxs_Ts_ToT_max,idxs_Tj_ToT_max,Ts_ToT_max,Tj_ToT_max = \
#         RC_budget.budget(time_data,Zsa,Zja,pwr_cases,Ta,ToT,Ts_thold,Tj_thold)
# pwr_budget_total = [sum(x) for x in pwr_budget]
# labels_Ts_ToT_max = [str(y_labels[idx_Ts[x]]) for x in idxs_Ts_ToT_max]
# labels_Tj_ToT_max = [str(y_labels[idx_Tj[x]]) for x in idxs_Tj_ToT_max]

# f = open(pwr_budget_filename, "w", newline='')
# my_comments = '# Power budget from fast thermal RC network estimation:'
# my_delim = ","
# my_header = u_labels + ['Total_power(W)','Label_Ts_ToT_max','Ts_ToT_max',\
#                              'Label_Tj_ToT_max','Tj_ToT_max']
# writer = csv.writer(f,delimiter=my_delim)
# writer.writerow([my_comments])
# writer.writerow(my_header)
# for idx in range(len(pwr_budget)):
#     row_list = list(np.append(pwr_budget[idx],pwr_budget_total[idx]))
#     row_list.extend([labels_Ts_ToT_max[idx],Ts_ToT_max[idx]])
#     row_list.extend([labels_Tj_ToT_max[idx],Tj_ToT_max[idx]])
#     writer.writerow(row_list)
# f.close()


### ===================== Compare the estimation and original data ============ 
from RC_compare import compare

## match the sensors in step response and user scenario for comparison

idx_sel_sensors_valid = [0,1,2,3,4,5] # Tskin
idx_sel_sensors_infer = [46,23,47,48,49,50]
save_file_suffix = '_ts'

# idx_sel_sensors_valid = [6,7,8,9,10] # T junction
# idx_sel_sensors_infer = [22,19,28,29,30]
# save_file_suffix = '_tj'
# idx_sel_sensors_valid = [8,9,10] # T junction
# idx_sel_sensors_infer = [28,29,30]
# save_file_suffix = '_tj_CC'


err_filename = os.path.splitext(output_filename)[0] + save_file_suffix + \
                os.path.splitext(output_filename)[1]
f = open(err_filename, "w")
## loop through the validation dataset
for idx_valid in range(n_valid):
    f.write("Comparison error for Case " + filenames_y[idx_valid+n_sources] + ":\n")
    ## assume both data have the same time step
    idx_time_end = min(y_data_valid[idx_valid].shape[0],y_data_infer[idx_valid].shape[0])
    cmp_ydata_orig = y_data_valid[idx_valid][:idx_time_end,idx_sel_sensors_valid]
    cmp_ydata_infer = y_data_infer[idx_valid][:idx_time_end,idx_sel_sensors_infer]
    if cmp_ydata_orig.shape[0] == cmp_ydata_infer.shape[0]:
        [data_RMSE,err_max] = compare(cmp_ydata_orig,cmp_ydata_infer)
        y_labels_sel = y_labels_valid[idx_valid][idx_sel_sensors_valid]
        np.savetxt(f,y_labels_sel.reshape((1,-1)),fmt='%s')
        np.savetxt(f,data_RMSE.reshape((1,-1)),fmt='%.3e')
        np.savetxt(f,err_max.reshape((1,-1)),fmt='%.3e')
f.close()


### ===================== Plot and save the comparison results ============ 
import thermal_data_plot

## plot the temperature or thermal impedance curves from step response CFD simulations

idx_sel_sensors_train = [22,19,37,36,35,50,0]
# idx_sel_sources = 11 ## 11 is SoC
# for idx_sel_sources in range(n_sources):
for idx_sel_sources in [11]:
    plot_xdata = time_data
    plot_ydata = y_data_train[idx_sel_sources][:,idx_sel_sensors_train]
    case="Temperature traces of step power (1W) on " + u_labels[idx_sel_sources]
    plot_ylabels = [x for x in y_labels[idx_sel_sensors_train]]
    fig, axes = thermal_data_plot.tjts_traces(plot_xdata,\
                            plot_ydata,plot_ylabels,case=case)
    fig.savefig(path_to_files+'\\'+os.path.commonprefix(filenames_y)+str(idx_sel_sources)+\
                            '_tjts.png',dpi=600,bbox_inches='tight')
        
## loop through the validation dataset

for idx_valid in range(n_valid):
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
    