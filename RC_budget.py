# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:09:48 2020

@author: XiaopengHuang
"""

import numpy as np
import RC_estimate
from scipy import interpolate

### Estimate the sensor temperatures at a range of power numbers for each source

def budget(time_data,Zsa,Zja,pwr_cases,Ta,ToT,Ts_thold=40,Tj_thold=100):
   
    ## for step power input and junction temperature 
    dTja = RC_estimate.infer_step(time_data,Zja,pwr_cases)
    y_data_infer_Tj = dTja + Ta

    ## for step power input and skin temperature 
    dTsa = RC_estimate.infer_step(time_data,Zsa,pwr_cases)
    y_data_infer_Ts = dTsa + Ta
    
    n_pwr_cases = pwr_cases.shape[0]
    pwr_cases_meet = []
    idxs_Ts_ToT_max = []
    idxs_Tj_ToT_max = []
    y_data_infer_Ts_ToT_max_list = []
    y_data_infer_Tj_ToT_max_list = []
    for idx_pwr_case in range(n_pwr_cases):

        # interplate time and y_data at ToT
        fun_y_Ts = interpolate.interp1d(time_data,y_data_infer_Ts[idx_pwr_case],kind='linear',axis=0)       
        y_data_infer_Ts_ToT = fun_y_Ts(ToT)
        y_data_infer_Ts_ToT_max = max(y_data_infer_Ts_ToT)
        idx_Ts_ToT_max = max(range(len(y_data_infer_Ts_ToT)), key=y_data_infer_Ts_ToT.__getitem__)
        
        fun_y_Tj = interpolate.interp1d(time_data,y_data_infer_Tj[idx_pwr_case],kind='linear',axis=0)       
        y_data_infer_Tj_ToT = fun_y_Tj(ToT)
        y_data_infer_Tj_ToT_max = max(y_data_infer_Tj_ToT)
        idx_Tj_ToT_max = max(range(len(y_data_infer_Tj_ToT)), key=y_data_infer_Tj_ToT.__getitem__)
        
        if (y_data_infer_Ts_ToT_max <= Ts_thold) & (y_data_infer_Tj_ToT_max <= Tj_thold):
            pwr_cases_meet.append(pwr_cases[idx_pwr_case])
            idxs_Ts_ToT_max.append(idx_Ts_ToT_max)
            idxs_Tj_ToT_max.append(idx_Tj_ToT_max)
            y_data_infer_Ts_ToT_max_list.append(y_data_infer_Ts_ToT_max)
            y_data_infer_Tj_ToT_max_list.append(y_data_infer_Tj_ToT_max)
    
    if pwr_cases_meet:
        ## remove the redudant power and form the power budget cases
        pwr_budget = []
        pwr_budget.append(pwr_cases_meet[0])
        for idx_pwr_case_meet in range(1,len(pwr_cases_meet)):
            pwr_case_new = pwr_cases_meet[idx_pwr_case_meet]
            if len(pwr_budget) > 0:
                for idx_pwr_budget in range(len(pwr_budget)):
                    if any(pwr_case_new > pwr_budget[idx_pwr_budget]):
                        if all(pwr_case_new >= pwr_budget[idx_pwr_budget]):
                            pwr_budget[idx_pwr_budget] = pwr_case_new
                            break
                        else:
                            if idx_pwr_budget == (len(pwr_budget) - 1):
                                pwr_budget.append(pwr_case_new)
                    else:
                        print("Satifying power case " + str(pwr_case_new) + " is redudant and removed")
                        break
                    
        idxs = [np.array(pwr_cases_meet).tolist().index(list(x)) for x in pwr_budget]
        idxs_Ts_ToT_max = [idxs_Ts_ToT_max[x] for x in idxs]
        idxs_Tj_ToT_max = [idxs_Tj_ToT_max[x] for x in idxs]
        Ts_ToT_max = [y_data_infer_Ts_ToT_max_list[x] for x in idxs]
        Tj_ToT_max = [y_data_infer_Tj_ToT_max_list[x] for x in idxs]
        
    else:
        print("WARNING: no power cases meet the ToT criteria. Lower pwr_opt_ptg or raise T_thold.")
        
    return pwr_budget,idxs_Ts_ToT_max,idxs_Tj_ToT_max,Ts_ToT_max,Tj_ToT_max