# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:08:38 2020

@author: XiaopengHuang
"""

import csv
import numpy as np
import RC_cleanup

def y_traces_import(path_to_files,filenames,filetype,cleanup_flag):

    ### define importer for different filetypes
    def import_csv(filenames):
        DATA_series = []

        for filename in filenames:
            file_path = path_to_files + "/" + filename
            fp = open(file_path,'r')
            tmp_reader = csv.reader(fp)
            print("Reading " + filename + "...")
            
            ## read the header
            col_labels = next(tmp_reader)
            del col_labels[0]
            y_labels = np.asarray(col_labels)
            
            ## check the format (to-do???)
            # if col_labels == None:
            #     raise ValueError("unknown file format")
        
            ## read the data
            raw_data = list(tmp_reader)
            time_data = np.asarray([x[0] for x in raw_data], dtype=np.float64)
            raw_data = np.asarray(raw_data, dtype=np.float64)
            raw_data = raw_data[:,1:]
            
            fp.close()
            del tmp_reader
            
            ## clean up the data if clean up flag is true
            if cleanup_flag:
                [time_data,y_data] = RC_cleanup.resample(time_data,raw_data,kind='linear')
            else:
                y_data = raw_data
                
            ## organize the data    
            
            DATA = [time_data, y_data, y_labels]
            DATA_series.append(DATA)
            
            print("Done.")
        
        return DATA_series    
            
    def import_txt(filenames):
        return filenames            
    
    ### choose a importer according to the filetype
    switcher = {
                'csv':import_csv(filenames),
                'txt':import_txt(filenames)}
    yDATA_series = switcher.get(filetype,"Invalid filetype")
    return yDATA_series


def u_traces_import(path_to_files,filenames,filetype,dynamic_flag=0):
    ### now only good for step power input (dynamic_flag=0)
    ### define importer for different filetypes
    def import_csv(filenames):
        DATA_series = []

        for filename in filenames:
            file_path = path_to_files + "/" + filename
            fp = open(file_path,'r')
            tmp_reader = csv.reader(fp)
            print("Reading " + filename + "...")
            
            ## read the comment
            comment = next(tmp_reader)
            ## read the header
            col_labels = next(tmp_reader)
            del col_labels[0:4] # assume 0-3 column label is scenarios,sys_tot,AP_tot,pilot_tot
            u_labels = np.asarray(col_labels)
            
            ## check the format (to-do???)
            # if col_labels == None:
            #     raise ValueError("unknown file format")
        
            ## read the data (for each scenario, it has two rows, power and lkg ratio)
            raw_data = np.asarray(list(tmp_reader), dtype=str)
            case_labels = raw_data[2::2,0] 
            u_data = np.asarray(raw_data[:,4:], dtype=np.float64)
            
            fp.close()
            del tmp_reader
                           
            ## organize the data    
            
            DATA = [case_labels, u_data, u_labels]
            DATA_series.append(DATA)
            
            print("Done.")
        
        return DATA_series
           
            
    def import_txt(filenames):
        return filenames            
    
    
    ### choose a importer according to the filetype
    switcher = {
                'csv':import_csv(filenames),
                'txt':import_txt(filenames)}
    uDATA_series = switcher.get(filetype,"Invalid filetype")
    return uDATA_series