# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:23:16 2020

@author: XiaopengHuang
"""
import os, sys
from natsort import natsorted

def setup():
    
    path_to_files = os.getcwd()
    _, dirnamesA, filenames = next(os.walk(path_to_files))
    mark_str_y = ["RC_network","csv"] # mark strings in the temperature file name
    mark_str_u = ["UserScenarios_Power","csv"] # mark strings in the power file name
    mark_str_exclude = ["power_budget.csv","longer.csv","ToT_results"] # mark strings to exclude the output power budget csv
    
    filenames = [x for x in filenames if not any(word in x for word in mark_str_exclude)]
    
    ### filter the temperature files
    filenames_y = [x for x in filenames if all(word in x for word in mark_str_y)]
    filenames_y = natsorted(filenames_y) # make names natural sorted based on numbers in file names

    ### filter the power files
    filenames_u = [x for x in filenames if all(word in x for word in mark_str_u)]
    filenames_u = natsorted(filenames_u) # make names natural sorted based on numbers in file names

    return [path_to_files,filenames_y,filenames_u]

