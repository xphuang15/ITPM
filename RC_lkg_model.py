# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:43:24 2021
Leakage power model

@author: XiaopengHuang
"""

import sys
from scipy import interpolate
from scipy.optimize import curve_fit
import math

def PWL(dT): # piece-wise linear approximation
    ## define the temperature-leakage scaling pairs in the format of (dT:leakage power scaling factor) 
    lkg_points = {0: 1, 20: 2.5, 40: 5, 60: 10, 80: 20, 100: 40}
    Ta = 25 # C, ambient temperature
    xdata = [float(x) for x in lkg_points.keys()]
    ydata = [float(x) for x in lkg_points.values()]
    kind = 'linear' # linear interpolation between two neighbour points
    fun_lkg_scale = interpolate.interp1d(xdata,ydata,kind=kind,axis=0)
    
    ## deal with the out-of-range input temperature
    dT_min = min(xdata)
    dT_max = max(xdata)
    dT_min_end = min(-40-Ta,dT_min) # low end temperature is assumed to be -40C with Tamb = 25C
    dT_max_end = max(250-Ta,dT_max) # high end temperature is assumed to be 250C with Tamb = 25C
    
    if dT < dT_min_end or dT > dT_max_end:
        lkg_scale = 9999
        sys.exit("Input delta temperature for leakage scaling is out of range ["+dT_min_end+","+dT_max_end+"]C")
    elif dT < dT_min:
        lkg_scale = lkg_points[dT_min]
    elif dT > dT_max:
        lkg_scale = lkg_points[dT_max]
    else:
        lkg_scale = fun_lkg_scale(dT)
    
    return lkg_scale

def curvefit(dT): # fitting with exponential curve
    ## define the temperature-leakage scaling pairs in the format of (dT:leakage power scaling factor) 
    lkg_points = {0: 1, 20: 2.5, 40: 5, 60: 10, 80: 20, 100: 40}
    func = lambda x, p: math.exp(p*x)
    xdata = [float(x) for x in lkg_points.keys()]
    ydata = [float(x) for x in lkg_points.values()]
    popt, pcov = curve_fit(func, xdata, ydata, p0=0.04)
    lkg_scale = func(dT,popt)
    return lkg_scale