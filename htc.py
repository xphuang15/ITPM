# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:01:22 2020

@author: XiaopengHuang
"""
import math
from scipy import interpolate

### heat transfer coefficient by natural convection and radiation for a flat plate
def hsa_lin(heatflux): # heat transfer coefficient between surface and ambient
        
    ## When Ta = 25C, h = 0.125*Ts + 6.06 = 0.125(Ts-Ta) + 9.185 = a*dTs + b
    ## q = h(Ts-Ta) = a(dTs + b/2a)^2 - b^2/4a
    ## dTs = -b/2a + sqrt((q+b^2/4a)/a)
    ## h = 0.5*b + sqrt(aq+b^2/4)
    a = 0.03 # vertical plate
    b = 9.185 # vertical plate
    hsa = 0.5*b + math.sqrt(a*heatflux+b*b/4)
    
    return hsa

## dimensional equations
def hsa_pow_dim(Ts, emiss=0.9, Ta=25): # heat transfer coefficient between surface and ambient

    C = 1.0*(1.32+0.59)/2 # constant, depends upon the orientation of surface
    L = 0.03*0.04/(2*(0.03+0.04)) # unit m, the characteristic length A/P
    h_conv = C*math.pow(abs(Ts-Ta)/L,0.22)
    ## assuming view factor is 1
    h_rad = 5.67e-8*emiss*((Ts+273.15)**2 + (Ta+273.15)**2)*((Ts+273.15) + (Ta+273.15))
    hsa = h_conv + h_rad
   
    return hsa

## dimensionless equations  !!! TO-DO
def hsa_pow_dimless(Ts, emiss=0.9, Ta=25): # heat transfer coefficient between surface and ambient

    C = 1.42 # constant, depends upon the orientation of surface
    L = 0.03*0.04/(2*(0.03+0.04)) # unit m, the characteristic length A/P
    h_conv = C*math.pow(abs(Ts-Ta)/L,0.25)
    ## assuming view factor is 1
    h_rad = 5.67e-8*emiss*((Ts+273.15)**2 + (Ta+273.15)**2)*((Ts+273.15) + (Ta+273.15))
    hsa = h_conv + h_rad
   
    return hsa