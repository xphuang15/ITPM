# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:48:22 2020

@author: XiaopengHuang
"""

from __future__ import division
import control as ctrl
import numpy as np
import sympy as sym

###====== Build Cauer RC ladder as a transfer function based on RC pairs ======
def list2tf_Ca(R,C):
    
    if isinstance(R,np.ndarray) & isinstance(R,list):
        raise("Error: 1st argument R has to be a numpy array or a list")
    if isinstance(C,np.ndarray) & isinstance(C,list):
        raise("Error: 2nd argument C has to be a numpy array or a list")
        
    ## convert R & C into array if they are 
    ## assume 1D array or list
    R = np.array(R)
    C = np.array(C)
    
    if (len(R.shape) > 1) | (len(C.shape) > 1):
        raise("Error: R and C should be one dimension array or list")
        
    ## build fraction format symbolic transfer function 
    def list_to_frac(l):
        expr = sym.Integer(0)
        for i in reversed(l[1:]):
            expr += i
            expr = 1/expr
        return l[0] + expr
    
    s = sym.symbols('s')
    try:  # for n_stage >=2
        R[1]
        n_stage = len(R)
        RC_list = []
        for idx_stage in range(n_stage):
            RC_list.extend([s*C[idx_stage],R[idx_stage]])        
    except: # for n_stage = 1
        n_stage = 1  # 1 stage case
        RC_list = [s*float(C),float(R)]
        
    Z_Ca = 1/list_to_frac(RC_list) # symbolic Cauer RC tf in fraction form
    Z_Ca_poly = sym.cancel(Z_Ca, extension=True) # symbolic Cauer RC tf in num/den polynomial form
    [num_poly,den_poly] = sym.fraction(Z_Ca_poly)
    num = [float(i) for i in sym.poly(num_poly,s).all_coeffs()] # get coeff of numerator poly
    den = [float(i) for i in sym.poly(den_poly,s).all_coeffs()] # get coeff of denominator poly

    return num,den


###====== Build Foster RC ladder as a transfer function based on RC pairs ======
def list2tf_Fo(R,C):

    if isinstance(R,np.ndarray) & isinstance(R,list):
        raise("Error: 1st argument R has to be a numpy array or a list")
    if isinstance(C,np.ndarray) & isinstance(C,list):
        raise("Error: 2nd argument C has to be a numpy array or a list")
        
    ## convert R & C into array if they are 
    ## assume 1D array or list
    R = np.array(R)
    C = np.array(C)
    
    if (len(R.shape) > 1) | (len(C.shape) > 1):
        raise("Error: R and C should be one dimension array or list")
          
    s = sym.symbols('s')
    try:
        R[1]
        n_stage = len(R) # only works when n_stage >=2
    except:
        n_stage = 1  # 1 stage case
        
    ## symbolic Foster RC tf in fraction form
    idx_stage = 0
    Z_Fo = R[idx_stage]/(1+s*C[idx_stage]*R[idx_stage])
    for idx_stage in range(1,n_stage):
        Z_Fo += R[idx_stage]/(1+s*C[idx_stage]*R[idx_stage])

    Z_Fo_poly = sym.cancel(Z_Fo, extension=True) # symbolic Cauer RC tf in num/den polynomial form
    [num_poly,den_poly] = sym.fraction(Z_Fo_poly)
    num = [float(i) for i in sym.Poly(num_poly,s).all_coeffs()] # get coeff of numerator poly
    den = [float(i) for i in sym.Poly(den_poly,s).all_coeffs()] # get coeff of denominator poly

    return num,den

###====== Convert Cauer RC ladder to Foster RC ladder ======
def Ca2Fo_RC(R,C):
    
    num,den = list2tf_Ca(R,C)
    num.reverse() # the coeff order is reversed of the element index
    den.reverse()
    
    s = sym.symbols('s')
    # num_poly = sym.Poly(num, s)
    # den_poly = sym.Poly(den, s)

    num_poly = num[0] 
    for idx_num in range(1,len(num)):
        num_poly += num[idx_num]*pow(s,idx_num)    
    den_poly = den[0]
    for idx_den in range(1,len(den)):
        den_poly += den[idx_den]*pow(s,idx_den)
    
    Z_Ca_poly = sym.together(num_poly/den_poly)
    Z_Fo_fract = sym.apart(Z_Ca_poly,full=True).doit()
    Z_Fo_fract_list = list(Z_Fo_fract._sorted_args)
    r = []
    c = []
    if len(Z_Fo_fract_list) == len(R): # partial fraction decomposition works using sym.apart 
        for idx_stage in range(len(R)):
            stage = Z_Fo_fract_list[idx_stage]
            [num_stage,den_stage] = sym.fraction(stage)
            ## assuming den_stage is c/(a*s+b), then C = a/c and R = c/b for R/(s*RC+1)
            num_c = [float(i) for i in sym.Poly(num_stage,s).all_coeffs()] # get coeff of numerator poly
            den_ab = [float(i) for i in sym.Poly(den_stage,s).all_coeffs()] # get coeff of denominator poly
            if (len(num_c) == 1) & (len(den_ab) == 2):
                r_tmp = num_c[0]/den_ab[1]
                c_tmp = den_ab[0]/num_c[0]
                if (r_tmp > 0.0) & (c_tmp > 0.0):
                    r.append(r_tmp)
                    c.append(c_tmp)
                else:
                    ("Error: negative value obtained by partial fraction decomposition sym.apart")
            else:
                raise("Error: check code")
    else:     # partial fraction decomposition by manual calculation
        # polyRoots = sym.solveset(den_poly,s).args
        # tau_list = [-1/x for x in polyRoots]
        raise("Error: partial fraction decomposition is not right")

    return np.array(r,dtype=np.float64),np.array(c,dtype=np.float64)


###====== Convert Foster RC ladder to Cauer RC ladder ======
def Fo2Ca_RC(r,c):
    
    num,den = list2tf_Fo(r,c)
    num.reverse() # the coeff order is reversed of the element index
    den.reverse()
    
    s = sym.symbols('s')
    # num_poly = sym.Poly(num, s)
    # den_poly = sym.Poly(den, s)
 
    num_poly = num[0] 
    for idx_num in range(1,len(num)):
        num_poly += num[idx_num]*pow(s,idx_num)    
    den_poly = den[0]
    for idx_den in range(1,len(den)):
        den_poly += den[idx_den]*pow(s,idx_den)

    R = []
    C = []
    for idx_stage in range(len(r),0,-1):
        quo, rem = sym.div(den_poly,num_poly)
        
        C.append(sym.Poly(quo,s).all_coeffs()[0])
        R_inv = sym.Poly(quo,s).all_coeffs()[1]
        R.append(1/R_inv)
        
        # go the the n-1 stage
        den_poly = R_inv*num_poly + rem
        num_poly = -rem/R_inv

    return np.array(R,dtype=np.float64),np.array(C,dtype=np.float64)