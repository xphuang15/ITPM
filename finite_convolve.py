# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:49:50 2020

@author: XiaopengHuang
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math

def convolve(y1, y2, dx = None):
    '''
    Compute the finite convolution of two signals of equal length.
    @param y1: First signal.
    @param y2: Second signal.
    @param dx: [optional] Integration step width.
    @note: Based on the algorithm at http://www.physics.rutgers.edu/~masud/computing/WPark_recipes_in_python.html.
    '''
    P = len(y1) #Determine the length of the signal
    z = [] #Create a list of convolution values
    for k in range(P):
        t = 0
        lower = max(0, k - (P - 1))
        upper = min(P - 1, k)
        for i in range(lower, upper):
            t += (y1[i] * y2[k - i] + y1[i + 1] * y2[k - (i + 1)]) / 2
        z.append(t)
    z = np.array(z) #Convert to a numpy array
    if dx != None: #Is a step width specified?
        z *= dx
    return z
