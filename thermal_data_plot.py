# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:50:52 2020

@author: XiaopengHuang
"""

import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def tjts_traces(plot_xdata,plot_ydata,plot_ylabels,case="Temperature Traces"):
    
    ChineseFont = FontProperties('Microsoft YaHei')
    plt.close("all")
    nsubplots = 1
    fig, axes = plt.subplots(1,nsubplots,sharex=False,sharey=False,figsize=(4*nsubplots,6))
    ax = axes
    ax.plot(plot_xdata, plot_ydata, linewidth=1)
    # ax.set_ylim((25,50))
    ax.set_ylabel("Temperature (Celsius)")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major',lw=1.5)
    ax.grid(which='minor',lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_xscale('log')
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title(case,fontproperties = ChineseFont)
    ax.legend(plot_ylabels,bbox_to_anchor=(1.7,0.5),loc='right')
    
    axes = ax
    
    return fig, axes
    
def tjts_compare_traces(plot_xdata_orig,plot_ydata_orig,plot_ylabels_orig,\
                        plot_xdata_infer,plot_ydata_infer,plot_ylabels_infer,case="Temperature Traces"):
    
    ChineseFont = FontProperties('Microsoft YaHei')
    plt.close("all")
    nsubplots = 1
    fig, axes = plt.subplots(1,nsubplots,sharex=False,sharey=False,figsize=(6*nsubplots,8))
    ax = axes
    # color = ax._get_lines.prop_cycler
    ax.plot(plot_xdata_orig,plot_ydata_orig,linewidth=2)
    plt.gca().set_prop_cycle(None)
    ax.plot(plot_xdata_infer,plot_ydata_infer,linestyle="dashed",linewidth=1)
    # ax.plot(plot_xdata_infer,plot_ydata_infer,marker='o',markersize=2,markerfacecolor='none',linestyle="none")
    # ax.set_ylim((20,90))
    ax.set_ylabel("Temperature (Celsius)")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major',lw=1.5)
    ax.grid(which='minor',lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_xscale('log')
    ax.set_xlim((plot_xdata_infer[1],plot_xdata_infer[-1]))
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title(case,fontproperties = ChineseFont)
    ax.legend(np.hstack([plot_ylabels_orig,plot_ylabels_infer]),bbox_to_anchor=(1.5,0.5),loc='right')
    
    axes = ax
    
    return fig, axes    

