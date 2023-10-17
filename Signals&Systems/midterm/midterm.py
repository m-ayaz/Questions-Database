# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:15:04 2022

@author: smosaya
"""

import numpy as np

import matplotlib.pyplot as plt


t=np.linspace(-1,5,1000)

x=(1-t)*(0<=t)*(t<=1)+(3-t)*(t>=3)*(t<=4)

plt.close("all")

plt.plot(t,x,linewidth=4)
plt.title(r"$x(t)$",fontsize=30)

plt.plot([-1,5],[0,0],'k',linewidth=2)
plt.plot([0,0],[-1.3,1.3],'k',linewidth=2)

plt.plot([-1,0],[1,1],'k--',linewidth=2)
plt.plot([-1,4],[-1,-1],'k--',linewidth=2)


plt.tight_layout(pad=0.1)

plt.xticks([-1,0,1,2,3,4,5],fontsize=25)
plt.yticks([-1,0,1],fontsize=25)