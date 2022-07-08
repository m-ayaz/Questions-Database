# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:38:59 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

t=np.linspace(-30,50,10000)
def x(t):
    return (t+1)*(t>-1)*(t<0)+1*(t>0)*(t<2)+0*(t>0)*(t<2)

#x=z(t-1)
plt.figure()
plt.plot(t,x(t),linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,1.5],'k',linewidth=2)
plt.xlim([-3,4])
plt.xticks(fontsize=15)
plt.yticks([0,1],fontsize=15)
plt.savefig('_3Q.eps')