# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 02:47:04 2021

@author: Mostafa
"""


import numpy as np
import matplotlib.pyplot as plt


x=[1,2,4,8,9,10,11,12,9,10,11,6,7,8,1,2,4]

plt.plot(np.arange(1,1+len(x)),x,'-o')
plt.grid('on')

plt.xticks(np.arange(1,1+len(x)),fontsize=14)
plt.yticks([0,2,4,6,8,10,12],fontsize=14)

plt.xlabel('Transmission Round',fontsize=14)
plt.ylabel('Congestion Window (in segments)',fontsize=14)

plt.tight_layout(pad=0.3)
plt.savefig('Q4.eps')