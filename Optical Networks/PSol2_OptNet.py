# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:38:26 2020

@author: Glory
"""

import numpy as np
import matplotlib.pyplot as plt

N=21

x=np.linspace(-(N-1)/2,(N-1)/2,N)

for i in range(N):
    for j in range(N):
        if x[i]>0 and x[j]>0:
            col='#ad1111'
        elif x[i]<0 and x[j]>0:
            col='#0a520a'
        elif x[i]>0 and x[j]<0:
            col='#1349a1'
        elif x[i]<0 and x[j]<0:
            col='#d4c311'
        else:
            col='#000000'
        plt.plot(x[i],x[j],'.',color=col)
        
plt.axis('equal')
plt.axis('off')
plt.savefig('p3.eps')