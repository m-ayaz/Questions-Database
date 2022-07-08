# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:42:34 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

f=np.linspace(-30,50,10000)
def X(f):
    return (2-abs(abs(f)-2))*(abs(f)<4)

#x=z(t-1)
plt.figure()
plt.plot(f,X(f+1),linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,2.5],'k',linewidth=2)
plt.plot([1,1],[0,2],'k--')
plt.plot([-3,-3],[0,2],'k--')
plt.xlim([-6,4])
plt.xticks([-5,-3,-1,1,3],fontsize=20)
plt.yticks([0,1,2],fontsize=20)
plt.title(r'$X(j\omega)$',fontsize=20)
plt.savefig('_6Q.eps')

