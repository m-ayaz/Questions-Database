# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:26:30 2021

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

t=np.linspace(-4.5,4.5,1000)

t1=t.copy()

t=np.mod(t,2)

x=0*(t<1)+(2-t)*(t>=1)

plt.plot(t1,x)

plt.plot([0,0],[-0.2,1.2],'k')
#plt.plot([1,1],[0,1],'k--')
#plt.plot([-1,-1],[0,1],'k--')
#plt.plot([3,3],[0,1],'k--')
#plt.plot([-3,-3],[0,1],'k--')
plt.plot([-4,4],[0,0],'k')
plt.plot([-4,4],[1,1],'k--')

#plt.axis('off')

plt.yticks([0,1],fontsize=20)
plt.xticks([-4,-3,-2,-1,0,1,2,3,4],fontsize=16)
plt.tight_layout(pad=.3)
#plt.title('$x(t)$',fontsize=15)