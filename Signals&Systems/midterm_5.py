# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:49:50 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

f=np.linspace(-30,50,10000)
def x(t):
    return (t>1)*(t<2)*-1+(t>-2)*(t<-1)*1+0+(t>4)*(t<5)*1+(t>-5)*(t<-4)*-1

#x=z(t-1)
plt.figure()
plt.plot(f,x(f),linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-1.2,1.2],'k',linewidth=2)
plt.xlim([-6,6])
plt.xticks([-5,-4,-2,-1,1,2,4,5],fontsize=15)
plt.yticks([-1,0,1],fontsize=20)
plt.title(r'$x(t)$',fontsize=20)
plt.savefig('_17Q.eps')

