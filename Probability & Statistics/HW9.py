# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:37:33 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

plt.plot([-3,3],[0,0],'k',linewidth=2)
plt.plot([0,0],[-.2,1.2],'k',linewidth=2)

x=np.linspace(-3,3,10000)
y=0.5*(x<0)*(x>=-1)+(x>=1)*1+(x>=0)*(x<1)*(0.5+2*x**3-1.5*x**4)
plt.plot(x,y,linewidth=3)
plt.xticks(fontsize=20,fontname='Times New Roman')
plt.yticks([0,.25,.5,.75,1],fontsize=20,fontname='Times New Roman')
plt.grid('on')
plt.tight_layout(pad=.3)