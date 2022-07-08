# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:43:47 2021

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

#from PIL import Image

from numpy import sin,cos,pi

x=np.linspace(0.2,.8,10000)
#x1=np.linspace(0.4,0.6,1000)

y=x**3

y1=(0.5+0.1*cos(x*64*2*pi))**3
y2=(0.125+0.075*cos(x*64*2*pi))

y1[abs(x-0.5)>0.05]=np.nan
y2[abs(x-0.5)>0.05]=np.nan

plt.close('all')

plt.plot(x,y,label='Large Signal')
plt.plot(x,y1,label='Exact Perturbation Signal')
plt.plot(x,y2,label='Approx. Perturbation Signal')
plt.legend(fontsize=12)
plt.xticks([])
plt.yticks([])
plt.xlabel('Variation over x',fontsize=13)
plt.ylabel('Variation over y',fontsize=13)