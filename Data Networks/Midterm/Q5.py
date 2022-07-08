# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:57:11 2020

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

plt.close('all')

y=[1,2,4,8,16,1,2,4,8,9,10,11,12,9,10,11]
x=np.arange(len(y))+1

plt.plot(x,y,marker='o',markersize=10)
plt.grid('on')
plt.tight_layout(pad=3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,17.5])
plt.xlabel('Transmission Round',fontsize=13)
plt.ylabel('CWND',fontsize=13)
plt.savefig('Q5.eps')