# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:09:05 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

x=[2,1,-1]*4
h=[1,1,1,-1]*3

plt.stem(x)
plt.xticks(np.linspace(0,11,12),fontsize=15)
plt.yticks([-1,0,1,2],fontsize=15)
plt.title(r'$x[n]$',fontsize=15)
plt.savefig('PSol9_Q1_1.eps')
plt.figure()
plt.stem(h)
plt.yticks([-1,0,1],fontsize=15)
plt.xticks(np.linspace(0,11,12),fontsize=15)
plt.title(r'$h[-n]=h[12-n]$',fontsize=15)
plt.savefig('PSol9_Q1_2.eps')
plt.figure()
x=[1,2,3,2,1]*9
y=[1,-1,1]
plt.stem(np.convolve(x,y))
plt.xlim([10.5,25.5])
plt.xticks(np.linspace(11,25,15),['-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7'],fontsize=15)
plt.yticks([0,1,2],fontsize=15)
plt.ylim([0,2.4])
plt.savefig('PSol9_Q2.eps')

plt.figure()
plt.stem([0,1,0,-1])
plt.xticks([0,1,2,3],fontsize=15)
plt.yticks([-1,0,1],fontsize=15)
plt.ylim([-1.4,1.4])
plt.savefig('PSol9_Q3_1.eps')