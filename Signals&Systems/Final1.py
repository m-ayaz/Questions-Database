# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:51:39 2021

@author: Mostafa
"""

import matplotlib.pyplot as plt

plt.close('all')

x=[1,-1,1,2,3,2,1,-1,1]
t=list(range(-4+2,4+1+2))

plt.stem(t,x,use_line_collection=True)

plt.xticks(fontsize=20)

plt.yticks([-1,0,1,2,3],fontsize=20)
plt.title(r'$x[n]$',fontsize=25)

plt.tight_layout(pad=0.1)
plt.savefig('PS10_Q3.eps')