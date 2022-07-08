# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 01:39:01 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

x=[-1,-2,3,2,1,2,3,-2,-1]
plt.stem(np.arange(-3,6,1),x)
plt.xlim([-3.5,5.5])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title(r'$x[n]$',fontsize=20)
plt.savefig('PS10_Q3.eps')