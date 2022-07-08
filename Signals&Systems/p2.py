# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:38:33 2020

@author: Mostafa
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:28:26 2020

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

pi=np.pi

t=np.arange(-6,7,1)+2
x=[0,0,0,-2,3,1,0,-1,-3,2,0,0,0]

plt.stem(t,x)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.title(r'$x[n]$',fontsize=20)

plt.savefig('Q12_Final.eps')