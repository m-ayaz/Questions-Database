# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 02:47:17 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt
import math

x=np.linspace(-6,6,1000)

y=1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

z=np.array(list(map(math.erfc,x/np.sqrt(2))))/2

plt.plot(x,y,linewidth=3)

#plt.tight_layout()

plt.xlabel('X')

plt.xticks(fontsize=20)
plt.yticks([0,.1,.2,.3,.4],fontsize=20)

plt.title('PDF',fontsize=20)


plt.figure()
plt.plot(x,1-z,linewidth=3)

#plt.tight_layout()

plt.xlabel('X')

plt.xticks(fontsize=20)
plt.yticks([0,.25,.5,.75,1],fontsize=20)

plt.title('CDF',fontsize=20)