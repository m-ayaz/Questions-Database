# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:14:58 2020

@author: Mostafa
"""

import matplotlib.pyplot as plt
import numpy as np

x=3

plt.plot([0,0],[0,2],'k',linewidth=x)
plt.plot([2,0],[2,2],'k',linewidth=x)
plt.plot([0,2],[0,0],'k',linewidth=x)
plt.plot([2,2],[0,2],'k',linewidth=x)
t=np.linspace(0,7,10000)
plt.plot(np.sin(t)+1,np.cos(t)+1,linewidth=x)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()