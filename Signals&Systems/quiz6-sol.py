# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:05:48 2021

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

plt.close('all')

plt.plot([-1.5,1.5],[0,0],'k')
plt.plot([0,0],[-1.5,1.5],'k')

t=np.linspace(0,6.3,1000)

plt.plot(np.cos(t),np.sin(t),':')

plt.plot(4/3,0,'gx',markersize=12)
plt.plot(2/3,0,'gx',markersize=12)
plt.plot(1,0,'go',markersize=12,mfc='None')
plt.plot(-0.02,0,'go',markersize=12,mfc='None')
plt.plot(0.02,0,'go',markersize=12,mfc='None')

plt.plot(4/3*np.cos(t),4/3*np.sin(t),'r')
plt.plot(2/3*np.cos(t),2/3*np.sin(t),'r')
plt.fill(np.append(4/3*np.cos(t),2/3*np.cos(t)),np.append(4/3*np.sin(t),2/3*np.sin(-t)),alpha=0.1,color='c')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.axis('equal')

plt.tight_layout(pad=0.1)