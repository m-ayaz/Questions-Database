# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:50:33 2021

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

from numpy import pi

t=np.linspace(0,2*pi,1000)

xc1=np.cos(t)
yc1=np.sin(t)

plt.plot(xc1,yc1)

plt.plot([0,0],[-2,2],'k')
plt.plot([-2,2],[0,0],'k')
plt.plot(1.5*np.cos(pi/4),1.5*np.sin(pi/4),'kx',markersize=14)
plt.plot(1.5*np.cos(pi/4),1.5*np.sin(-pi/4),'kx',markersize=14)
plt.plot(-0,0,'ko',markersize=14,mfc='None')
plt.plot(-0.5,0,'kx',markersize=14)
plt.plot(0.5*xc1,0.5*yc1,'k--',linewidth=0.5)
plt.plot(1.5*xc1,1.5*yc1,'k--',linewidth=0.5)
plt.text(1.5,0.1,r'$|z|=\frac{3}{2}$',fontsize=14)
plt.text(.1,0.1,r'$|z|=\frac{1}{2}$',fontsize=14)

plt.axis('equal')
plt.axis('off')
plt.tight_layout(pad=0)