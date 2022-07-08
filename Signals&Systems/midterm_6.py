# -*- coding: utf-8 -*-
"""
Created on Tue May 26 00:07:47 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

f=np.linspace(-30,50,10000)
def ab_X(f):
    return abs(3*f)*(abs(f)<3*np.pi)

def an_X(f):
    return np.sign(f)*np.pi/2

plt.figure()
plt.plot(f,ab_X(f),linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-1.2,35],'k',linewidth=2)
plt.xlim([-13,13])
plt.xticks([-3*np.pi,3*np.pi],[r'$-3\pi$',r'$3\pi$'],fontsize=15)
plt.yticks([0,9*np.pi],[r'$0$',r'$9\pi$'],fontsize=15)
#plt.yticks([-1,0,1],fontsize=20)
plt.title(r'$|X(j\omega)|$',fontsize=20)
plt.savefig('_21Q_ab.eps')

plt.figure()
plt.plot(f,an_X(f),linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-2,2],'k',linewidth=2)
plt.xlim([-13,13])
plt.xticks([-3*np.pi,3*np.pi],[r'$-3\pi$',r'$3\pi$'],fontsize=15)
plt.yticks([-np.pi/2,np.pi/2],[r'$-\frac{\pi}{2}$',r'$\frac{\pi}{2}$'],fontsize=25)
#plt.yticks([-1,0,1],fontsize=20)
plt.title(r'$\angle X(j\omega)$',fontsize=20)
plt.savefig('_21Q_an.eps')