# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:08:49 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

k=np.arange(-3,4,1)

p=-1+1j*k

plt.plot(np.real(p),np.imag(p),'x',markersize=13,mfc='none')

plt.xticks([-1,0],fontsize=18)
plt.yticks(k,[
        r'$-\frac{6\pi}{T}$',
        r'$-\frac{4\pi}{T}$',
        r'$-\frac{2\pi}{T}$',
        r'$0$',
        r'$\frac{2\pi}{T}$',
        r'$\frac{4\pi}{T}$',
        r'$\frac{6\pi}{T}$',
        ],fontsize=18)
plt.xlim([-2,2])
plt.plot([0,0],[-3.4,3.4],'k',linewidth=2)
plt.plot([-2,2],[0,0],'k',linewidth=2)
plt.savefig('PSol11_Q11.eps')

def H(omega,z=None,p=None):
    if z==None:
        z_dist=1
    else:
        z_dist=np.prod(1j*omega-np.array([z]).T,0)
    if list(p)==None:
        p_dist=1
    else:
        p_dist1=1j*omega-np.array([p]).T
        p_dist=np.prod(p_dist1,0)
    return np.abs(
            z_dist/p_dist
            ),p_dist1,z_dist
    
plt.figure()
k=np.arange(-30,40,1)

po=-1+1j*k
omega=np.linspace(-600,600,100000)
a,b,c=H(omega,p=po)
#plt.plot(omega,1/np.sqrt(
#        (1+(omega)**2)*
#        (1+(omega+1)**2)*
#        (1+(omega+2)**2)*
#        (1+(omega-2)**2)*
#        (1+(omega-1)**2)*
#        (1+(omega-3)**2)*
#        (1+(omega+3)**2)*
#        (1+(omega-4)**2)*
#        (1+(omega-5)**2)*
#        (1+(omega-6)**2)*
#        (1+(omega+7)**2)*
#        (1+(omega-7)**2)*
#        (1+(omega+6)**2)*
#        (1+(omega+5)**2)*
#        (1+(omega+4)**2)*1
#        ),linewidth=3)
plt.plot(omega,5+np.cos(2*pi*omega/300))
plt.plot([-600,600],[0,0],'k',linewidth=2)
plt.plot([0,0],[0,7],'k',linewidth=2)
plt.xticks([])
plt.yticks([])
#plt.xlim([-6,10])
plt.savefig('PSol11_Q11_2.eps')