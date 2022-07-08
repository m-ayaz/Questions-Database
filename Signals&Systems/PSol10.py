# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:40:24 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt
pi=np.pi

x=np.array([0,0,-1,-2,3,2,1,2,3,-2,-1,0,0,0,0])
y=np.array([0,0,0,0,-1,-2,3,2,1,2,3,-2,-1,0,0])

plt.stem(np.arange(-7,8,1),x/2+y/2)

plt.xticks(fontsize=13)

plt.yticks([2,1,-1,-.5],fontsize=13)

plt.savefig('PSol10_Q3.eps')

omega=np.linspace(-6,6,10000)
def H(omega,a):
    return np.angle(
            (-a+np.exp(-1j*omega))/(1-a*np.exp(-1j*omega))
            )
plt.figure()
plt.plot(omega,H(omega,0.5),linewidth=3)
plt.xticks([-pi,0,pi],[r'$-\pi$',r'$0$',r'$\pi$'],fontsize=15)
plt.yticks([-pi,0,pi],[r'$-\pi$',r'$0$',r'$\pi$'],fontsize=15)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[-4,4],'k',linewidth=2)
#plt.title('')
plt.savefig('PSol10_Q7_1.eps')

plt.figure()
plt.plot(omega,H(omega,-0.5),linewidth=3)
plt.xticks([-pi,0,pi],[r'$-\pi$',r'$0$',r'$\pi$'],fontsize=15)
plt.yticks([-pi,0,pi],[r'$-\pi$',r'$0$',r'$\pi$'],fontsize=15)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[-4,4],'k',linewidth=2)
plt.savefig('PSol10_Q7_2.eps')

n=np.arange(-5,10,1)
x=(0.5)**n*(n>=0)
y=1.25*(0.5)**n-0.75*(-0.5)**n
y=y*(n>=0)

plt.figure()
plt.stem(n,x)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title(r'$x[n]$',fontsize=15)
plt.savefig('PSol10_Q7_4_1.eps')

plt.figure()
plt.stem(n,y)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title(r'$y[n]$',fontsize=15)
plt.savefig('PSol10_Q7_4_2.eps')

''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''






def H(omega,z=None,p=None):
    if z==None:
        z_dist=1
    else:
        z_dist=np.prod(1j*omega-np.array([z]).T,0)
    if p==None:
        p_dist=1
    else:
        p_dist=np.prod(1j*omega-np.array([p]).T,0)
    return np.abs(
            z_dist/p_dist
            )
    
plt.figure()
plt.plot(omega,H(omega,p=[-0.2+1j,-0.2-1j]),linewidth=3)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[0,3],'k',linewidth=2)
plt.xticks([])
plt.yticks([])
plt.savefig('PSol10_Q7_zp_1.eps')

plt.figure()
plt.plot(omega,H(omega,z=[-0.2+1j,-0.2-1j],p=[-1]),linewidth=3)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[0,3],'k',linewidth=2)
plt.xticks([])
plt.yticks([])
plt.ylim([-0.2,3])
plt.savefig('PSol10_Q7_zp_2.eps')

plt.figure()
plt.plot(omega,H(omega,z=[-1/3],p=[-3/3]),linewidth=3)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[0,3],'k',linewidth=2)
plt.xticks([])
plt.yticks([])
plt.ylim([-0.2,1.3])
plt.savefig('PSol10_Q7_zp_3.eps')

plt.figure()
plt.plot(omega,H(omega,z=[20,10],p=[-10,-20]),linewidth=3)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[0,3],'k',linewidth=2)
plt.xticks([])
plt.yticks([])
plt.ylim([-0.2,2])
plt.savefig('PSol10_Q7_zp_4.eps')

plt.figure()
plt.plot(omega,H(omega,z=[-0.2+1j,-0.2-1j,1],p=[-1]),linewidth=3)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[0,3],'k',linewidth=2)
plt.xticks([])
plt.yticks([])
plt.ylim([-0.2,2])
plt.savefig('PSol10_Q7_zp_5.eps')

plt.figure()
plt.plot(omega,H(omega,z=[-1/3,1/3]),linewidth=3)
plt.plot([-6,6],[0,0],'k',linewidth=2)
plt.plot([0,0],[0,3],'k',linewidth=2)
plt.xticks([])
plt.yticks([])
plt.ylim([-0.2,2])
plt.savefig('PSol10_Q7_zp_6.eps')