# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:27:51 2021

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

plt.close('all')

f=np.linspace(-3,3,1000)

f1=f.copy()

#f=2*np.mod(f/2-0.375,0.75)-0.75

x=(1-abs(f))*(abs(f)<=0.5)+0.5*(abs(f)>0.5)*(abs(f)<0.75)

plt.plot(f1,x)

plt.plot([0,0],[-0.2,1.2],'k')
plt.plot([-3,3],[0,0],'k')
#plt.plot([0.75]*2,[0,0.5],'k:')
#plt.plot([-0.75]*2,[0,0.5],'k:')
#plt.plot([3*0.75]*2,[0,0.5],'k:')
#plt.plot([-3*0.75]*2,[0,0.5],'k:')
#plt.plot([0.75]*2,[0,0.25],'k:')
#plt.plot([-0.75]*2,[0,0.25],'k:')

plt.yticks([0.5,1],fontsize=15)
plt.xticks([-3*0.75,-1*0.75,0.75,3*0.75],fontsize=15)
plt.xlim([-1.5,1.5])

plt.tight_layout(pad=0.1)








plt.figure()

x=[0,1]*3+[0]



plt.plot([len(x)/2-0.5]*2,[-0.2,1.2],'k')
plt.plot([0,len(x)-1],[0,0],'k')

plt.yticks([0,1],fontsize=16)
plt.xticks([0,2,4,6],[r'$-3\pi$',r'$-\pi$',r'$\pi$',r'$3\pi$'],fontsize=16)

plt.plot(x)
plt.title(r'$F_s=2$',fontsize=20)
plt.tight_layout(pad=0.1)


x=[0,1,0]*3

plt.figure()

plt.plot([len(x)/2-0.5]*2,[-0.2,1.2],'k')
plt.plot([0,len(x)-1],[0,0],'k')

plt.yticks([0,1],fontsize=16)
plt.xticks([0,2,3,5,6,8],[r'$-4\pi$',r'$-2\pi$',r'$-\pi$',r'$\pi$',r'$2\pi$',r'$4\pi$'],fontsize=16)

plt.plot(x)
plt.title(r'$F_s=3$',fontsize=20)
plt.tight_layout(pad=0.1)