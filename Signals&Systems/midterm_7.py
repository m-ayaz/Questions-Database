# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:14:40 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

def X(f):
    return (1-abs(f/1.5))*(abs(f)<1.5)

f=np.linspace(-60,60,10000)
plt.plot(f,X(f),linewidth=4)
plt.plot([0,0],[0,1.3],'k',linewidth=2)
plt.plot([-3,3],[0,0],'k',linewidth=2)
plt.xticks([-1.5,1.5],fontsize=15)
plt.yticks(fontsize=15)
plt.xlim([-3,3])


plt.figure()
Y=X(2*f)+X(2*f-2)+X(2*f+2)+X(2*f-4)+X(2*f+4)+X(2*f-6)+X(2*f+6)
plt.plot(f,2*Y,linewidth=4)
plt.plot([0,0],[0,2.7],'k',linewidth=2)
plt.plot([-60,60],[0,0],'k',linewidth=2)
plt.xticks([-0.5,0.5],[r'$-\pi$',r'$\pi$'],fontsize=25)
plt.yticks([0,4/3,2],['0',r'$\frac{4}{3}$','2'],fontsize=25)
plt.plot([.5,.5],[0,4/3],'k--',linewidth=2)
plt.plot([-.5,-.5],[0,4/3],'k--',linewidth=2)
#plt.plot([-7,7],[4/3,4/3],'k--',linewidth=2)
plt.xlim([-1,1])



plt.figure()
Y=X(1.5*f)+X(1.5*f-1.5)+X(1.5*f+1.5)+X(1.5*f-3)+X(1.5*f+3)#+X(1.5*f-6)+X(2*f+6)
plt.plot(f,1.5*Y,linewidth=4)
plt.plot([0,0],[0,2.7],'k',linewidth=2)
plt.plot([-60,60],[0,0],'k',linewidth=2)
plt.xticks([-0.5,0.5],[r'$-\pi$',r'$\pi$'],fontsize=25)
plt.yticks([0,2],['0','2'],fontsize=25)
plt.plot([.5,.5],[0,6/3],'k--',linewidth=2)
plt.plot([-.5,-.5],[0,6/3],'k--',linewidth=2)
plt.xlim([-1,1])


plt.figure()
Y=X(f)+X(f-1)+X(f+1)+X(f-2)+X(f+2)+X(f-3)+X(f+3)
plt.plot(f,Y,linewidth=4)
plt.plot([0,0],[0,2.7],'k',linewidth=2)
plt.plot([-60,60],[0,0],'k',linewidth=2)
plt.xticks([-0.5,0.5],[r'$-\pi$',r'$\pi$'],fontsize=25)
plt.yticks([0,2],['0','2'],fontsize=25)
plt.plot([.5,.5],[0,6/3],'k--',linewidth=2)
plt.plot([-.5,-.5],[0,6/3],'k--',linewidth=2)
plt.xlim([-1,1])
