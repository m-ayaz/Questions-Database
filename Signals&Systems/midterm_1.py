# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:34:30 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

t=np.linspace(-30,50,10000)
def x(t):
    return (2-(t-1)/2)*(abs(t-1)>0)*(abs(t-1)<1)+1*(abs(t-1)>1)*(abs(t-1)<2)

#x=z(t-1)
plt.figure()
plt.plot(t,x(t),linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-3,5])
plt.xticks(fontsize=15)
plt.yticks([0,1,2,3],fontsize=15)
plt.savefig('m_1Q.eps')
#plt.title(r'$x(t)$',fontsize=20)

opt1=x(t/2-3)
opt2=x((16-t)/2-3)
opt3=x(-t/2-3)
opt4=x(-(-16-t)/2-3)

plt.figure()
plt.plot(t,opt1,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-3,19])
plt.xticks([4,6,10,12],fontsize=15)
plt.yticks([0,1,2,3],fontsize=15)
plt.savefig('m_1Q_opt1.eps')

plt.figure()
plt.plot(t,opt2,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-3,19])
plt.xticks([4,6,10,12],fontsize=15)
plt.yticks([0,1,2,3],fontsize=15)
plt.savefig('m_1Q_opt2.eps')

plt.figure()
plt.plot(t,opt3,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-17,2])
plt.xticks([-4,-6,-10,-12],fontsize=15)
plt.yticks([0,1,2,3],fontsize=15)
plt.savefig('m_1Q_opt3.eps')

plt.figure()
plt.plot(t,opt4,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-17,2])
plt.xticks([-4,-6,-10,-12],fontsize=15)
plt.yticks([0,1,2,3],fontsize=15)
plt.savefig('m_1Q_opt4.eps')