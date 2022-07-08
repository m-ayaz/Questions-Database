# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:35:41 2020

@author: Glory
"""

import numpy as np
import matplotlib.pyplot as plt
def x(t):
    return abs(np.mod(t,2)-1)

def y(t):
    tx=abs(np.mod(t,3))
    return 2*(0<tx)*(tx<1)+1*(1<tx)*(tx<2)

def x1(t):
    return t*(t>0)*(t<1)+(2-t)*(t>1)*(t<2)

t=np.linspace(-4.5,4.5,2000)
plt.plot(t,x(t),linewidth=4)
plt.plot([0,0],[-0.2,1.2],'k')
plt.plot([-4.5,4.5],[-0,0],'k')
plt.grid('on')
plt.xticks(np.linspace(-4,4,9),fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x(t)$',fontsize=20)
plt.savefig('PS5_Q1_1.eps')
plt.figure()

#plt.arrow([0,0],[1,1],1,2)


t=np.linspace(-4.5,4.5,2000)
plt.plot(t,y(t),linewidth=4)
plt.plot([0,0],[-0.2,2.8],'k')
plt.plot([-4.5,4.5],[-0,0],'k')
plt.grid('on')
plt.xticks(np.linspace(-4,4,9),fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x(t)$',fontsize=20)
plt.savefig('PS5_Q1_2.eps')
plt.figure()
#t=np.linspace(-1,4,1000)
#plt.plot(t,x(t),linewidth=4)
#plt.plot([0,0],[-0.2,1.2],'k')
#plt.plot([-1.2,4.2],[-0,0],'k')
#plt.grid('on')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#plt.savefig('PS3_Q2_1.eps')
#
#plt.figure()
#
#t=np.linspace(-1,4,1000)
#plt.plot(t,y(t),linewidth=4)
#plt.plot([0,0],[-0.2,1.2],'k')
#plt.plot([-1.2,4.2],[-0,0],'k')
#plt.grid('on')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#plt.savefig('PS3_Q2_2.eps')
#
#plt.figure()
#
#t=np.linspace(-1,4,1000)
#plt.plot(t,x1(t),linewidth=4)
#plt.plot([0,0],[-0.2,1.2],'k')
#plt.plot([-1.2,4.2],[-0,0],'k')
#plt.grid('on')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#plt.savefig('PS3_Q2_3.eps')