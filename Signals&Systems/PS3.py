# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:35:41 2020

@author: Glory
"""

import numpy as np
import matplotlib.pyplot as plt
def x(t):
    return t*(t>0)*(t<1)+(t>1)*(t<2)+(3-t)*(t>2)*(t<3)

def y(t):
    return (t>0)*(t<2)

def x1(t):
    return t*(t>0)*(t<1)+(2-t)*(t>1)*(t<2)

def y1(t):
    return (t>0)*(t<1)

t=np.linspace(-1,4,1000)
plt.plot(t,x(t),linewidth=4)
plt.plot([0,0],[-0.2,1.2],'k')
plt.plot([-1.2,4.2],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PS3_Q2_1.eps')

plt.figure()

t=np.linspace(-1,4,1000)
plt.plot(t,y(t),linewidth=4)
plt.plot([0,0],[-0.2,1.2],'k')
plt.plot([-1.2,4.2],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PS3_Q2_2.eps')

plt.figure()

t=np.linspace(-1,4,1000)
plt.plot(t,x1(t),linewidth=4)
plt.plot([0,0],[-0.2,1.2],'k')
plt.plot([-1.2,4.2],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PS3_Q2_3.eps')

plt.figure()

t=np.linspace(-1,4,1000)
plt.plot(t,y1(t),linewidth=4)
plt.plot([0,0],[-0.2,1.2],'k')
plt.plot([-1.2,4.2],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PSol3_Q2.eps')