# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:28:26 2020

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

pi=np.pi
omega=np.linspace(-2.5*pi,2.5*pi,10000)

x=np.arctan(np.tan(omega/2))

plt.plot(omega,abs(x),linewidth=3)

plt.xticks([-2*pi,-pi,0,pi,2*pi],[r'$-2\pi$',r'$-\pi$',r'$0$',r'$\pi$',r'$2\pi$'],fontsize=20)
plt.yticks([pi/2],['4'],fontsize=20)
#plt.ylim([0,1.8])
plt.plot([0,0],[-0.2,1.8],'k',linewidth=3)
plt.plot([-2.5*pi,2.5*pi],[0,0],'k',linewidth=2)

plt.plot([pi,pi],[0,pi/2],'k--',linewidth=2)
plt.plot([-pi,-pi],[0,pi/2],'k--',linewidth=2)

plt.plot([-2.5*pi,2.5*pi],[pi/2,pi/2],'k--',linewidth=2)

plt.title(r'$H(e^{j\omega})$',fontsize=20)

plt.savefig('Q8_Final.eps')