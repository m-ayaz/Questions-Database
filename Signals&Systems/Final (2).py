# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:48:29 2022

@author: M
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import pi,exp,sin,cos,linspace

plt.close("all")

w=np.linspace(-pi,pi,10000)

x=(0.75*pi<abs(w))*(abs(w)<pi)*1+(3-4/pi*abs(w))*(abs(w)<=0.75*pi)

plt.plot(w,x,linewidth=3)
plt.plot([0,0],[-0.2,3.2],"k",linewidth=2)
plt.plot([-pi,pi],[0,0],"k",linewidth=2)

plt.xticks([-pi,-0.75*pi,0,0.75*pi,pi],[r"$-\pi$",r"$-\frac{3\pi}{4}$",r"$0$",r"$\frac{3\pi}{4}$",r"$\pi$"],fontsize=20)
plt.yticks([1,2,3],fontsize=20)
plt.grid("on")
plt.title(r"$H(e^{j\omega})$",fontsize=25)
plt.tight_layout(pad=0.1)

plt.figure()
x=[0,0,2,-4,1,3,0,0,0]
plt.stem([-2,-1,0,1,2,3,4,5,6],x)

plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.title(r"$x[n]$",fontsize=25)
plt.tight_layout(pad=0.1)