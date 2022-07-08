# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:02:44 2020

@author: Mostafa
"""

import matplotlib.pyplot as plt
import numpy as np

x1=np.linspace(-1,10)
x2=np.linspace(-1,10)
x3=np.linspace(-1,10)
y1=(30-2*x1)/3
y2=(24-3*x2)/2

u=2.4
w=8.4

plt.plot(x1,y1,'b')
plt.plot(x2,y2,'b')
plt.plot([0,0],[0,13],'k')
plt.plot([0,10],[0,0],'k')
plt.plot(x3,(-12*x3+105)/10,'r')
plt.xticks(np.linspace(0,10,11))
plt.yticks(np.linspace(0,10,11))
plt.grid('on')
plt.plot(u,w,'ro')
#plt.arrow(3,6.9,5/5,6/5)
plt.savefig('PSol6_Q3.png',dpi=1000)

#plt.axis('equal')