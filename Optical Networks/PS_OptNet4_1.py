# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:53:53 2021

@author: Mostafa
"""

import numpy as np
from scipy.stats import norm

from numpy import sqrt

import matplotlib.pyplot as plt

N=100

noise=np.random.randn(1,16*N)+np.random.randn(1,16*N)*1j

noise=noise/3

u=np.array([[-3,-1,1,3]])

x=u+u.T*1j

u=[-5.5,5.5]

plt.plot(np.real(x),np.imag(x),'ko')

plt.plot([2,2],u,'r')
plt.plot([-2,-2],u,'r')
#plt.plot([-4,-4],u,'r')
#plt.plot([4,4],u,'r')
plt.plot([0,0],u,'r')

#plt.plot(u,[4,4],'r')
plt.plot(u,[2,2],'r')
#plt.plot(u,[-4,-4],'r')
plt.plot(u,[-2,-2],'r')
plt.plot(u,[0,0],'r')
w=noise+np.kron(np.reshape(x,[1,16])[0],[1]*N)
plt.plot(np.real(w),np.imag(w),'b.',markersize=2)


plt.axis('equal')
plt.xticks([-3,-1,1,3],fontsize=15)
plt.yticks([-3,-1,1,3],fontsize=15)