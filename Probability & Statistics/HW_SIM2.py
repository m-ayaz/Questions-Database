# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 22:17:26 2020

@author: Mostafa
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

n=100
m=1000

p=0.2

x=np.random.rand(m,n)<p

x=x+0

y=np.sum(x,1)

#plt.plot(y)

#z=[]

u=y==np.linspace([0],[n],n+1)

z=np.sum(u/m,1)
#############################
var=n*p*(1-p)
avg=n*p
gaussian=1/np.sqrt(var*np.pi*2)*np.exp(-0.5*(np.arange(n+1)-avg)**2/var)
plt.plot(z)
plt.plot(gaussian)