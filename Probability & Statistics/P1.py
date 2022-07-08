# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:07:02 2019

@author: Life & Fun
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-1,5,10000)
y=np.exp(x)/(np.exp(x)+1)
y=(1+x*np.exp(-x))*(x>0)
plt.plot(x,y)
plt.plot(x,x*0+1,'--')
plt.plot(x,x*0,'--')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.savefig('Q2C.eps')