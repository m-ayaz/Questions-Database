# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 02:15:35 2019

@author: Life & Fun
"""

import numpy as np
import matplotlib.pyplot as plt
from texplotlib import savetex
arr=np.array
x=np.linspace(0,1,1000)
n=np.logspace(0,1,6)
n=np.floor(n)
n=set(n)
n=list(n)
n=arr([n]).transpose()
y=x**n/(x**n+(1-x)**n)
plt.plot(x,y.transpose())
leg=[]
for i in range(0,len(n)):
    leg.append('n = '+str(int(n[i][0])))
plt.legend(leg)
plt.xlabel('p',fontsize=15)
plt.ylabel('All the answers hold true',fontsize=15)
plt.tight_layout(pad=0.1)
# plt.savefig('HW3_Q6.pdf')
plt.savefig('HW3_Q6.eps')
savetex("HW3_Q6",save_type="full")
#print('عنوان'.encode(encoding="UTF-8"))