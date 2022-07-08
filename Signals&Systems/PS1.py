# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:32:36 2020

@author: Glory
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:19:44 2020

@author: Glory
"""

import numpy as np
import matplotlib.pyplot as plt
n=100000
t=np.linspace(-2,3,n)
x=(t<0)*(t>-1)+(t<1)*(t>0)*(1+t)+(t>1)*(3-t)*(t<2)

x_t_n1=(t>0)*(t<1)+(t<2)*(t>1)*(t)+(t>2)*(4-t)*(t<3)

plt.plot(t,x,linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
h=plt.legend(['x(t)'],prop={'size': 20})
plt.grid('on')
plt.savefig('PS1_Q3.eps')
#h.FontSize=80