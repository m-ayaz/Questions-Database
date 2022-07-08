# -*- coding: utf-8 -*-
"""
Created on Tue May  4 01:45:34 2021

@author: Mostafa
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:26:30 2021

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

plt.close('all')

t=np.linspace(-3,4,1000)

t1=t.copy()

#t=np.mod(t,2)

x=(t>-1)*(2)*(t<0)+(t>=0)*(t<=1)*(2-t)+(t>1)*(t<=2)*(t)+(t>2)*(t<=3)*2
x=x/2

x=(t>-1)*(1)*(t<0)+(t>=0)*(t<=1)*(1-t)

plt.plot(t1,x)

plt.plot([0,0],[-0.2,1.2],'k')
#plt.plot([1,1],[0,1],'k--')
#plt.plot([-1,-1],[0,1],'k--')
#plt.plot([3,3],[0,1],'k--')
#plt.plot([-3,-3],[0,1],'k--')
plt.plot([-4,4],[0,0],'k')
plt.plot([-4,4],[1,1],'k--')
#plt.plot([-4,4],[1/2,1/2],'k--')

#plt.axis('off')

plt.yticks([0,1],fontsize=20)
plt.xticks([-3,-2,-1,0,1,2,3],fontsize=16)
plt.tight_layout(pad=.3)
#plt.title('$x(t)$',fontsize=15)