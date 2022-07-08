# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:19:51 2020

@author: Mostafa
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:34:30 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

t=np.linspace(-30,50,10000)
def x(t):
    return (6-t)*(t>4)*(t<6)+t/2*(t>2)*(t<4)+1*(t>0)*(t<2)

#x=z(t-1)
plt.figure()
plt.plot(t,x(4-2*t),linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-3,5])
plt.xticks(fontsize=20)
plt.yticks([0,1,2,3],fontsize=20)
plt.savefig('m_2Q.eps')
#plt.title(r'$x(t)$',fontsize=20)

opt1=x(6-t)
opt2=x(t)
opt3=x(3-t)
opt4=x(t+5)

plt.figure()
plt.plot(t,opt1,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-2,8])
plt.xticks([0,2,5,6,8],fontsize=20)
plt.yticks([0,1,2,3],fontsize=20)
plt.savefig('m_2Q_opt1.eps')

plt.figure()
plt.plot(t,opt2,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-2,8])
plt.xticks([0,2,4,6],fontsize=20)
plt.yticks([0,1,2,3],fontsize=20)
plt.savefig('m_2Q_opt2.eps')

plt.figure()
plt.plot(t,opt3,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-5,5])
plt.xticks([3,1,-1,-3],fontsize=20)
plt.yticks([0,1,2,3],fontsize=20)
plt.savefig('m_2Q_opt3.eps')

plt.figure()
plt.plot(t,opt4,linewidth=4)
plt.plot([-30,50],[0,0],'k',linewidth=2)
plt.plot([0,0],[-0.2,3],'k',linewidth=2)
plt.xlim([-7,3])
plt.xticks([-5,-3,-1,1],fontsize=20)
plt.yticks([0,1,2,3],fontsize=20)
plt.savefig('m_2Q_opt4.eps')