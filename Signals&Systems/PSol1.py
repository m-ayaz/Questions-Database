# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 14:02:46 2020

@author: Glory
"""

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
t=np.linspace(-20,20,n)
def x(t):
    return (t<0)*(t>-1)+(t<1)*(t>0)*(1+t)+(t>1)*(3-t)*(t<2)

#x_t_n1=(t>0)*(t<1)+(t<2)*(t>1)*(t)+(t>2)*(4-t)*(t<3)

#x_2t_n1=(2*t>0)*(2*t<1)+(2*t<2)*(2*t>1)*(2*t)+(2*t>2)*(4-2*t)*(2*t<3)

#x=(t<0)*(t>-1)+(t<1)*(t>0)*(1+t)+(t>1)*(3-t)*(t<2)

#plt.plot(t,x,linewidth=3.5)
#plt.plot(t,np.zeros(n),color='k',linewidth=2)
#plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
#h=plt.legend(['x(t)'],prop={'size': 20})
#plt.savefig('PS1_Q3.eps')

#plt.figure()
plt.plot(t,x(t-1),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([1,1],[0,1],linestyle=':',color='k')
plt.plot([2,2],[0,2],linestyle=':',color='k')
h=plt.legend([r'$x(t-1)$'],prop={'size': 20})
plt.xlim([-1,5])
plt.savefig('PSol1_Q3_a_1.eps')

plt.figure()
plt.plot(t,x(2*t-1),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([0.5,0.5],[0,1],linestyle=':',color='k')
plt.plot([1,1],[0,2],linestyle=':',color='k')
h=plt.legend([r'$x(2t-1)$'],prop={'size': 20})
plt.xlim([-1,5])
plt.savefig('PSol1_Q3_a_2.eps')
###########################################
plt.figure()
plt.plot(t,x(t-3),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([3,3],[0,1],linestyle=':',color='k')
plt.plot([4,4],[0,2],linestyle=':',color='k')
h=plt.legend([r'$x(t-3)$'],prop={'size': 20})
plt.xlim([1,9])
plt.savefig('PSol1_Q3_b_1.eps')

plt.figure()
plt.plot(t,x(t/2-3),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([6,6],[0,1],linestyle=':',color='k')
plt.plot([8]*2,[0,2],linestyle=':',color='k')
h=plt.legend([r'$x\left(\frac{t}{2}-3\right)$'],prop={'size': 20})
plt.xlim([2,15])
plt.savefig('PSol1_Q3_b_2.eps')

plt.figure()
plt.plot(t,x(-t/2-3),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([-6]*2,[0,1],linestyle=':',color='k')
plt.plot([-8]*2,[0,2],linestyle=':',color='k')
h=plt.legend([r'$x\left(-\frac{t}{2}-3\right)$'],prop={'size': 20})
plt.xlim([-18,-2])
plt.savefig('PSol1_Q3_b_3.eps')
###########################################
plt.figure()
plt.plot(t,x(t/2),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([2]*2,[0,2],linestyle=':',color='k')
#plt.plot([4,4],[0,2],linestyle=':',color='k')
h=plt.legend([r'$x\left(\frac{t}{2}\right)$'],prop={'size': 20})
plt.xlim([-3,8])
plt.savefig('PSol1_Q3_c_1.eps')

plt.figure()
plt.plot(t,x(t/2+3/2),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([-3]*2,[0,1],linestyle=':',color='k')
plt.plot([-1]*2,[0,2],linestyle=':',color='k')
h=plt.legend([r'$x\left(\frac{t+3}{2}\right)$'],prop={'size': 20})
plt.xlim([-7,4])
plt.savefig('PSol1_Q3_c_2.eps')

plt.figure()
plt.plot(t,-2*x(t/2+3/2),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(-4.5,.5,n),color='k',linewidth=2)
plt.plot([-3]*2,[0,-2],linestyle=':',color='k')
plt.plot([-1]*2,[0,-4],linestyle=':',color='k')
h=plt.legend([r'$-2x\left(\frac{t+3}{2}\right)$'],prop={'size': 20})
plt.xlim([-7,8])
plt.savefig('PSol1_Q3_c_3.eps')
###########################################
plt.figure()
plt.plot(t,x(t/3),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([3]*2,[0,2],linestyle=':',color='k')
#plt.plot([4,4],[0,2],linestyle=':',color='k')
h=plt.legend([r'$x\left(\frac{t}{3}\right)$'],prop={'size': 20})
plt.xlim([-5,8])
plt.savefig('PSol1_Q3_d_1.eps')

plt.figure()
plt.plot(t,x(t/3+2/3),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([-2]*2,[0,1],linestyle=':',color='k')
plt.plot([1]*2,[0,2],linestyle=':',color='k')
h=plt.legend([r'$x\left(\frac{t+2}{3}\right)$'],prop={'size': 20})
plt.xlim([-7,5])
plt.savefig('PSol1_Q3_d_2.eps')

plt.figure()
plt.plot(t,x(-t/3+2/3),linewidth=3.5)
plt.plot(t,np.zeros(n),color='k',linewidth=2)
plt.plot(np.zeros(n),np.linspace(0,2.2,n),color='k',linewidth=2)
plt.plot([-1]*2,[0,2],linestyle=':',color='k')
plt.plot([2]*2,[0,1],linestyle=':',color='k')
h=plt.legend([r'$x\left(\frac{2-t}{3}\right)$'],prop={'size': 20})
plt.xlim([-7,8])
plt.savefig('PSol1_Q3_d_3.eps')
#h.FontSize=80