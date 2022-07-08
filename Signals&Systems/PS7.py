# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:56:47 2020

@author: Glory
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:35:41 2020

@author: Glory
"""

import numpy as np
import matplotlib.pyplot as plt
def x(t):
    return -1*(-2<t)*(t<-1)+t*(t<1)*(t>-1)+1*(1<t)*(t<2)

def Y(f):
    return -1*(-3<f)*(f<-2)+(f+1)*(f<-1)*(f>-2)+1*(2<f)*(f<3)+(f-1)*(f<2)*(f>1)

def z(t):
    return (np.abs(t)+1)*(np.abs(t)>0)*(np.abs(t)<1)+2*(np.abs(t)>1)*(np.abs(t)<2)

def X(f):
    return (np.abs(f)<1)*(1-np.abs(f))

def H(f):
    return (np.abs(f)<1)*1

def X2(f):
    return (np.abs(f)<1)*np.sqrt(1-f**2*(np.abs(f)<1))

#def y(t):
#    tx=abs(np.mod(t,3))
#    return -(-2<tx)*(tx<-1)+x*(abs(x)<1)+1*(1<tx)*(tx<2)
#
#def x1(t):
#    return t*(t>0)*(t<1)+(2-t)*(t>1)*(t<2)

t=np.linspace(-3,3,2000)
plt.plot(t,x(t),linewidth=4)
plt.plot([0,0],[-1.2,1.2],'k',linewidth=4)
plt.plot([-3,3],[-0,0],'k',linewidth=4)
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x(t)$',fontsize=20)
plt.savefig('PS6_Q1_4.eps')
plt.figure()





f=np.linspace(-4,4,2000)
plt.plot(f,Y(f),linewidth=4)
plt.plot([0,0],[-1.2,1.2],'k',linewidth=4)
plt.plot([-3,3],[-0,0],'k',linewidth=4)
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$Y(j\omega)$',fontsize=20)
plt.savefig('PS6_Q2_4.eps')
plt.figure()











t=np.linspace(-3,5,2000)
plt.plot(t,z(t-1),linewidth=4)
plt.plot([0,0],[-.2,2.5],'k',linewidth=4)
plt.plot([-3,5],[-0,0],'k',linewidth=4)
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x(t)$',fontsize=20)
plt.savefig('PS6_Q3_4.eps')
plt.figure()














f=np.linspace(-2,2,2000)
#plt.subplot(1,2,1)
plt.plot(f,X(f),linewidth=4)
plt.plot([0,0],[-.2,1.2],'k',linewidth=4)
plt.plot([-2,2],[-0,0],'k',linewidth=4)
#plt.grid('on')
plt.xticks([-1,1],[r'$-\omega_0$',r'$\omega_0$'],fontsize=20)
plt.yticks([0,1],fontsize=20)
plt.title(r'$X(j\omega)$',fontsize=20)
plt.savefig('PS7_Q7_1.eps')
plt.figure()



f=np.linspace(-2,2,2000)
#plt.subplot(1,2,2)
plt.plot(f,H(f),linewidth=4)
plt.plot([0,0],[-.2,1.2],'k',linewidth=4)
plt.plot([-2,2],[-0,0],'k',linewidth=4)
#plt.grid('on')
plt.xticks([-1,1],[r'$-\omega_0$',r'$\omega_0$'],fontsize=20)
plt.yticks([0,1],fontsize=20)
plt.title(r'$H(j\omega)$',fontsize=20)
plt.savefig('PS7_Q7_2.eps')
plt.figure()



f=np.linspace(-2,2,2000)
plt.plot(f,X2(f),linewidth=4)
plt.plot([0,0],[-.2,1.2],'k',linewidth=4)
plt.plot([-2,2],[-0,0],'k',linewidth=4)
#plt.grid('on')
plt.xticks([-1,1],[r'$-\omega_0$',r'$\omega_0$'],fontsize=20)
plt.yticks([0,1],fontsize=20)
plt.title(r'$X_2(j\omega)$',fontsize=20)
plt.savefig('PS7_Q7_3.eps')
plt.figure()




f=np.linspace(-2,2,2000)
plt.plot(f,X(f),linewidth=4)
plt.plot([0,0],[-.2,1.2],'k',linewidth=4)
plt.plot([-2,2],[-0,0],'k',linewidth=4)
#plt.grid('on')
plt.xticks([-1,1],[r'$-\omega_0$',r'$\omega_0$'],fontsize=20)
plt.yticks([0,1],fontsize=20)
plt.title(r'$X_1(j\omega)$',fontsize=20)
plt.savefig('PS7_Q7_4.eps')
plt.figure()





