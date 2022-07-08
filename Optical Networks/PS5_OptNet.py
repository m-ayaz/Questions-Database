# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:24:27 2020

@author: Glory
"""

import numpy as np
import matplotlib.pyplot as plt
sin=np.sin
cos=np.cos

y=[[-3,-1,1,3]]

x=np.array(y)

x=x+1j*x.T

t=np.linspace(0,2*np.pi,10000)

plt.plot(np.real(x),np.imag(x),'k.',markersize=20)
plt.plot([0,0],[-4,4],'k')
plt.plot([-4,4],[0,0],'k')
#plt.text(-0.5,4.5,'Inphase',fontsize=20)
plt.ylim(-5,5)
#h=plt.margins(0,0)
plt.axis('equal')
plt.grid('on')
plt.xticks(y[0],fontsize=20)
plt.yticks(y[0],fontsize=20)
plt.savefig('Unshaped.eps')
plt.savefig('Unshaped.png',dpi=500)

plt.figure()

plt.plot(2*sin(t),2*cos(t),color='r',linewidth=4)
plt.plot(2*sin(t),3+0.5*cos(t),color='g',linewidth=4)
plt.plot(-3+0.5*sin(t),2*cos(t),color='b',linewidth=4)
plt.plot(0.7*sin(t)+3,0.7*cos(t)+3,color='y',linewidth=4)

plt.plot(np.real(x),np.imag(x),'k.',markersize=20)
plt.plot([0,0],[-4,4],'k')
plt.plot([-4,8],[0,0],'k')
#plt.text(-0.5,4.5,'Inphase',fontsize=20)


plt.plot(2*sin(t),-3+0.5*cos(t),color='g',linewidth=4)


plt.plot(3+0.5*sin(t),2*cos(t),color='b',linewidth=4)


plt.plot(0.7*sin(t)-3,0.7*cos(t)+3,color='y',linewidth=4)
plt.plot(0.7*sin(t)+3,0.7*cos(t)-3,color='y',linewidth=4)
plt.plot(0.7*sin(t)-3,0.7*cos(t)-3,color='y',linewidth=4)
plt.legend([r'$P_1$',r'$P_2$',r'$P_3$',r'$P_4$'],fontsize=20)
#plt.legend([,'a','','a'])
#plt.plot(2*sin(t),2*cos(t),color='r',linewidth=4)
plt.xlim(-5,5)
#h=plt.margins(0,0)
plt.axis('equal')
plt.grid('on')
plt.xticks(y[0]+[5,7,9],['-3','-1','1','3'],fontsize=20)
#plt.xtickslabels(y[0]+[5,7,9],fontsize=20)
plt.yticks(y[0],['-3','-1','1','3'],fontsize=20)
plt.savefig('Shaped.eps')
plt.savefig('Shaped.png',dpi=500)