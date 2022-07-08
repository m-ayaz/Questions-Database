# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 10:46:13 2021

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

t=np.linspace(0,6.3,1000)
x=np.cos(t)
y=np.sin(t)

a=-0.8

plt.plot([0,0],[-1.2,1.2],'k',linewidth=2)
plt.plot([-1.2,2.2],[0,0],'k',linewidth=2)

plt.plot(2,0,'bo',mfc='None',markersize=13)
plt.plot(0.5,0,'bo',mfc='None',markersize=13)
plt.plot(a,0,'bx',mfc='None',markersize=13)

plt.text(2,0.05,r'$2$',fontsize=14)
plt.text(0.5,0.1,r'$\frac{1}{2}$',fontsize=14)
plt.text(a+0.05,0.05,r'$a$',fontsize=14)

plt.text(0.71,0.71,r'$|z|=1$',fontsize=14)
plt.text(a*0.71,a*0.71,r'$|z|=|a|$',fontsize=14)

plt.plot(x,y,'k')
plt.plot(x*abs(a),y*abs(a),'k:')
#plt.xlim([-1.2,1.2])
#plt.ylim([-1.2,1.2])





plt.axis('equal')
plt.tight_layout(pad=0.1)
plt.axis('off')


plt.figure()

t=np.linspace(0,6.3,1000)
x=np.cos(t)
y=np.sin(t)

#a=-0.8

plt.plot([0,0],[-1.2,1.2],'k',linewidth=2)
plt.plot([-1.2,3.2],[0,0],'k',linewidth=2)

plt.plot(3,0,'bo',mfc='None',markersize=13)
plt.plot(2,0,'bx',mfc='None',markersize=13)
plt.plot(5/6,0,'bx',mfc='None',markersize=13)

plt.text(3,0.05,r'$3$',fontsize=14)
plt.text(2,0.05,r'$2$',fontsize=14)
plt.text(5/6,0.05,r'$\frac{5}{6}$',fontsize=14)

plt.text(0.71,0.71,r'$|z|=1$',fontsize=14)
#plt.text(a*0.71,a*0.71,r'$|z|=|a|$',fontsize=14)

plt.plot(x,y,'k')
#plt.plot(x*abs(a),y*abs(a),'k:')
#plt.xlim([-1.2,1.2])
#plt.ylim([-1.2,1.2])





plt.axis('equal')
plt.tight_layout(pad=0.1)
plt.axis('off')