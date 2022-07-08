# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 02:01:50 2021

@author: Mostafa
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 01:57:38 2021

@author: Mostafa
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 01:38:54 2021

@author: Mostafa
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:27:51 2021

@author: Mostafa
"""

import numpy as np

import matplotlib.pyplot as plt

plt.close('all')

t=np.linspace(0,6.29,1000)

plt.plot(np.cos(t),np.sin(t),'k')

plt.plot([-2,2],[0,0],'k')
plt.plot([0,0],[-1.5,1.5],'k')
plt.text(0.7,0.7,r'$|z|=1$',fontsize=14)

plt.plot(2*np.cos(np.pi/6),2*np.sin(np.pi/6),'bx',markersize=15)
plt.text(2*np.cos(np.pi/6),2*np.sin(np.pi/6)*1.1,r'$z=2e^{j\frac{\pi}{6}}$',fontsize=12)
plt.plot(2*np.cos(np.pi/6),-2*np.sin(np.pi/6),'bx',markersize=15)
plt.text(2*np.cos(np.pi/6),-2*np.sin(np.pi/6)*1.2,r'$z=2e^{-j\frac{\pi}{6}}$',fontsize=12)

plt.plot(-0.5,0,'bx',markersize=15)
plt.text(-0.7,0.1,r'$z=-\frac{1}{2}$',fontsize=12)

plt.plot(-1.5,0,'bo',markersize=15,mfc='None')
plt.text(-1.8,0.15,r'$z=-\frac{3}{2}$',fontsize=12)

plt.axis('equal')
plt.axis('off')

plt.tight_layout(pad=0.1)









plt.figure()

f=np.linspace(-4,4,1000)

h=(1-abs(f)*4/3/np.pi)*(abs(f)<3*np.pi/4)

plt.plot(f,h)
plt.xticks([-np.pi,-3*np.pi/4,3*np.pi/4,np.pi],[r'$-\pi$',r'$-\frac{3\pi}{4}$',r'$\frac{3\pi}{4}$',r'$\pi$'],fontsize=20)
plt.yticks([0,1],fontsize=22)
plt.plot([0,0],[-0.2,1.3],'k')
plt.plot([-4,4],[0,0],'k')

plt.tight_layout(pad=0.1)








plt.figure()

f=np.linspace(-4,4,1000)

h=1*(abs(f)<np.pi/3)

plt.plot(f,h)
plt.xticks([-np.pi,-np.pi/3,np.pi/3,np.pi],[r'$-\pi$',r'$-\frac{\pi}{3}$',r'$\frac{\pi}{3}$',r'$\pi$'],fontsize=22)
plt.yticks([0,1],fontsize=22)
plt.plot([0,0],[-0.2,1.3],'k')
plt.plot([-4,4],[0,0],'k')

plt.tight_layout(pad=0.1)








plt.figure()

f=np.linspace(-1.5,1.5,1000)

x=abs(f)*(abs(f)<1)

plt.close('all')

plt.plot(f,x)
plt.plot([-1.5,1.5],[0,0],'k')
plt.plot([0,0],[-0.2,1.3],'k')

plt.xticks([-1,1],[r'$-2\pi$',r'$2\pi$'],fontsize=17)
plt.yticks([0,1],fontsize=17)
plt.xlabel(r'$\omega$',fontsize=22)
plt.tight_layout(pad=0.1)