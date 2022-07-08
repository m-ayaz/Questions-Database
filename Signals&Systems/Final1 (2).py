# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:51:39 2021

@author: Mostafa
"""

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

x=[1,-1,1,3,1,-1,1]
t=list(range(-3+2,3+1+2))

plt.stem(t,x,use_line_collection=True)

plt.xticks(fontsize=20)

plt.yticks([-1,0,1,2,3],fontsize=20)
plt.title(r'$x[n]$',fontsize=25)

plt.tight_layout(pad=0.1)
plt.savefig('PS10_Q3.eps')

plt.figure()

w=np.linspace(-2*np.pi,2*np.pi,1000)
X=(abs(w)>np.pi)*(-1/np.pi*abs(w)+2)+(abs(w)<=np.pi)

plt.plot(w,X,linewidth=3)
plt.plot([0,0],[-0.2,1.3],'k',linewidth=2)
plt.plot([np.pi]*2,[-0,1],'k--',linewidth=2)
plt.plot([-np.pi]*2,[-0,1],'k--',linewidth=2)
plt.plot([-7,7],[0,0],'k',linewidth=2)
plt.xticks([-2*np.pi,-np.pi,np.pi,2*np.pi],[r'$-2\pi$',r'$-\pi$',r'$\pi$',r'$2\pi$'],fontsize=20)
plt.yticks([0,1],fontsize=20)
plt.title(r'$X(j\omega)$',fontsize=20)
plt.tight_layout(pad=0.1)
#plt.savefig()