# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:25:51 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

f=np.linspace(-5,5,1000)

H1=[1,1,2,1]*3+[1]
X=[1,0.5,0.5,0.5]*3+[1]

plt.plot(H1,linewidth=3)
plt.plot([6,6],[-0.2,2.7],'k')
plt.plot([0,12],[0,0],'k')
plt.xticks(np.arange(2,12,2),[r'$-2\pi$',r'$-\pi$',r'$0$',r'$\pi$',r'$2\pi$'],fontsize=15)
plt.yticks([0,1,2],fontsize=15)
plt.title(r'$\hat H(e^{j\omega})$',fontsize=20)
plt.savefig('SQ_1.eps')

plt.figure()

plt.plot(X,linewidth=3)
plt.plot([6,6],[-0.2,1.3],'k')
plt.plot([0,12],[0,0],'k')
plt.xticks(np.arange(2,12,2),[r'$-2\pi$',r'$-\pi$',r'$0$',r'$\pi$',r'$2\pi$'],fontsize=15)
plt.yticks([0,1],fontsize=15)
plt.title(r'$\hat X(e^{j\omega})$',fontsize=20)
plt.savefig('SQ_2.eps')

plt.figure()

Y=H1*np.array(X)

plt.plot(Y,linewidth=3)
plt.plot([6,6],[-0.2,1.3],'k')
plt.plot([0,12],[0,0],'k')
plt.xticks(np.arange(2,12,2),[r'$-2\pi$',r'$-\pi$',r'$0$',r'$\pi$',r'$2\pi$'],fontsize=15)
plt.yticks([0,1],fontsize=15)
plt.title(r'$\hat Y(e^{j\omega})$',fontsize=20)
plt.savefig('SQ_3.eps')


plt.figure()
Y1=[0]*0+list(Y[0:5])+[0]*0

plt.plot(Y1,'b',linewidth=3)
plt.plot([0,0],[0,1],'b',linewidth=3)
plt.plot([4,4],[0,1],'b',linewidth=3)
plt.plot([2,2],[-0.2,1.3],'k')
plt.plot([-3,7],[0,0],'k')
plt.xticks([0,1,2,3,4],[r'$-\pi R$',r'$-\frac{\pi R}{2}$',r'$0$',r'$\frac{\pi R}{2}$',r'$\pi R$'],fontsize=15)
plt.yticks([0,1],fontsize=15)
plt.title(r'$Y({j\omega})$',fontsize=20)
plt.savefig('SQ_4.eps')
