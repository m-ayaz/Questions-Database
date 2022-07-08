# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:40:26 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import remez, freqz
plt.close('all')

f1=1
f=5

filt_vec=[f1/f]

N=np.arange(0,10000,1)

filt_vec=np.sinc(2*N*f1/f)

''''''''''''''''''''''''''''''''''''
fsampling=1e6

analogfilter_taps=remez(50, [0, 100e3, 200e3, 0.5*fsampling], [1, 1e-40], fs=fsampling)
#            analogfilter_taps=[0]*35+[1]+[0]*34
w, h = freqz(analogfilter_taps, [1])
plt.subplot(2,2,1)
plt.plot(w,10*np.log10(h))
plt.title('Figure 1',fontsize=14)
plt.xticks([])





''''''''''''''''''''''''''''''''''''
fsampling=1e6

analogfilter_taps=remez(100, [0, 100e3, 200e3, 0.5*fsampling], [1, 1e-40], fs=fsampling)
#            analogfilter_taps=[0]*35+[1]+[0]*34
w, h = freqz(analogfilter_taps, [1])
plt.subplot(2,2,2)
plt.plot(w,10*np.log10(h))
plt.title('Figure 2',fontsize=14)
plt.xticks([])









''''''''''''''''''''''''''''''''''''
fsampling=1e6

analogfilter_taps=remez(30, [0, 100e3, 200e3, 0.5*fsampling], [1, 1e-40], fs=fsampling)
#            analogfilter_taps=[0]*35+[1]+[0]*34
w, h = freqz(analogfilter_taps, [1])
plt.subplot(2,2,3)
plt.plot(w,10*np.log10(h))
plt.title('Figure 3',fontsize=14)
plt.xticks([])
#plt.yticks([])












''''''''''''''''''''''''''''''''''''
fsampling=1e6

analogfilter_taps=remez(10, [0, 100e3, 200e3, 0.5*fsampling], [1, 1e-40], fs=fsampling)
#            analogfilter_taps=[0]*35+[1]+[0]*34
w, h = freqz(analogfilter_taps, [1])
plt.subplot(2,2,4)
plt.plot(w,10*np.log10(h))
plt.title('Figure 4',fontsize=14)
plt.xticks([])

plt.figure()

t=np.linspace(0,2*np.pi,10000)

x=np.cos(10*t)
#x=list(x)

y=list(x)+list(2*x)+list(2*x)+list(x)+list(x)+list(2*x)+list(x)

plt.plot(np.linspace(0,7,70000),y,'k',linewidth=0.8)
plt.plot([0,7],[2,2],'r')
plt.plot([0,7],[-2,-2],'r')
for i in range(8):
    plt.plot([i,i],[-2,2],'r',linewidth=2)
    
plt.title('ASK Modulated Signal',fontsize=20)
plt.xlabel('time (msec)',fontsize=14)
plt.yticks([-2,-1,0,1,2],fontsize=14)
plt.xticks(fontsize=14)







plt.figure()

plt.plot([0,3],[0,4],'k',linewidth=3)
plt.plot([3,7],[4,4],'k',linewidth=3,label=r'$S(t)$')
plt.plot([0,1],[0,0],'r',linewidth=3,label=r'$\bar S(t)$')
#plt.plot([1,1],[0.5,1],'r:',linewidth=3)
plt.plot([1,1],[0,0.5],'r',linewidth=3)
#plt.plot([1,1],[0.5,1.5],'r:',linewidth=3)
plt.plot([1,1,1.3],[.5,1.5,1.5],'r:',linewidth=3)
plt.legend(fontsize=15)
plt.grid('on')
plt.title('CVSD input (Analog Signal)',fontsize=20)
plt.xlabel('time (msec)',fontsize=14)
plt.ylabel('Voltage (volt)',fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylim([-0.3,5])