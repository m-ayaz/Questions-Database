# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:28:24 2020

@author: Glory
"""

import numpy as np
#from multiprocessing import Pool
import time
import os
import matplotlib.pyplot as plt

t1=np.linspace(-4,9,14)

t2=np.linspace(-2,20,23)

tu=np.linspace(-6,29,36)

x=[0]*4+5*[1]+[0]*5

y=[0]*4+6*[1]+3*[0]+6*[1]+4*[0]

y1=np.convolve(x,y)

plt.figure()
plt.stem(tu,y1)
plt.figure()

plt.stem(t1,x,use_line_collection=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([-4,9],[0,0],'k')
plt.ylim([-0.1,1.3])
plt.title(r'$x[n]$',fontsize=20)
plt.savefig('PS4_Q1_a_1.eps')
plt.figure()
plt.stem(t2,y,use_line_collection=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([-2,20],[0,0],'k')
plt.ylim([-0.1,1.3])
plt.title(r'$h[n]$',fontsize=20)
plt.savefig('PS4_Q1_a_2.eps')





t1=np.linspace(-6,6,13)
t2=np.linspace(-4,4,9)

x=[0,0,0,1,2,3,4,3,2,1,0,0,0]

y=[0,0,0,1,1,1,0,0,0]

y1=np.convolve(x,y)

tu=np.linspace(-10,10,21)

plt.figure()
plt.stem(tu,y1)
plt.grid('on')
plt.figure()

plt.stem(t1,x,use_line_collection=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([-6,6],[0,0],'k')
plt.ylim([-0.1,5])
plt.title(r'$x[n]$',fontsize=20)
plt.savefig('PS4_Q1_b_1.eps')
plt.figure()
plt.stem(t2,y,use_line_collection=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([-4,4],[0,0],'k')
plt.ylim([-0.1,1.3])
plt.title(r'$h[n]$',fontsize=20)
plt.savefig('PS4_Q1_b_2.eps')


plt.figure()




t=np.linspace(-2,4,1000)
x=np.sin(np.pi*t)*(t>0)*(t<2)+0
#h=

plt.plot(t,x,linewidth=4)
plt.plot([0,0],[-1.2,1.2],'k')
plt.plot([-1.2,4.2],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x(t)$',fontsize=20)
plt.savefig('PS4_Q2_a_1.eps')

plt.figure()

t=np.linspace(-1,4,1000)
h=2*(t>1)*(t<3)+0



plt.plot(t,h,linewidth=4)
plt.plot([0,0],[-.3,3.2],'k')
plt.plot([-1.2,4.2],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$h(t)$',fontsize=20)
plt.savefig('PS4_Q2_a_2.eps')


tu=np.linspace(-3,8,1999)
y1=np.convolve(x,h)/sum(h)
plt.figure()
plt.plot(tu,y1)
plt.figure()















plt.figure()



t=np.linspace(-1,6,1000)
xs=(t>1)*(t<2)+(t>4)*(t<5)+0
ys=(t>2)*(t<4)+0

tu=np.linspace(-2,12,1999)
y1=np.convolve(xs,ys)/140
plt.figure()
plt.plot(tu,y1)
plt.figure()

plt.plot(t,xs,linewidth=4)
plt.plot([0,0],[-.3,2.2],'k')
plt.plot([-1,6],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title(r'$x(t)$',fontsize=20)
plt.savefig('PS4_Q2_b_1.eps')

plt.figure()

plt.plot(t,ys,linewidth=4)
plt.plot([0,0],[-.3,2.2],'k')
plt.plot([-1,6],[-0,0],'k')
plt.grid('on')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$h(t)$',fontsize=20)
plt.savefig('PS4_Q2_b_2.eps')










t=np.linspace(-7,8,16)

x=[0,0,0,0,0,1,2,3,2,2,1,0,0,0,0,0]



plt.figure()
plt.stem(t,x,use_line_collection=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([-7,8],[0,0],'k')
plt.ylim([-0.1,4])
plt.yticks([1,2,3])
plt.title(r'$x[n]$',fontsize=20)
plt.savefig('PS4_Q3.eps')