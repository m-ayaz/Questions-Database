# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:10:00 2020

@author: Glory
"""


import numpy as np
#from multiprocessing import Pool
import time
import os
import matplotlib.pyplot as plt


x=[1,2,3,2,2,1,0,0,0,0]

y=[0]

for i in range(10):
    y.append(x[i]-2*y[-1])
    
plt.stem([-3,-2,-1,0,1,2,3],y[0:7])
plt.grid('on')
plt.title(r'$y[n]$',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PSol4_Q3.eps')
plt.savefig('PSol4_Q3.png')
plt.figure()

plt.figure()
plt.stem([-2,-1,0,1,2,3,4],[0,0,1,-2,1,0,0])
plt.grid('on')
plt.title(r'$u_2[n]$',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('PSol4_Q5.eps')
plt.savefig('PSol4_Q5.png')
plt.figure()

    

def x(n):
    return (n>=0)*(n<=4)

def h(n):
    return (n>=2)*(n<=7)+(n>=11)*(n<=16)

n=np.arange(-7,25,1)
n1=np.arange(-2,11,1)
n2=np.arange(-2,25,1)
#a=9
x1=[0]*4+[1]*5+[0]*4
h1=[1]*6+[0]*3+[1]*6

plt.stem(n2,np.convolve(x1,h1),use_line_collection=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x[n]*h[n]$',fontsize=20)
plt.savefig('PSol4_Q1_a.eps')
plt.savefig('PSol4_Q1_a.png')

plt.figure()

x2=[0,0,0,1,2,3,4,3,2,1,0,0,0]
h2=[1,1,1]

n2=np.arange(-2-5,13-5,1)

plt.stem(n2,np.convolve(x2,h2),use_line_collection=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x[n]*h[n]$',fontsize=20)
plt.savefig('PSol4_Q1_b.eps')
plt.savefig('PSol4_Q1_b.png')


n2=np.arange(1,5,0.01)

plt.figure()

x2=(1<n2)*(n2<2)+(4<n2)*(n2<5)
h2=(2<n2)*(n2<4)

n2=np.linspace(1,5,799)

plt.plot(n2*2,np.convolve(x2+0,h2+0)/100,linewidth=3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x(t)*h(t)$',fontsize=20)
plt.grid('on')
plt.savefig('PSol4_Q1_c.eps')
plt.savefig('PSol4_Q1_c.png')


plt.figure()

n2=np.linspace(-0.5,2.5,10000)

x2=(0<n2)*(n2<2)*np.sin(np.pi*n2)
h2=(0<n2)*(n2<2)

n2=np.linspace(0,6,19999)

plt.plot(n2,np.convolve(x2+0,h2+0)/max(np.convolve(x2+0,h2+0))*4/np.pi,linewidth=3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title(r'$x(t)*h(t)$',fontsize=20)
plt.grid('on')
plt.savefig('PSol4_Q1_d.eps')
plt.savefig('PSol4_Q1_d.png')
#b=2

for i in range(23):
#    plt.subplot(a,b,i+1,aspect=20)
    plt.figure()
    plt.stem(n,x(i-n),'b',markerfmt=' ',use_line_collection=True)
    plt.stem(n+0.3,h(n),'g',markerfmt=' ',use_line_collection=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('PSol4_Q1_'+str(i)+'.eps')
    plt.savefig('PSol4_Q1_'+str(i)+'.png')
#    plt.Axes.set_aspect(15:4)
#    plt.legend(['x[n]','h[n]']) 

#plt.subplot(a,b,2)
#plt.stem(n,x(-n),'b',markerfmt=' ')
#plt.stem(n+0.3,h(n),'g',markerfmt=' ')
#plt.legend(['x[n]','h[n]']) 
#
def x(n):
    return (n>=0)*(n<=4)

def h(n):
    return (n>=2)*(n<=7)+(n>=11)*(n<=16)

n=np.arange(-7,25,1)


#plt.subplot(a,b,3)
#plt.stem(n,x(-n),'b',markerfmt=' ')
#plt.stem(n+0.3,h(n),'g',markerfmt=' ')
#plt.legend(['x[n]','h[n]']) 
#
#plt.subplot(a,b,4)
#plt.stem(n,x(-n),'b',markerfmt=' ')
#plt.stem(n+0.3,h(n),'g',markerfmt=' ')
#plt.legend(['x[n]','h[n]']) 
#
#plt.subplot(a,b,5)
#plt.stem(n,x(-n),'b',markerfmt=' ')
#plt.stem(n+0.3,h(n),'g',markerfmt=' ')
#plt.legend(['x[n]','h[n]']) 

#file=open('tempo.txt','w')
#
#
#for i in range(23):
#    x='\n\\begin{subfigure}{0.24\\textwidth}\n'
#    y='\\includegraphics[width=\\wid]{PSol4_Q1_'+str(i)+'.eps}\n'
#    z='\\end{subfigure}\n%'
#    file.writelines(x+y+z)
#    
#file.close()

#x=[0,0,0,1,2,3,4,3,2,1,0,0,0]
#
#y=[0,0,0,1,1,1,0,0,0]
#
#y1=np.convolve(x,y)
#
#plt.plot(y1)