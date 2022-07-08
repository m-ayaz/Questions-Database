# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:45:14 2020

@author: Glory
"""

import numpy as np

from scipy.signal import resample_poly

from scipy import signal

import commpy

import matplotlib.pyplot as plt

#def round
#def tester(inputSignal,alpha,gamma,Length,ASE_var):
#    
#    N=len(inputSignal[0])
#    
#    L_eff=(1-exp(-alpha*Length))/alpha
#    Power=np.abs(inputSignal[0])**2+np.abs(inputSignal[1])**2
#    output=inputSignal*exp(-alpha/2*Length*0+1j*gamma*L_eff*Power)
#    output=output+sqrt(ASE_var/4)*np.random.randn(2,N)+1j*sqrt(ASE_var/4)*np.random.randn(2,N)
#    return output

c=299792458

f1=c/1.55172e-6/1e12
f2=c/1.55252e-6/1e12
f3=c/1.55332e-6/1e12

f=[f1/1e12,f2/1e12,f3/1e12]

f=[round(f1,2),round(f2,2),round(f3,2)]

#f=np.round(f,3)

for i in range(3):
    for j in range(3):
        for k in range(3):
            if not(f[i]==f[j]==f[k]) and (f[i]<=f[j]) and (f[i]!=f[k]) and (f[j]!=f[k]):
#            if True:
                print('\\\&f_1='+str(f[i])+'THz\quad f_2='+str(f[j])+'THz\quad f_3='+str(f[k])+'THz\quad:\quad f_{\\text{FWM}}='+str(round(f[i]+f[j]-f[k],2))+'THz')
        