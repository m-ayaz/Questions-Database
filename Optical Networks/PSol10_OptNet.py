# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 01:39:26 2021

@author: Mostafa
"""

#import numpy as np
from numpy import prod,arange
#import math as m

def bernolli(n,k,p):
    
    if k==0:
        return (1-p)**n
    
    ind=arange(k)
    
    return prod([
            (n-ind)/(k-ind)*p*(1-p)**(n/k-1)
            ])

x=[]

for m in range(21,30):
    temp=0

    for i in range(21,m+1):
#        print(i)
        temp+=bernolli(m,i,0.6)
    
    x.append(temp)


y=0

for i in range(101,135+1):
#        print(i)
    y+=bernolli(135,i,0.6)

#x.append(temp)