# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:52:14 2020

@author: Mostafa
"""

import numpy as np

N=np.array([10,100,1000])
U=np.array([0.3,0.7,2]).T

for n in N:
    for u in U:
        print(max([1,2*n/(30+n*u)])*7500)