# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:55:13 2021

@author: Mostafa
"""

import numpy as np
from scipy.stats import norm

from numpy import sqrt

import matplotlib.pyplot as plt

plt.close('all')

cdf = norm.cdf

snr=np.logspace(0.8,4,10000)

p1=1-(1-2*cdf(-sqrt(snr/10)))**2
p2=1-(1-2*cdf(-sqrt(snr/10)))*(1-cdf(-sqrt(snr/10)))
p3=1-(1-cdf(-sqrt(snr/10)))**2

p1p=1-(1-2*cdf(-sqrt(snr/6.6)))**2
p2p=1-(1-2*cdf(-sqrt(snr/6.6)))*(1-cdf(-sqrt(snr/10)))
p3p=1-(1-cdf(-sqrt(snr/6.6)))**2

plt.semilogx(p1*0.25+p2*0.5+p3*0.25)
plt.semilogx(p1p*0.5+p2p*24/56+p3p*4/56,'--')

plt.legend(['Unshaped','Shaped'])

plt.xlabel('SNR (dB)')

plt.grid('on')

plt.xticks([1,1e1,1e2,1e3,1e4],[0,10,20,30,40])

plt.figure()

plt.semilogx(snr,p1)
plt.semilogx(snr,p2,'--')
plt.semilogx(snr,p3,':',linewidth=2)
plt.legend(['Inner (4 points)','Mid (8 points)','Corner (4 points)'])
plt.xlabel('SNR (dB)')

plt.grid('on')