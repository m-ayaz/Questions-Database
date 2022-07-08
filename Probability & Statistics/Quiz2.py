# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 01:18:11 2020

@author: Mostafa
"""

import matplotlib.pyplot as plt
import numpy as np

plt.plot([0,0],[0,1],'k')
plt.plot([0,1],[0,0],'k')
plt.plot([0,1],[1,1],'k')
plt.plot([1,1],[0,1],'k')
plt.plot([0,1],[0,1],'r')
plt.tight_layout()
plt.axis('equal')
plt.axis('off')