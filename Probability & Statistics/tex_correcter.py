# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:35:27 2022

@author: M
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 17:42:28 2022

@author: M
"""

import os

files=os.listdir(os.getcwd())

for file in files:
    if not file[-4:]=='.tex':
        continue
    print(file)
    fff=open(file,'r')
#    os.system(f'xelatex {file}')