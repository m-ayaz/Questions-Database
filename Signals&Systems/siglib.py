import numpy as np
import matplotlib.pyplot as plt

def plotter(x,y,_xlabel=None,_ylabel=None,_legend=(),_title=None,_axis=None,_grid='coarse',_newfigure=True):
    if _newfigure==True:
        plt.figure()
    plt.plot(x,y)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    if _grid=='coarse':
        plt.grid('on')
    elif _grid=='fine':
        plt.grid('on')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    if _axis=='equal':
        plt.axis('equal')
        
#    plt.axis(_axis)
    plt.legend(_legend)

plt.plot([1,2],[4,3])
xx=None
#plt.legend(xx)
plt.title(xx)
plt.xlabel(xx)
#plt.grid('on')
plt.axis()