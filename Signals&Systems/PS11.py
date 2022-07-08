# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:09:05 2020

@author: Mostafa
"""

import numpy as np
import matplotlib.pyplot as plt

a=4
b=15
c=3

plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([2],[2],'kx',markersize=b,mfc='none')
plt.plot([-2],[2],'kx',markersize=b,mfc='none')
plt.plot([2],[-2],'kx',markersize=b,mfc='none')
plt.plot([-2],[-2],'kx',markersize=b,mfc='none')

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

#a=3

plt.xticks([-2,2],fontsize=15)
plt.yticks([-2,2],['-2j','2j'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q3_1.eps')













plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([2],[2],'ko',markersize=b,mfc='none')
plt.plot([-2],[2],'kx',markersize=b,mfc='none')
plt.plot([2],[-2],'ko',markersize=b,mfc='none')
plt.plot([-2],[-2],'kx',markersize=b,mfc='none')
plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.xticks([-2,2],fontsize=15)
plt.yticks([-2,2],['-2j','2j'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q3_2.eps')

















plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([2],[2],'kx',markersize=b,mfc='none')
plt.plot([-2],[2],'ko',markersize=b,mfc='none')
plt.plot([2],[-2],'kx',markersize=b,mfc='none')
plt.plot([-2],[-2],'ko',markersize=b,mfc='none')
plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.xticks([-2,2],fontsize=15)
plt.yticks([-2,2],['-2j','2j'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q3_3.eps')


















plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([2],[2],'ko',markersize=b,mfc='none')
plt.plot([-2],[2],'ko',markersize=b,mfc='none')
plt.plot([2],[-2],'ko',markersize=b,mfc='none')
plt.plot([-2],[-2],'ko',markersize=b,mfc='none')
plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.xticks([-2,2],fontsize=15)
plt.yticks([-2,2],['-2j','2j'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q3_4.eps')






























''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
a=3
b=10
c=3

plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([-.2,-.2],[2,-2],'ko',markersize=b,mfc='none')
plt.plot([-1],[0],'kx',markersize=b,mfc='none')
#plt.plot([2],[-2],'kx',markersize=b,mfc='none')
#plt.plot([-2],[-2],'kx',markersize=b,mfc='none')

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

#a=3

plt.xticks([],fontsize=15)
plt.yticks([-2,2],[r'$-j\omega_0$',r'$j\omega_0$'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q7_1.eps')













plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([-.2,-.2],[2,-2],'kx',markersize=b,mfc='none')
#plt.plot([-1],[0],'kx',markersize=b,mfc='none')
#plt.plot([2],[-2],'kx',markersize=b,mfc='none')
#plt.plot([-2],[-2],'kx',markersize=b,mfc='none')

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

#a=3

plt.xticks([],fontsize=15)
plt.yticks([-2,2],[r'$-j\omega_0$',r'$j\omega_0$'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q7_2.eps')










a=4
b=10
c=3






plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([-1],[0],'ko',markersize=b,mfc='none')
plt.plot([-3],[0],'kx',markersize=b,mfc='none')
#plt.plot([2],[-2],'kx',markersize=b,mfc='none')
#plt.plot([-2],[-2],'ko',markersize=b,mfc='none')
plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.xticks([],fontsize=15)
plt.yticks([],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q7_3.eps')


















plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([2],[0],'ko',markersize=b,mfc='none')
plt.plot([-2],[0],'kx',markersize=b,mfc='none')
plt.plot([1],[0],'ko',markersize=b,mfc='none')
plt.plot([-1],[0],'kx',markersize=b,mfc='none')
plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.xticks([-2,2,-1,1],['-b','b','-a','a'],fontsize=15)
plt.yticks([],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q7_4.eps')








a=3
b=10
c=3

plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
plt.plot([-.2,-.2],[2,-2],'ko',markersize=b,mfc='none')
plt.plot([-1],[0],'kx',markersize=b,mfc='none')
plt.plot([1],[0],'ko',markersize=b,mfc='none')
#plt.plot([2],[-2],'kx',markersize=b,mfc='none')
#plt.plot([-2],[-2],'kx',markersize=b,mfc='none')

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

#a=3

plt.xticks([],fontsize=15)
plt.yticks([-2,2],[r'$-j\omega_0$',r'$j\omega_0$'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q7_5.eps')





a=3
b=10
c=3

plt.figure()
#fig.patch.set_visible(False)
#ax.axis('off')
#plt.plot([-.2,-.2],[2,-2],'ko',markersize=b,mfc='none')
plt.plot([-1],[0],'ko',markersize=b,mfc='none')
plt.plot([1],[0],'ko',markersize=b,mfc='none')
#plt.plot([2],[-2],'kx',markersize=b,mfc='none')
#plt.plot([-2],[-2],'kx',markersize=b,mfc='none')

plt.xlabel(r'$\Re$',fontsize=20)
plt.ylabel(r'$\Im$',fontsize=20)

plt.plot([0,0],[-a,a],'k',linewidth=c)
plt.plot([-a,a],[0,0],'k',linewidth=c)

#a=3

plt.xticks([],fontsize=15)
plt.yticks([],[r'$-j\omega_0$',r'$j\omega_0$'],fontsize=15)
plt.axis([-a,a,-a,a])
plt.axis('equal')
#plt.ylim([-1.4,1.4])
plt.savefig('PS11_Q7_6.eps')