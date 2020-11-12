# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:37:46 2020

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# =============================================================================
# plot exponential distribution
# =============================================================================

lamb = range(1,5)
x = np.linspace(0,4,1000)
for i in lamb:
    f = i*np.exp(-i*x)
    avg = 1/i
    avg_y = i*np.exp(-1)
    c = f'C{i-1}'
    plt.plot(x,f,c, label = fr'$\lambda = $ {i}, gns. = {avg:.2f}')
    plt.plot(avg,avg_y,c + 'x', markersize = 15)

fontsize = 15
plt.xlabel('$x$',fontsize = fontsize)
plt.ylabel('$p(x)$',fontsize = fontsize)
plt.title(r'Eksponentiel fordeling med rate $\lambda$', fontsize= fontsize)
plt.legend()
plt.savefig('exp_dist.png', dpi = 500)
plt.show()
# =============================================================================
# plot poisson distribution
# =============================================================================

k = range(1,5)
x = np.arange(0,10)
for i in k:
    f = np.exp(-i)*i**x/(factorial(x))
    c = f'C{i-1}'
    avg =  i
    avg_y = np.exp(-i)*i**i/(factorial(i))
    plt.plot(x,f,c + 'o--', label = fr'$k = $ {i}, gns. = {avg}')
    plt.plot(avg,avg_y,c + 'x', markersize = 15)

fontsize = 15
plt.xlabel('$x$',fontsize = fontsize)
plt.ylabel('$p(x)$',fontsize = fontsize)
plt.title(r'Poisson fordeling med parameter $k$', fontsize= fontsize)
plt.legend()
plt.savefig('poiss_dist.png', dpi = 500)
plt.show()

# =============================================================================
# %% genereate numbers for non-parametric distribution
# =============================================================================

np.random.seed(0)
lamb = 3
x = np.random.exponential(lamb, size = 20).round(1)
for i in x:
    print(str(i), end = ',\\hspace{1pt} ')
print()
nr  = [sum(np.floor(x/2).astype('int') + 1 == i) for i in range(1,6)]
for i in nr:
    print(str(i), end = ',\\hspace{1pt} ')
print()
for i in range(5):
    print('\\frac{%d}{40}' % nr[i])

bins = [0,2,4,8,10]
plt.hist(x,bins = bins, density = True, histtype = 'step')
plt.xlabel('x',fontsize = fontsize)
plt.ylabel('p(x)',fontsize = fontsize)
plt.title('Ikke parametrisk pdf', fontsize = fontsize)
plt.savefig('ikke_para_pdf.png', dpi = 500)
plt.show()



p = np.array(nr)/40

def F(x):
    if x <= 0:
        return(0)
    elif x > 10:
        return(F(10))
    else:
        idx  = int(np.ceil(x/2)) - 1
        return((x - 2*idx)*p[idx] + F(2*idx))
    
x_emp = np.linspace(0,11,1000)
F_emp = np.array([F(i) for i in x_emp])

u = 0.7
x_u = x_emp[(F_emp <= u).argmin()]

plt.plot(x_emp,F_emp)
plt.plot((0,x_u),(u,u),'r--')
plt.plot((x_u,x_u),(0,u),'r--')
plt.text(0.1, u + 0.05, 'u', fontsize = fontsize)
plt.text(x_u + 0.4,0.05, '$F^{-1}(u)$', fontsize = fontsize)
plt.xlabel('x',fontsize = fontsize)
plt.ylabel('F(x)',fontsize = fontsize)
plt.title('Ikke parametrisk cdf', fontsize = fontsize)
plt.savefig('ikke_para_cdf.png', dpi = 500)
plt.show()


