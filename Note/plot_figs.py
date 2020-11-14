# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:37:46 2020

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
fontsize = 15

# =============================================================================
# %% plot exponential distribution
# =============================================================================

lamb = range(1,5)
x = np.linspace(0,4,1000)
for i in lamb:
    f = i*np.exp(-i*x)
    avg = 1/i
    avg_y = i*np.exp(-1)
    c = f'C{i-1}'
    plt.plot(x,f,c, label = fr'$\lambda = $ {i}, $\mu \pm \sigma$ = {avg:.2f} $\pm$ {avg:.2f}')
    plt.plot(avg,avg_y,c + '*', markersize = 15)
    plt.plot((0,2*avg),(avg_y,avg_y),c + '|-', markersize = 15)

fontsize = 15
plt.xlabel('$x$',fontsize = fontsize)
plt.ylabel('$p_X(x)$',fontsize = fontsize)
plt.title(r'Eksponentiel fordeling med rate $\lambda$', fontsize= fontsize)
plt.legend()
plt.savefig('exp_dist.png', dpi = 500)
plt.show()

# =============================================================================
# %% plot poisson distribution
# =============================================================================

k = range(1,5)
x = np.arange(0,10)
for i in k:
    f = np.exp(-i)*i**x/(factorial(x))
    c = f'C{i-1}'
    avg =  i
    avg_y = np.exp(-i)*i**i/(factorial(i))
    plt.plot(x,f,c + 'o--', label = fr'$\lambda = $ {i}, $\mu$ = {avg}')
    plt.plot(avg,avg_y,c + '*', markersize = 15)


plt.xlabel('$x$',fontsize = fontsize)
plt.ylabel('$p_X(x)$',fontsize = fontsize)
plt.title(r'Poisson fordeling med parameter $\lambda$', fontsize= fontsize)
plt.legend()
plt.savefig('poiss_dist.png', dpi = 500, bbox_inches='tight')
plt.show()

# =============================================================================
# %% poisson example
# =============================================================================

omkomne = np.hstack([np.repeat(deaths,nr) for deaths, nr in enumerate((109,65,22,3,1,0))])
lamb = omkomne.mean()
bins = np.arange(6)
vals, *_ = plt.hist(omkomne,bins = bins,density = True, rwidth = 0.8, label = 'Observationer (normaliseret)')
p = np.exp(-lamb)*lamb**bins/(factorial(bins))
for i in range(len(vals)):
    if i == 0:
        dx = 0.95
        dy = -0.03
    else:
        dx = 0.25
        dy = 0.02
    plt.text(i+dx,vals[i]+dy,vals[i], color = 'C0')
    plt.text(i+dx,vals[i]+dy + 0.03,f'{p[i]:.2f}', color = 'C1')

plt.xlabel('Antal', fontsize = fontsize)
plt.plot(bins+0.5,p, 'o-',label = f'Poisson fordeling, $\lambda$ = {lamb:.2f}')
labels = [str(i) for i in bins]; labels[-1] = '5+'
plt.xticks(bins + 0.5, labels = labels)
plt.legend()
plt.title('Omkomne Preussiske soldater fra\n hestespark pr. år pr. korps',
          fontsize = fontsize)
plt.savefig('poiss_example.png', dpi = 500, bbox_inches = 'tight')

# =============================================================================
# %% Normal distribution
# =============================================================================

mean = [-1,1]
std = [1,2]
x = np.linspace(-6,6,1000)
f = lambda x,mu,sig : (1/np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))


plt.subplot(211)
plt.title('Normalfordelingen med forskellige $\mu$ og $\sigma$', fontsize = fontsize)
sig = std[0]
for i in range(len(mean)):
    mu = mean[i]
    y = f(x,mu,sig)
    c = f'C{i}'
    plt.plot(x,y,c, label = fr'$\mu$ = {mu}, $\sigma$ = {sig}')
    plt.plot(mu,f(mu-sig,mu,sig),c + '*', markersize = 15)
    plt.plot((mu-sig,mu+sig),(f(mu-sig,mu,sig),f(mu+sig,mu,sig)),c + '|-', markersize = 15)
plt.ylim(0,0.45)
plt.xticks([])
plt.legend()
plt.subplot(212)
mu = mean[0]
for i in range(len(std)):
    sig = std[i]
    y = f(x,mu,sig)
    c = f'C{i}'
    plt.plot(x,y,c, label = fr'$\mu$ = {mu}, $\sigma$ = {sig}')
    plt.plot(mu,f(mu-sig,mu,sig),c + '*', markersize = 15)
    plt.plot((mu-sig,mu+sig),(f(mu-sig,mu,sig),f(mu+sig,mu,sig)),c + '|-', markersize = 15)
plt.ylim(0,0.45)
plt.legend()
plt.xlabel('x', fontsize = fontsize)
plt.savefig('normal.png', dpi = 500)
        

