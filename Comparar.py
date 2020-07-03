# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:34:02 2020

@author: gutierrez
"""

from pylab import *
from numpy import histogram, load
from scipy import stats

optimizado = load('control_opt.npz')
no_optimizado = load('control_det.npz')



figure()
rcParams['figure.figsize'] = 15, 10
rc('font', family='serif', size='28')
rc('xtick', labelsize='large')
rc('ytick', labelsize='large')
rc('lines', linewidth=4, color='r')
rcParams['figure.figsize'] = 15, 10
maxDo = optimizado['maxDist']
maxDno = no_optimizado['maxDist']
hist,x = histogram(maxDo,bins=100, density=True)
kernel = stats.gaussian_kde(maxDo)
# hist = stats.gaussian_kde(hist)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist)
plot(x, kernel(x), label = 'Optimizado')

kernel = stats.gaussian_kde(maxDno)
# hist = stats.gaussian_kde(hist)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist)
ylabel('PDF')
plot(x, kernel(x), '--',label = 'No optimizado')
title('Max Distancia')
xlabel('Max Distancia (m)')
tight_layout()
savefig('../latex/images/maxDistancia.eps', format='eps', dpi=120)




figure()
rcParams['figure.figsize'] = 15, 10
rc('font', family='serif', size='28')
rc('xtick', labelsize='large')
rc('ytick', labelsize='large')
rc('lines', linewidth=4, color='r')
rcParams['figure.figsize'] = 15, 10
maxcco = optimizado['costes']
maxccno = no_optimizado['costes']
hist,x = histogram(maxcco,bins=100, density=True)
kernel = stats.gaussian_kde(maxcco)
# hist = stats.gaussian_kde(hist)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist)
plot(x, kernel(x), label = 'Optimizado')
hist,x = histogram(maxccno,bins=100, density=True)
print('No opt {}'.format(mean(maxccno)))
print('Opt {}'.format(mean(maxcco)))
kernel = stats.gaussian_kde(maxccno)
# hist = stats.gaussian_kde(hist)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist)
plot(x, kernel(x), '--',label = 'No optimizado')
title('Coste')
ylabel('PDF')
xlabel('Coste')
legend()
tight_layout()
savefig('../latex/images/costeControl.eps', format='eps', dpi=120)


figure()
rcParams['figure.figsize'] = 15, 10
rc('font', family='serif', size='28')
rc('xtick', labelsize='large')
rc('ytick', labelsize='large')
rc('lines', linewidth=4, color='r')
rcParams['figure.figsize'] = 15, 10
maxffo = optimizado['final']
maxffno = no_optimizado['final']
hist,x = histogram(maxffo,bins=100, density=True)
kernel = stats.gaussian_kde(maxffo)
# hist = stats.gaussian_kde(hist)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist)
plot(x, kernel(x), label = 'Optimizado')

kernel = stats.gaussian_kde(maxffno)
# hist = stats.gaussian_kde(hist)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist)
plot(x, kernel(x), '--',label = 'No optimizado')

legend()
ylabel('PDF')
title('Pos Final')
xlabel('Pos Final (m)')
legend()
tight_layout()
savefig('../latex/images/posFinal.eps', format='eps', dpi=120)
