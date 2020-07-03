# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:07:25 2020

@author: gutierrez
"""

from pylab import *
from numpy import histogram, load
from scipy import stats
import glob as gl

files = gl.glob('Montecarlo2_sto_*pro_last*')
costes1 = []
for file in files:
    lul = load(file, allow_pickle = True)['results']
    maxD1 = [max(sqrt(sum((caso['pos']**2).reshape([-1,3]), axis=-1)))for caso in lul]
    costes1.extend([caso['costF'].item() for caso in lul])


costes1=array(costes1)
print(mean(costes1))
hist1,x1 = histogram(costes1,bins=100, density=True)
hist1 = hist1
kernel = stats.gaussian_kde(costes1)
x1 = 0.5*(x1[1:]+x1[:-1])
# plot(x1,hist1,'-^k', label = 'Optimizado')
plot(x1, kernel(x1),'-k', label = 'Optimizado')







# lul = load('Montecarlo_det2_3.npz', allow_pickle = True)['results']
# maxD2 = [max(sqrt(sum((caso['pos']**2).reshape([-1,3]), axis=-1)))for caso in lul]
files = gl.glob('Montecarlo_det_*pro_last*')
costes = []
for file in files:
    lul = load(file, allow_pickle = True)['results']
    maxD = [max(sqrt(sum((caso['pos']**2).reshape([-1,3]), axis=-1)))for caso in lul]
    costes.extend([caso['costF'].item() for caso in lul])
    

costes=array(costes)
print(mean(costes))
hist,x = histogram(costes,bins=100, density=True)
hist = hist
kernel = stats.gaussian_kde(costes)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist,'--or',label = 'Deterministic')
plot(x, kernel(x), '--r',label = 'Deterministico')
# semilogy()

# semilogy()
ylabel('PDF')
xlabel('Función de coste')
title('Coste')

legend()


figure()

files = gl.glob('Montecarlo2_sto_*pro_last*')
costes1 = []
for file in files:
    lul = load(file, allow_pickle = True)['results']
    maxD1 = [max(sqrt(sum((caso['pos']**2).reshape([-1,3]), axis=-1)))for caso in lul]
    costes1.extend([caso['costF'].item() for caso in lul])


costes1=array(maxD1)
hist1,x1 = histogram(costes1,bins=100, density=True)
# hist1 = hist1*(sum(costes1<1)/len(costes1))
kernel = stats.gaussian_kde(costes1)
x1 = 0.5*(x1[1:]+x1[:-1])
# plot(x1,hist1,'-^k', label = 'Optimizado')
plot(x1/1000, kernel(x1),'-k', label = 'Optimizado')







# lul = load('Montecarlo_det2_3.npz', allow_pickle = True)['results']
# maxD2 = [max(sqrt(sum((caso['pos']**2).reshape([-1,3]), axis=-1)))for caso in lul]
files = gl.glob('Montecarlo_det_*pro_last*')
costes = []
for file in files:
    lul = load(file, allow_pickle = True)['results']
    maxD = [max(sqrt(sum((caso['pos']**2).reshape([-1,3]), axis=-1)))for caso in lul]
    costes.extend([caso['costF'].item() for caso in lul])
    

costes=array(maxD)
hist,x = histogram(costes,bins=100, density=True)
# hist = hist*(sum(costes<1)/len(costes))
kernel = stats.gaussian_kde(costes)
x = 0.5*(x[1:]+x[:-1])
# plot(x,hist,'--or',label = 'Deterministic')
plot(x/1000, kernel(x), '--r',label = 'Deterministico')
# semilogy()

# semilogy()
ylabel('PDF')
xlabel('Máxima distancia (km)')
title('Máxima distancia')
legend()

#1.8 vs 3.4











# figure()
# maxD1 = array(maxD1)
# hist11,x11 = histogram(maxD1/1000,bins=20, density=True)
# kernel11 = stats.gaussian_kde(maxD1/1000)
# x11 = 0.5*(x11[1:]+x11[:-1])
# # plot(x,hist, label = 'Optimizado')
# plot(x11, kernel(x11), label = 'Optimizado')

# maxD2 = array(maxD1)
# hist22,x22 = histogram(maxD2/1000,bins=20, density=True)
# kernel22 = stats.gaussian_kde(maxD2/1000)
# x22 = 0.5*(x22[1:]+x22[:-1])
# # plot(x,hist, label = 'Optimizado')
# plot(x11, kernel(x11), label = 'Deterministico')