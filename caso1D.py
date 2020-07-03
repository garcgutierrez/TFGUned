# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:53:06 2019

tuto_chaospy 2

@author: gutierrez
"""
from scipy.integrate import ode
from scipy.integrate import RK23
from scipy import interpolate
import chaospy as cp
import numpy as np
from controlOpt import *
from chaospy_tuto3 import *
from wind import *

import time
# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator



def solver2(A1,A2,plotT = False):
    return(calcControl(A1,A2, plotT))

def solver3(A1, A2, A3, reinic, x1,x2,u0):
    tic()
    try:
        turb.genNewCoef(r_[A1,A1,A1,A2,A2,A2,A3,A3,A3], 3)
    # wind = turb.windSpeeds[3600:-3600:Nint]
        wind = turb.windSpeeds[3600:-3600:Nint]
        rs = calcControl(t, wind, False, reinic, x1, x2, u0)
    except:
        rs = {'x1':x1,'x2':x2,'u':u0}
    toc()
    return(rs)


def f(t, y, arg1):
    return [-arg1*y]

def solver (*node):
    # node : tuple of the uncertain stochastic parameters
    r = ode(f).set_integrator('dopri5')
#    print(node)
    r.set_initial_value(node[0], 0)
    r.set_f_params(node[1])
    t1 = 10
    dt=0.1
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
    results = r.y
    return [results,t1]

rcParams['figure.figsize'] = 15, 10
rc('font', family='serif', size='28')
rc('xtick', labelsize='large')
rc('ytick', labelsize='large')
# rc('lines', linewidth=4, color='r')
# rc('lines', markersize=7, color='r')

for inp in arange(3):
    Nh = 3
    Nint = 1
    t = arange(0, (Nh-2)*3600)[::Nint]
    turb = WindTurbSimulator(5*(1.0+cos(2*pi*(arange(Nh))/4)))
    wind = turb.windSpeeds[3600:-3600:Nint]
    
    # plot(t,wind)
    turb.genNew()
    wind = turb.windSpeeds[3600:-3600:Nint]
    plot(t,wind,'--', linewidth=3.0)
    
    
    lul = turb.getPSD()
    N_t = 1
    psds2 =  array(lul)[:,:3]
    rand_cf = array([[numpy.random.normal(0, sqrt(0.5*abs(psd)), 2) for psd in psds] for psds in psds2])
    rand_cf = (rand_cf[:,:,0]+1j*rand_cf[:,:,1]).reshape(-1)
    print('outer' ,abs(rand_cf[1]))
    turb.genNewCoef(rand_cf, 3)
    # wind = turb.windSpeeds[3600:-3600:Nint]
    wind = turb.windSpeeds[3600:-3600:Nint]
    plot(t,wind,'--', linewidth=3.0)


windNT = turb.windSpeedsNT[3600:-3600:Nint]
plot(t,windNT,'r', label = 'Componente deterministica', linewidth=4.0)
legend()
ylabel('Intensidad del viento (m/s)')
xlabel('Tiempo (s)')
tight_layout()
savefig('../latex/images/windExample.eps', format='eps', dpi=120)

rs = calcControl(t, windNT, False)

# plot(rs['t'], rs['x1'], label = 'Vel (m/s)')
# plot(rs['t'], rs['u'], label = 'Control')
# plot(t, windNT, label = 'Wind')

# plot(rs['t'], rs['x2'], label = 'Pos')
# legend()

figure()
rcParams['figure.figsize'] = 15, 10
rc('font', family='serif', size='28')
rc('xtick', labelsize='large')
rc('ytick', labelsize='large')
rc('lines', linewidth=4, color='r')
rcParams['figure.figsize'] = 15, 10
fig, ax1 = subplots()
ax1.set_xlabel('t (s)')
# ax1.plot(hs/1000, 1/nons,'--r', label = '$N_{h}/N_s$')
# ax1.plot(rs['t'], rs['x1'], '-b', label = 'V')
ax1.plot(rs['t'], rs['u'], '--k',label = 'Control')
ax1.plot(t, windNT,'-.g', label = 'Wind ')

ax1.set_ylabel('u (N), wind (m/s)')
grid()
# ax1.set_ylim([0,10])
ax2 = ax1.twinx()
ax2.plot(rs['t'], array(rs['x2'].value)/1000, ':r', label = 'Pos')
ax2.plot(rs['t'], rs['x1'], '-b', label = 'V')
ax2.plot(np.nan, '--k',label = 'Control')
ax2.plot(np.nan,'-.g', label = 'Wind')
# ax2.set_ylim([0,1])
#ax2.set_xlim([0.1,20])
ax2.set_ylabel('x (km), v (m/s)')
# legend(loc='center left')
legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=4, fancybox=True, shadow=True)

tight_layout()
savefig('../latex/images/optimalControl.eps', format='eps', dpi=120)


turb.genNew()
wind = turb.windSpeeds[3600:-3600:Nint]
fwind = interpolate.interp1d(t, wind, fill_value='extrapolate')
fcontrol = interpolate.interp1d(t, rs['u'], fill_value='extrapolate')
r = ode(dirigible1D).set_integrator('vode')
r.set_initial_value([0, 0, 0], 0).set_f_params(fcontrol, fwind)
dt = 0.1
sol = []
while r.successful() and r.t < t[-2]:
    sol.append(r.y)
    r.integrate(r.t + dt)
    print(r.t+dt)
print(r.y)
print('Integration done')

time2 = arange(0,t[-2],dt)
v = r_[array(sol)[:,0]]
x = r_[array(sol)[:,1]]
wind = fwind(t)

rcParams['figure.figsize'] = 15, 10
fig, ax1 = subplots()
ax1.set_xlabel('t (s)')
ax1.plot(rs['t'], rs['u'], '--k',label = 'Control')
ax1.plot(t, wind,'-.g', label = 'Wind ')
grid()
ax1.set_ylabel('u (N), wind (m/s)')
ax2 = ax1.twinx()
ax2.plot(time2, x[:-1]/1000, ':r', label = 'Pos')
ax2.plot(time2, v[:-1] , '-b', label = 'V')
ax2.plot(np.nan, '--k',label = 'Control')
ax2.plot(np.nan,'-.g', label = 'Wind')
ax2.set_ylabel('x (km), v (m/s)')
legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=4, fancybox=True, shadow=True)
grid()
tight_layout()
savefig('../latex/images/notSoOptimalControl.eps', format='eps', dpi=120)


Cost_opt = rs['coste'][-1]


dist_a = cp.Normal(0 , sqrt(0.5*abs(psds2.reshape(-1)[1])))
dist_b = cp.Normal(0 , sqrt(0.5*abs(psds2.reshape(-1)[4])))
dist_c = cp.Normal(0 , sqrt(0.5*abs(psds2.reshape(-1)[7])))

dist = cp. J(dist_a, dist_b)

samples = dist.sample(size =10)

P = cp.orth_ttr(3 , dist)
nodes, weights = cp.generate_quadrature(3, dist)

windS = []

fwinS = []
for coeff in nodes.reshape(2,-1).T:
    print(coeff)
    turb.genNewCoef(r_[0,coeff[0],0,coeff[1], 0,0], 2)
    wind = turb.windSpeeds[3600:-3600:Nint]
    windS.append(wind)
    fwinS.append(interpolate.interp1d(arange(3600), wind, kind='linear', fill_value='extrapolate'))

fcontrol = interpolate.interp1d(t, rs['u'], copy=False, kind='cubic',fill_value='extrapolate', assume_sorted=True)

tu = r_[geomspace(1, 1000, 20)-1,arange(1000, 3600, 200), 3600]
plot(tu, fcontrol(tu),'*')
plot(t, rs['u'])
u_0 = fcontrol(tu)
from numba import jit

from scipy.integrate import odeint
@jit
def funcToOptimize(u, fwind_series, weights):
    tic()
    fcontrol = interpolate.interp1d(tu, u, copy=False,kind='cubic', fill_value='extrapolate',assume_sorted=True)
    final = empty(len(fwind_series))
    maxDist = empty(len(fwind_series))
    costes = empty(len(fwind_series))
    for ii in arange(len(fwind_series)):
        fwind = fwind_series[ii]

        sol = odeint(dirigible1Dt, [0,0,0], arange(0,3600,0.1), args=(fcontrol, fwind))
        maxX = abs(array(sol)[:, 1]).max()
        coste_total = array(sol)[-1, 2]
        # finalX = abs(array(sol))[-1, 1]
        #final.append(finalX)
        costes[ii] = coste_total
        maxDist[ii] = maxX
        # print('Partial_cost: {}'.format(coste_total))
    mean_coste = sum(costes*weights)
    toc()
    media_dist = sum(maxDist*weights)
    if(media_dist**2>5000**2):
        mean_coste+=(media_dist**2-5000**2)*100000
        
            
    # mean_coste = sum(costes*weights)
    print('Medio: {}   media_dist: {}'.format(mean_coste, media_dist))
    return(mean_coste)

from scipy import optimize

result = optimize.minimize(funcToOptimize, u_0, args=(fwinS, weights.reshape(-1)), method = 'Nelder-Mead', options = {'disp':True, 'maxiter':1000})
uo = result['x']
# savez('result_opt.npz', result
plot(t, rs['u'])
plot(tu, uo)
plot(tu, u_0)
# figure()
# plot(arange(0,t[-2],dt), r_[array(sol)[:,2]],'-.b')
# plot(rs['t'], rs['coste'])
# plot(rs['t'], cost_m)
# show()




final = []
costes = []
maxDist = []
for i in arange(5000):
    turb.genNew()
    wind = turb.windSpeeds[3600:-3600:Nint]
    fwind = interpolate.interp1d(t, wind, kind='cubic', fill_value='extrapolate')
    fcontrol = interpolate.interp1d(tu, uo, kind='cubic', fill_value='extrapolate')
    r = ode(dirigible1D).set_integrator('vode')
    r.set_initial_value([0, 0, 0], 0).set_f_params(fcontrol, fwind)
    dt = 0.01
    sol = []
    while r.successful() and r.t < t[-1]:
        sol.append(r.y)
        r.integrate(r.t + dt)
        # print(r.t+dt, r.integrate(r.t+dt))
    print(r.y)
    print('Integration done')
    coste_opt = rs['coste']
    array(sol)[:,0]
    maxX = abs(array(sol)[:, 1]).max()
    coste_total = array(sol)[-1, 2]
    finalX = abs(array(sol))[-1, 1]
    final.append(finalX)
    costes.append(coste_total)
    maxDist.append(maxX)

savez('control_opt.npz',maxDist = array(maxDist), costes = array(costes)/Cost_opt, final = array(final))



# sol_pet = solver2(0,0, True)
# t=sol_pet['t']
# plot(t, mean2)
