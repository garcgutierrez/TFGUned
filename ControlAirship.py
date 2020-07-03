# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:46:03 2020

@author: gutierrez
"""

from airship import *
from pylab import *
from scipy.interpolate import interp1d
nt = 1800*5
xs = zeros(nt)
ys = zeros(nt)
zs = zeros(nt)

us = zeros(nt)
vs = zeros(nt)
ws = zeros(nt)
thrust = zeros(nt)
rudder = zeros(nt)
# elevator = zeros(nt)
motor = zeros(nt)
t = linspace(0, 1800,nt)
dt = diff(t)[0]

def problemaControl(zep, nt=80*10**2, dt=0.1):
    # nt = 80*10**3
    
    
    # zep = airship()
    # zep.set_u(0, 0, 10**1,10**1, 0, 0.1)
    zep.Wind = array([5, 1])
    for i in arange(nt):
        zep.RK4(dt)
        xs[i] = zep.x[3]
        ys[i] = zep.x[4]
        zs[i] = zep.x[5]
        us[i] = zep.y[3]
        vs[i] = zep.y[4]
        ws[i] = zep.y[5]
        thrust[i] = 2*zep.calcF(dt*i)
        motor[i] = abs(thrust[i] *us[i])
        rudder[i] =  zep.calcDeltar(dt*i)
        # elevator[i] =
    return({'x': xs,'y': ys,'z': zs,
            'u': us,'v': vs,'w': ws, 'motor':motor, 'thrust':thrust, 'rudder' : rudder})

def confControl(ControlPoints):
    print(ControlPoints)
    zep = airship()
    zep.Wind = array([5, 1])
    nuu = int(len(ControlPoints)/3)
    tpoints = ControlPoints[:nuu]*100
    epoints = ControlPoints[nuu:2*nuu]
    rpoints = ControlPoints[2*nuu:3*nuu]
    
    # Npoints = 10
    
    t_c = linspace(t[0], t[-1], nuu, endpoint=True)
    ft = interp1d(t_c, tpoints,kind='cubic', bounds_error=None, fill_value='extrapolate')
    et = interp1d(t_c, epoints, kind='cubic',bounds_error=None, fill_value='extrapolate')
    rt = interp1d(t_c, rpoints,kind='cubic', bounds_error=None, fill_value='extrapolate')
    zep.setControl(ft, et, rt)
    tray = problemaControl(zep, nt, dt)
    distF = sqrt(tray['x']**2+tray['y']**2+tray['z']**2).max()
    c = trapz(tray['motor'],t)
    if(distF<1000):
        coste = c
    else:
        coste = c+distF**2*10000
    print('Coste : {}    Dist: {}'.format(coste, distF))
    return({'coste':coste,'tray':tray,'t':t})
    

# tray = problemaControl(zep)
from scipy.optimize import minimize
confControlS = lambda x: confControl(x)['coste'] 
Npoints= 5
res = minimize(confControlS, r_[zeros(Npoints), zeros(Npoints), zeros(Npoints)], method='Nelder-Mead', tol=1e-2, options={'maxiter': 100,'disp': True})
# plot(tray['u'])


XX = res['x']
resultado = confControl(XX)
tray = resultado['tray']
figure()
t = resultado['t']
plot(t, tray['x'])
plot(t, tray['y'])
plot(t, tray['z'])

figure()
t = resultado['t']
plot(t, tray['u'], label='u')
plot(t, tray['v'], label = 'v')
plot(t, tray['w'], label = 'w')
legend()

figure()
plot(t, tray['motor'], label='Coste')
plot(t, tray['thrust'], label ='Thrust')
plot(t, tray['Rudder'], label ='Rudder')
legend()