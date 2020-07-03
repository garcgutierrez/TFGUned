# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:10:05 2019

@author: gutierrez
"""

from pylab import *
import glob as gl
from scipy import integrate 
rc('font', family='serif', size='24')
rc('xtick', labelsize='large')
rc('ytick', labelsize='large')
rc('lines', linewidth=6, color='r')

files = gl.glob('older/u*_v2.npz')
bins_u = linspace(0,50,100)
mid_point = (bins_u[:-1]+bins_u[1:])/2
N_total = 0
N_total2 = 0
histo = zeros([31,len(bins_u)-1])
hist_total = zeros(len(bins_u)-1)
data = []
for file1 in files:
    try:
        datosv = load(file1.replace('u','v'))
        print(file1)
        datos = load(file1)
#        indices = argwhere(datos['levels']<60)
        uwind = sqrt(datos['U'][:,3,:,:]**2+datosv['U'][:,3,:,:]**2)
        uwind = uwind.transpose([1,0,2]).reshape(31, -1)
        hist_total += histogram(uwind[:],bins_u)[0] 
        for i in arange(31): 
            histo[i,:]+= histogram(uwind[i,:],bins_u)[0] 
        N_total += shape(uwind)[1]
        N_total2 += len(uwind)
#        data.append(uwind)
    except Exception as e:
        print(e)
lat = datos['lat']
histo = histo/N_total
histo *= (1/trapz(y=histo, x=mid_point)).reshape(31,1)

histo2 = hist_total/N_total2
histo2 *= (1/trapz(y=histo2, x=mid_point))



savez_compressed('pdf_wind', pdf = histo, u =bins_u)

from scipy.stats import rayleigh
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

def adjustar_Weibull(mid_pointa, histoa):
    a_ajustar_wei = lambda x,c,l,s: weibull_min.pdf(x, c, l, s)
    popt_wei, pcov = curve_fit(a_ajustar_wei, mid_pointa, histoa)
    ajustado_wey = a_ajustar_wei(mid_pointa, popt_wei[0], popt_wei[1], popt_wei[2])
    savez('adjWind.npz', parametros=popt_wei)
    return(ajustado_wey)
    
    

rms_v = [sqrt(trapz(y=histo[i,:]*mid_point**2,x=mid_point)) for i in arange(31)]
#a_ajustar = lambda x,l,s:rayleigh.pdf(x,l,s)
#
#
#popt, pcov = curve_fit(a_ajustar, mid_point, histo)
#
#ajustado_ray = a_ajustar(mid_point, popt[0], popt[1])
#ajustado_wey = adjustar_Weibull(mid_point, histo)


ajustado_wet_2d = histo.copy()
for i in arange(31):
    ajustado_wet_2d[i,:] = adjustar_Weibull(mid_point, histo[i,:])


histo_total = histo2
histo_total_adjust = adjustar_Weibull(mid_point, histo_total)

histo_total = histo[3, :]
histo_total_adjust = adjustar_Weibull(mid_point, histo_total)

style.use('seaborn-paper')
figure(5)
fig, ax1 = subplots()
#ax1.set_xlabel('h (km)')
#ax1.plot(hs/1000, 1/nons,'--r', label = '$N_{h}/N_s$')
#ax1.set_ylabel('$N_{h}/N_s, V_{h}/V_s$')
#ax1.set_ylim([0,10])
#ax2 = ax1.twinx()
#ax2.plot(np.nan, '--r', label = '$N_{h}/N_s$')
#ax2.plot(np.nan,'b', label = '$V_{h}/V_s$')
#ax1.plot(hs/1000, 1/vss,'b', label = '$V_{h}/V_s$')
#ax2.plot(hs/1000, 1/rs,'-.g', label = '$r$')
#ax2.set_ylim([0,1])


rcParams['figure.figsize'] = 15, 10
plot(mid_point, histo_total, label = 'Original Data')
plot(mid_point, histo_total_adjust, '--', label = 'Weibull')

ax1.set_xlabel('v (m/s)')
ax1.set_ylabel('PDF')
#legend()
#tight_layout()
#plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind2_pdf.eps', format='eps', dpi=1000)
int_lat = 10

ax2 = ax1.twinx()
#figure(2)
#rcParams['figure.figsize'] = 15, 10
ax2.plot(mid_point,integrate.cumtrapz(y=histo2, x=mid_point,initial=0), label = 'Original Data')
ax2.plot(mid_point,integrate.cumtrapz(y=histo_total_adjust, x=mid_point,initial=0),'--', label = 'Weibull distribution')
ax2.set_xlabel('v (m/s)')
ax2.set_ylabel('CDF')
legend(loc='center right')
title('30N')
tight_layout()

plt.savefig('C:/Users/gutierrez/OneDrive - UNED/DoctoradoULE/PolynomialChaos_Propeller/Imagenes/wind_30_total.eps', format='eps', dpi=1000)


histo_total = histo[7, :]
histo_total_adjust = adjustar_Weibull(mid_point, histo_total)
figure(7)
style.use('seaborn-paper')

fig, ax1 = subplots()
#ax1.set_xlabel('h (km)')
#ax1.plot(hs/1000, 1/nons,'--r', label = '$N_{h}/N_s$')
#ax1.set_ylabel('$N_{h}/N_s, V_{h}/V_s$')
#ax1.set_ylim([0,10])
#ax2 = ax1.twinx()
#ax2.plot(np.nan, '--r', label = '$N_{h}/N_s$')
#ax2.plot(np.nan,'b', label = '$V_{h}/V_s$')
#ax1.plot(hs/1000, 1/vss,'b', label = '$V_{h}/V_s$')
#ax2.plot(hs/1000, 1/rs,'-.g', label = '$r$')
#ax2.set_ylim([0,1])


rcParams['figure.figsize'] = 15, 10
plot(mid_point, histo_total, label = 'Original Data')
plot(mid_point, histo_total_adjust, '--', label = 'Weibull')

ax1.set_xlabel('v (m/s)')
ax1.set_ylabel('PDF')
#legend()
#tight_layout()
#plt.savefig('C:/Users/gutierrez/OneDrive - UNED/DoctoradoULE/PolynomialChaos_Propeller/Imagenes//wind_20_eps', format='eps', dpi=1000)
int_lat = 10

ax2 = ax1.twinx()
#figure(2)
#rcParams['figure.figsize'] = 15, 10
ax2.plot(mid_point,integrate.cumtrapz(y=histo2, x=mid_point,initial=0), label = 'Original Data')
ax2.plot(mid_point,integrate.cumtrapz(y=histo_total_adjust, x=mid_point,initial=0),'--', label = 'Weibull distribution')
ax2.set_xlabel('v (m/s)')
ax2.set_ylabel('CDF')
legend(loc='center right')
title('20N')
tight_layout()

plt.savefig('C:/Users/gutierrez/OneDrive - UNED/DoctoradoULE/PolynomialChaos_Propeller/Imagenes/wind_20_total.eps', format='eps', dpi=1000)


histo_total = histo[11, :]
histo_total_adjust = adjustar_Weibull(mid_point, histo_total)
style.use('seaborn-paper')
figure(6)
fig, ax1 = subplots()
#ax1.set_xlabel('h (km)')
#ax1.plot(hs/1000, 1/nons,'--r', label = '$N_{h}/N_s$')
#ax1.set_ylabel('$N_{h}/N_s, V_{h}/V_s$')
#ax1.set_ylim([0,10])
#ax2 = ax1.twinx()
#ax2.plot(np.nan, '--r', label = '$N_{h}/N_s$')
#ax2.plot(np.nan,'b', label = '$V_{h}/V_s$')
#ax1.plot(hs/1000, 1/vss,'b', label = '$V_{h}/V_s$')
#ax2.plot(hs/1000, 1/rs,'-.g', label = '$r$')
#ax2.set_ylim([0,1])


rcParams['figure.figsize'] = 15, 10
plot(mid_point, histo_total, label = 'Original Data')
plot(mid_point, histo_total_adjust, '--', label = 'Weibull')

ax1.set_xlabel('v (m/s)')
ax1.set_ylabel('PDF')
#legend()
#tight_layout()
#plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind2_pdf.eps', format='eps', dpi=1000)
int_lat = 10

ax2 = ax1.twinx()
#figure(2)
#rcParams['figure.figsize'] = 15, 10
ax2.plot(mid_point,integrate.cumtrapz(y=histo2, x=mid_point,initial=0), label = 'Original Data')
ax2.plot(mid_point,integrate.cumtrapz(y=histo_total_adjust, x=mid_point,initial=0),'--', label = 'Weibull distribution')
ax2.set_xlabel('v (m/s)')
ax2.set_ylabel('CDF')
title('10N')
legend(loc='center right')
tight_layout()

plt.savefig('C:/Users/gutierrez/OneDrive - UNED/DoctoradoULE/PolynomialChaos_Propeller/Imagenes/wind_10_total.eps', format='eps', dpi=1000)

histo_total = histo[15, :]
histo_total_adjust = adjustar_Weibull(mid_point, histo_total)
style.use('seaborn-paper')
figure(9)
fig, ax1 = subplots()
#ax1.set_xlabel('h (km)')
#ax1.plot(hs/1000, 1/nons,'--r', label = '$N_{h}/N_s$')
#ax1.set_ylabel('$N_{h}/N_s, V_{h}/V_s$')
#ax1.set_ylim([0,10])
#ax2 = ax1.twinx()
#ax2.plot(np.nan, '--r', label = '$N_{h}/N_s$')
#ax2.plot(np.nan,'b', label = '$V_{h}/V_s$')
#ax1.plot(hs/1000, 1/vss,'b', label = '$V_{h}/V_s$')
#ax2.plot(hs/1000, 1/rs,'-.g', label = '$r$')
#ax2.set_ylim([0,1])


rcParams['figure.figsize'] = 15, 10
plot(mid_point, histo_total, label = 'Original Data')
plot(mid_point, histo_total_adjust, '--', label = 'Weibull')

ax1.set_xlabel('v (m/s)')
ax1.set_ylabel('PDF')
#legend()
#tight_layout()
#plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind2_pdf.eps', format='eps', dpi=1000)
int_lat = 10

ax2 = ax1.twinx()
#figure(2)
#rcParams['figure.figsize'] = 15, 10
ax2.plot(mid_point,integrate.cumtrapz(y=histo2, x=mid_point,initial=0), label = 'Original Data')
ax2.plot(mid_point,integrate.cumtrapz(y=histo_total_adjust, x=mid_point,initial=0),'--', label = 'Weibull distribution')
ax2.set_xlabel('v (m/s)')
ax2.set_ylabel('CDF')
title('Equator')
legend(loc='center right')
tight_layout()

plt.savefig('C:/Users/gutierrez/OneDrive - UNED/DoctoradoULE/PolynomialChaos_Propeller/Imagenes/wind_00_total.eps', format='eps', dpi=1000)













# figure(1)
# rcParams['figure.figsize'] = 15, 10
# plot(mid_point, histo[int_lat:-int_lat, : ].T)
# #plot(mid_point, ajustado_ray)
# plot(mid_point, ajustado_wet_2d[int_lat:-int_lat, : ].T,'--')
# xlabel('v (m/s)')
# ylabel('pdf')

# tight_layout()
# int_hist = integrate.cumtrapz(y=histo, x=mid_point,initial=0)
# #figure(1)
# #pcolormesh(lat, mid_point, histo.T)
# #
# #figure(2)
# #pcolormesh(lat, mid_point, ajustado_wet_2d.T)






# figure(3)
# pcolormesh( mid_point, lat, int_hist)
# cbar = colorbar()
# cbar.set_label('CDF')
# CS = contour(mid_point, lat, int_hist,[0.66, 0.85,0.95, 0.99], colors='red', shading='gouraud')
# clabel(CS, inline=1, fontsize=20, fmt='%1.2f')
# ylabel(r'lat (deg)')
# xlabel(r'v (m/s)')
# xlim([0,45])
# tight_layout()
# plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind2_distributio_function.eps', format='eps', dpi=1000)
    

# #data = array(data)
# #ray = rayleigh.fit(data)
