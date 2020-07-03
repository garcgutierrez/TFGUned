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

files = gl.glob('air*_v2.npz')
bins_u = linspace(200, 250, 100)
mid_point = (bins_u[:-1]+bins_u[1:])/2
N_total = 0
N_total2 = 0
nLat = 39
histo = zeros([nLat,len(bins_u)-1])
hist_total = zeros(len(bins_u)-1)
data = []

for file1 in files:
    try:
        # datosv = load(file1.replace('u','v'))
        print(file1)
        datos = load(file1)
#        indices = argwhere(datos['levels']<60)
        uwind = datos['U'][:,3,:,:]
        uwind = uwind.transpose([1,0,2]).reshape(nLat, -1)
        hist_total += histogram(uwind[:],bins_u)[0] 
        for i in arange(nLat): 
            histo[i,:]+= histogram(uwind[i,:],bins_u)[0] 
        N_total += shape(uwind)[1]
        N_total2 += len(uwind)
#        data.append(uwind)
    except Exception as e:
        print(e)
lat = datos['lat']
histo = histo/N_total
histo *= (1/trapz(y=histo, x=mid_point)).reshape(nLat,1)

histo2 = hist_total/N_total2
histo2 *= (1/trapz(y=histo2, x=mid_point))



savez_compressed('pdf_air', pdf = histo, u =bins_u)

from scipy.stats import rayleigh
from scipy.stats import maxwell
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

def adjustar_Weibull(mid_pointa, histoa):
    a_ajustar_wei = lambda x,c,l,s: weibull_min.pdf(x, c, l, s)
    popt_wei, pcov = curve_fit(a_ajustar_wei, mid_pointa, histoa)
    print(popt_wei, pcov)
    ajustado_wey = a_ajustar_wei(mid_pointa, popt_wei[0], popt_wei[1], popt_wei[2])
    savez('adjTemp.npz', parametros=popt_wei)
    return(ajustado_wey)
    
def adjustar_Ray(mid_pointa, histoa):
    a_ajustar_ray = lambda x,l,s: rayleigh.pdf(x, l, s)
    popt_wei, pcov = curve_fit(a_ajustar_ray, mid_pointa, histoa)
    print(popt_wei, pcov)
    ajustado_ray = a_ajustar_ray(mid_pointa, popt_wei[0], popt_wei[1])
    return(ajustado_ray)

def adjustar_Maxwell(mid_pointa, histoa):
    a_ajustar_ray = lambda x,l,s: maxwell.pdf(x, l, s)
    popt_wei, pcov = curve_fit(a_ajustar_ray, mid_pointa, histoa)
    print(popt_wei, pcov)
    ajustado_ray = a_ajustar_ray(mid_pointa, popt_wei[0], popt_wei[1])
    return(ajustado_ray)

rms_v = [sqrt(trapz(y=histo[i,:]*mid_point**2,x=mid_point)) for i in arange(nLat)]
#a_ajustar = lambda x,l,s:rayleigh.pdf(x,l,s)
#
#
#popt, pcov = curve_fit(a_ajustar, mid_point, histo)
#
#ajustado_ray = a_ajustar(mid_point, popt[0], popt[1])
#ajustado_wey = adjustar_Weibull(mid_point, histo)


# ajustado_wet_2d = histo.copy()
# for i in arange(nLat):
#     ajustado_wet_2d[i,:] = adjustar_Weibull(mid_point, histo[i,:])


histo_total = histo[2, :]
histo_total_adjust = adjustar_Weibull(mid_point-mid_point[0], histo_total)
# histo_total_adjust = adjustar_Maxwell(mid_point-mid_point[0], histo_total)

figure(5)
fig, ax1 = subplots()


rcParams['figure.figsize'] = 15, 10
plot(mid_point, histo_total, label = 'Original Data')
plot(mid_point, histo_total_adjust, '--', label = 'Weibull')

ax1.set_xlabel('T (K)')
ax1.set_ylabel('PDF')
# legend()
tight_layout()
# plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/te22_pdf.eps', format='eps', dpi=1000)
int_lat = 10

ax2 = ax1.twinx()
#figure(2)
#rcParams['figure.figsize'] = 15, 10
ax2.plot(mid_point,integrate.cumtrapz(y=histo_total, x=mid_point,initial=0),'-s', label = 'Original Data')
ax2.plot(mid_point,integrate.cumtrapz(y=histo_total_adjust, x=mid_point,initial=0),'--s', label = 'Weibull')
ax2.set_xlabel('T (K)')
ax2.set_ylabel('CDF')
legend(loc='center right')
tight_layout()

plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/te2_total.eps', format='eps', dpi=1000)



figure(1)
rcParams['figure.figsize'] = 15, 10
plot(mid_point, histo[int_lat:-int_lat, : ].T)
#plot(mid_point, ajustado_ray)
plot(mid_point, ajustado_wet_2d[int_lat:-int_lat, : ].T,'--')
xlabel('v (m/s)')
ylabel('pdf')

tight_layout()
int_hist = integrate.cumtrapz(y=histo, x=mid_point,initial=0)
#figure(1)
#pcolormesh(lat, mid_point, histo.T)
#
#figure(2)
#pcolormesh(lat, mid_point, ajustado_wet_2d.T)






figure(3)
pcolormesh( mid_point, lat, int_hist)
cbar = colorbar()
cbar.set_label('CDF')
CS = contour(mid_point, lat, int_hist,[0.66, 0.85,0.95, 0.99], colors='red', shading='gouraud')
clabel(CS, inline=1, fontsize=20, fmt='%1.2f')
ylabel(r'lat (deg)')
xlabel(r'v (m/s)')
xlim([0,45])
tight_layout()
plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind2_distributio_function.eps', format='eps', dpi=1000)
    

#data = array(data)
#ray = rayleigh.fit(data)
