# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:10:05 2019

@author: gutierrez
"""

from pylab import *
import glob as gl
from scipy import integrate 
rc('font', family='serif', size='22')
rc('xtick', labelsize='large')
rc('ytick', labelsize='large')
rc('lines', linewidth=6, color='r')

files = gl.glob('older/u*_v2.npz')
bins_u = linspace(0,20,40)
bins_lat = linspace(0,360,200)
mid_point_x,mid_point_y = meshgrid((bins_u[:-1]+bins_u[1:])/2, (bins_u[:-1]+bins_u[1:])/2)
N_total = 0
N_total2 = 0
histo = zeros([31, 144, len(bins_u)-1,len(bins_u)-1])
histo_c = zeros([31, len(bins_u)-1,len(bins_u)-1])
rms_lat_lon = zeros([31, 144])
mean_lat_lon = zeros([31, 144])
hist_total = zeros([len(bins_u)-1,len(bins_u)-1])
data = []
for file1 in files:
    try:
        datosv = load(file1.replace('u','v'))
        print(file1)
        datos = load(file1)
#        indices = argwhere(datos['levels']<60)
        uwind = datos['U'][:,3,:,:]
        vwind = datosv['U'][:,3,:,:]
        uwind = uwind.transpose([1,2,0])
        vwind = vwind.transpose([1,2,0])
#        hist_total += histogram2d(uwind[:],bins_u, bins_lat)[0] 
        for i in arange(31):
            for j in arange(144):
                histo_c[i,:,:] += histogram2d(uwind[i,j,:],vwind[i,j,:],bins_u)[0]
                histo[i,j,:,:]+= histogram2d(uwind[i,j,:],vwind[i,j,:],bins_u)[0]
        N_total += shape(uwind)[1]
        N_total2 += len(uwind)
#        data.append(uwind)
    except Exception as e:
        print(e)
from scipy.integrate import simps
        
lat = datos['lat']
lon = datos['lon']
Int = simps(histo_c,(bins_u[:-1]+bins_u[1:])/2, axis=-1)
Int = simps(Int,(bins_u[:-1]+bins_u[1:])/2, axis=-1)
savez('joint_pdf', histo_c = histo_c/Int.reshape([-1,1,1]), histo = histo, lat = lat,lon = lon, bins_u = bins_u)        
Nlat = 15
figure()
style.use('seaborn-paper')

pcolormesh(mid_point_x,mid_point_y,histo_c[Nlat,:,:]/Int[Nlat],shading='gouraud')
colorbar()
xlabel(r'$u$(m/s)')
ylabel(r'$v$(m/s)')
title(r'Función de densidad de probabilidad conjunta. Latitud: {} grados'.format(lat[Nlat]))
lat = datos['lat']
lon = datos['lon']
plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/uv_pdf.eps', dpi=1000)

Nlat = 3
figure()
style.use('seaborn-paper')
Int = simps(histo_c,(bins_u[:-1]+bins_u[1:])/2, axis=-1)
Int = simps(Int,(bins_u[:-1]+bins_u[1:])/2, axis=-1)
pcolormesh(mid_point_x,mid_point_y, histo_c[Nlat,:,:]/Int[Nlat],shading='gouraud')
colorbar()
xlabel(r'$u$(m/s)')
ylabel(r'$v$(m/s)')
title(r'Función de densidad de probabilidad conjunta. Latitud: {} grados'.format(lat[Nlat]))
plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/uv_pdf2.eps', dpi=1000)

# for j in arange(144):
#     for i in arange(31): 
#         histo[i,j,:] = histo[i,j,:,:]/trapz(x = (mid_point_x*mid_point_y).reshape(-1), y =histo[i,j,:,:].reshape(-1))
#         # histo_c[i,j,:] = integrate.cumtrapz(initial = 0, y = histo[i,j,:], x = mid_point)
# #         rms_lat_lon[i, j] = sqrt(trapz(y=histo[i,j,:]*mid_point**2,x=mid_point))
# #         mean_lat_lon[i, j] = trapz(y=histo[i,j,:]*mid_point,x=mid_point)
# from matplotlib.image import NonUniformImage
# fig=figure() 
# ax = fig.add_subplot()
# im = NonUniformImage(ax, interpolation='bilinear')
# im.set_data(mid_point_x[0,:],mid_point_y[:,0],histo[3, 5,:,:])
# ax.images.append(im)
# show()
#pcolormesh(mid_point_x,mid_point_y,histo[3, 5,:,:])
# histo = histo/N_total

# histo *= (1/trapz(y=histo, x=mid_point)).reshape(31,144,1)

# #rms_v = [sqrt(trapz(y=histo[i,:]*mid_point**2,x=mid_point)) for i in arange(31)]

# from pylab import matplotlib as pl

# contour( lon, bins_u[:-1],histo_c[0,:,:].T,[0.66,0.95,], colors='red', shading='gouraud')
# contour( lon, bins_u[:-1],histo_c[15,:,:].T,[0.66,0.95,], colors='blue', shading='gouraud',linestyles='dashed')
# contour( lon, bins_u[:-1],histo_c[-1,:,:].T,[0.66,0.95,], colors='yellow', shading='gouraud',linestyles=':')
# ylim([0,40])

# #contour( lon, bins_u[:-1],histo_c[22,:,:].T,[0.66,0.95,], colors='black', shading='gouraud')

# figure(54)
# rcParams['figure.figsize'] = 15, 10
# pcolormesh(lon, lat, rms_lat_lon, shading='gouraud')
# cbar = colorbar()
# cbar.set_label('RMS (m/s)')
# CS =contour( lon, lat, rms_lat_lon,[8, 10,12], colors='black' ,linestyles='dashed')
# clabel(CS, inline=1, fontsize=16, fmt='%1.2f')
# ylabel(r'lat (deg)')
# xlabel(r'lon (deg)')
# #xlim([0,45])
# tight_layout()

# plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/rms_lat_lon.eps', format='eps', dpi=1000)


# figure(55)
# rcParams['figure.figsize'] = 15, 10
# pcolormesh(lon, lat, mean_lat_lon, shading='gouraud')
# cbar = colorbar()
# cbar.set_label('Mean (m/s)')
# CS =contour( lon, lat, mean_lat_lon,[8, 10,12], colors='black' ,linestyles='dashed')
# clabel(CS, inline=1, fontsize=16, fmt='%1.2f')
# ylabel(r'lat (deg)')
# xlabel(r'lon (deg)')
# #xlim([0,45])
# tight_layout()

# plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/mean_lat_lon.eps', format='eps', dpi=1000)





# histo2 = hist/N_total2
# histo2 *= (1/trapz(y=histo2, x=mid_point))



# #savez_compressed('pdf_wind_2d', pdf = histo, u =bins_u)

# from scipy.stats import rayleigh
# from scipy.stats import weibull_min
# from scipy.optimize import curve_fit

# def adjustar_Weibull(mid_pointa, histoa):
#     a_ajustar_wei = lambda x,c,l,s: weibull_min.pdf(x, c, l, s)
#     popt_wei, pcov = curve_fit(a_ajustar_wei, mid_pointa, histoa)
#     ajustado_wey = a_ajustar_wei(mid_pointa, popt_wei[0], popt_wei[1], popt_wei[2])
#     return(ajustado_wey)
    
    
    
# #a_ajustar = lambda x,l,s:rayleigh.pdf(x,l,s)
# #
# #
# #popt, pcov = curve_fit(a_ajustar, mid_point, histo)
# #
# #ajustado_ray = a_ajustar(mid_point, popt[0], popt[1])
# #ajustado_wey = adjustar_Weibull(mid_point, histo)


# ajustado_wet_2d = histo.copy()
# for i in arange(31):
#     ajustado_wet_2d[i,:] = adjustar_Weibull(mid_point, histo[i,:])


# histo_total = histo2
# histo_total_adjust = adjustar_Weibull(mid_point, histo_total)


# figure(5)
# rcParams['figure.figsize'] = 15, 10
# plot(mid_point, histo_total, label = 'Original Data')
# plot(mid_point, histo_total_adjust, '--', label = 'Weibull')

# xlabel('v (m/s)')
# ylabel('PDF')
# legend()
# tight_layout()
# plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind_pdf.eps', format='eps', dpi=1000)
# int_lat = 10


# figure(2)
# rcParams['figure.figsize'] = 15, 10
# plot(mid_point,integrate.cumtrapz(y=histo2, x=mid_point,initial=0), label = 'Original Data')
# plot(mid_point,integrate.cumtrapz(y=histo_total_adjust, x=mid_point,initial=0),'--', label = 'Weibull')
# xlabel('v (m/s)')
# ylabel('CDF')
# legend()
# tight_layout()

# plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind_cdf.eps', format='eps', dpi=1000)



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





# style.use('grayscale')
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
# plt.savefig('../../StratosphericPropulsion/PaperSerio/eps_foto/wind_distributio_function.eps', format='eps', dpi=1000)
    

# #data = array(data)
#ray = rayleigh.fit(data)
