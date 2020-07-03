from pylab import *

import datetime
import math
import numpy
import random
import operator
from collections import Counter
import functools
#homegrown weibull probability function
#scipy.stats.exonweib.pdf gives different result from R dweibull function
def weibullpdf(data,scale,shape):
    return [(shape/scale)*((x/scale)**(shape-1))*math.exp(-1*(x/scale)**shape) for x in data]
    
#matrix-vector multiplier from http://code.activestate.com/recipes/121574-matrix-vector-multiplication/
#equivalent to %*% in R
def matmult4(m, v):
    return [functools.reduce(operator.add,map(operator.mul,r,v)) for r in m]



class TurbulentWind2():
    debug = False
    def calcCoeff(psd):
#        print(psd)
        ab = numpy.random.normal(0, sqrt(0.5*abs(psd)), 2)
        return(ab[0] + 1j*ab[1])
        
    def setCoeff(coeff, ind):
#        psdp = self.psd(omegas[ind],V)
        self.coeff[ind] = coeff
    
    def calcPsdp():
        return(self.psd(omegas[ind],V))
        
    def calcTurbulence(self, omegas, V):
        omegas = 2*pi*self.freq
        psds = array([self.psd(omega, V) for omega in omegas])
        self.coeff = array([TurbulentWind2.calcCoeff(psd) for psd in psds])
#        psds = array([self.psd(omega, V) for omega in omegas])
#        self.coeff = array([TurbulentWind.calcCoeff(psd) for psd in psds])
        self.turb = numpy.fft.irfft(self.coeff,norm='ortho')
        
    def calcTimeSeries(self, mean):
        self.calcTurbulence(2*pi*self.freq, mean)
        return( mean + self.turb) 
        
    def psd(self, omega, V):
        num = 12.3*self.Vhat/(log(10/self.z0 + 1)*log(self.h/self.z0 + 1))
        if(abs(V)>0.5):
            den = 1 + 192*(self.h*omega*log(10/self.z0 + 1)/(abs(V)*log(self.h/self.z0 + 1)))**(5/3)
        else:
            den = 1 + 192*(self.h*omega*log(10/self.z0 + 1)/(log(self.h/self.z0 + 1)))**(5/3)
        return(num/den)
    
    def __init__(self, h = 100, Vmean = 5, z0 = 0.1, n = 64*60+2):
        self.Vhat = Vmean # Mean wind speed at h = 10m
        self.h = h        # Height above the ground
        self.z0 = z0       # Surface roughness coefficient
        self.freq = numpy.fft.fftfreq(n, d = 1)[:int(n/2)] # 1 second spacing
#        self.coeff = zeros(shape(self.freq))
    
        if(TurbulentWind.debug):
            print('0 freq:{}    last:{} '.format(self.freq[0], self.freq[-1]))


class TurbulentWind():
    debug = False
#    counter = 0
    def calcCoeff(self, psd):
#        print(psd)
#        self.counter +=1
#        print('counter: {} '.format(self.counter))
        ab = numpy.random.normal(0, sqrt(0.5*abs(psd)), 2)
        return(ab[0] + 1j*ab[1])

    def calcTurbulence(self, omegas, V):
        psds = array([self.psd(omega, V) for omega in omegas])
        self.coeff = array([self.calcCoeff(psd) for psd in psds])
        # print('Calc Turb: {}'.format(abs(self.coeff[0])))
        self.turb = numpy.fft.irfft(self.coeff,norm='ortho')
    
    def calcTurbulenceCoeff(self, omegas, V, coeff, nint):
        psds = array([self.psd(omega, V) for omega in omegas])
        self.coeff = array([self.calcCoeff(psd) for psd in psds])
        self.coeff[:nint] = coeff
        # print('coeff ',abs(self.coeff[0]))
        self.turb = numpy.fft.irfft(self.coeff,norm='ortho')
        
    def calcTimeSeries(self, mean):
        self.calcTurbulence(2*pi*self.freq, mean)
        return( mean + self.turb) 
    
    def calcTimeSeriesCoeff(self, mean, coeff, nint):
        self.calcTurbulenceCoeff(2*pi*self.freq, mean, coeff, nint)
        return( mean + self.turb) 
        
    def psd(self, omega, V):
        num = 12.3*self.Vhat/(log(10/self.z0 + 1)*log(self.h/self.z0 + 1))
        if(abs(V)>0.5):
            den = 1 + 192*(self.h*omega*log(10/self.z0 + 1)/(abs(V)*log(self.h/self.z0 + 1)))**(5/3)
        else:
            den = 1 + 192*(self.h*omega*log(10/self.z0 + 1)/(log(self.h/self.z0 + 1)))**(5/3)
    
        return(num/den)
    
    def __init__(self, h = 100, Vmean = 5, z0 = 0.1, n = 64*60+2):
        self.Vhat = Vmean # Mean wind speed at h = 10m
        self.h = h        # Height above the ground
        self.z0 = z0       # Surface roughness coefficient
        self.freq = numpy.fft.fftfreq(n, d = 1)[:int(n/2)] # 1 second spacing
        if(TurbulentWind.debug):
            print('0 freq:{}    last:{} '.format(self.freq[0], self.freq[-1]))


class WindTurbSimulator():
    Nmin = 40
    hanner = numpy.hanning(2*Nmin*60)
    filter_array = r_[hanner[:Nmin*60], ones(3600-Nmin*60), hanner[Nmin*60:]]
    filter_array2 = r_[hanner[Nmin*60:], zeros(3600), hanner[:Nmin*60]]
    #filter_array = filter_array/(filter_array+filter_array2[::-1])
    def __init__(self, speed_series, mean_speed=9.0, max_speed=30.0, n_states=30, start_time=datetime.datetime(2012,1,1), hist_length=24):
        # WindHour = WindResource()
        nhours = len(speed_series)
        self.nhours = nhours
        # [WindHour.getNext() for i in arange(nhours)]
        turbWind = TurbulentWind(n = 60*(60 + WindTurbSimulator.Nmin) + 2) 
        self.turbWind = turbWind
        shape_t = shape(turbWind.calcTimeSeries(speed_series[0]))
        turbCollection = [turbWind.calcTimeSeries(speed) for speed in speed_series]
        windCollection = [ones(shape_t)*speed for speed in speed_series]
        self.WindHour = speed_series
        self.windSpeeds = zeros(nhours*3600)
        self.windSpeedsNT = zeros(nhours*3600)
        for i in arange(0, nhours-1):
            # print(shape(turbCollection[i]))
            # print(shape(WindTurbSimulator.filter_array))
            self.windSpeeds[i*3600:(i+1)*3600+WindTurbSimulator.Nmin*60]+= turbCollection[i]*WindTurbSimulator.filter_array
            self.windSpeedsNT[i*3600:(i+1)*3600+WindTurbSimulator.Nmin*60]+= windCollection[i]*WindTurbSimulator.filter_array

    def getPSD(self):
        omegas = 2*pi*self.turbWind.freq
        psds = [array([self.turbWind.psd(omega, V) for omega in omegas]) for V in self.WindHour]
        return(psds)
    def genNew(self):
        turbWind = TurbulentWind(n = 60*(60 + WindTurbSimulator.Nmin) + 2) 
        WindHour = self.WindHour
        nhours = self.nhours
        shape_t = shape(turbWind.calcTimeSeries(0))
        self.shape_t = shape_t
        turbCollection = [turbWind.calcTimeSeries(speed) for speed in WindHour]
#        windCollection = [ones(shape_t)*speed for speed in WindHour.speed_series]
        self.windSpeeds = zeros(nhours*3600)
        # self.windSpeedsNT = zeros(nhours*3600)
        for i in arange(0, nhours-1):
            # print(shape(turbCollection[i]))
            # print(shape(WindTurbSimulator.filter_array))
            self.windSpeeds[i*3600:(i+1)*3600+WindTurbSimulator.Nmin*60]+= turbCollection[i]*WindTurbSimulator.filter_array
            
    def genNewCoef(self, coeff, nint):
        turbWind = TurbulentWind(n = 60*(60 + WindTurbSimulator.Nmin) + 2) 
        WindHour = self.WindHour
        nhours = self.nhours
        shape_t = self.shape_t
        turbCollection = [turbWind.calcTimeSeriesCoeff(speed[0], coeff[speed[1]*nint:(speed[1]+1)*nint],nint) for speed in zip(WindHour, arange(nhours))]
#        windCollection = [ones(shape_t)*speed for speed in WindHour.speed_series]
        self.windSpeeds = zeros(nhours*3600)
        # self.windSpeedsNT = zeros(nhours*3600)
        for i in arange(0, nhours-1):
            # print(shape(turbCollection[i]))
            # print(shape(WindTurbSimulator.filter_array))
            self.windSpeeds[i*3600:(i+1)*3600+WindTurbSimulator.Nmin*60]+= turbCollection[i]*WindTurbSimulator.filter_array
#            self.windSpeedsNT[i*3600:(i+1)*3600+WindSimulator.Nmin*60]+= windCollection[i]*WindSimulator.filter_array
# class WindSimulator():
#     Nmin = 40
#     hanner = numpy.hanning(2*Nmin*60)
#     filter_array = r_[hanner[:Nmin*60], ones(3600-Nmin*60), hanner[Nmin*60:]]
#     filter_array2 = r_[hanner[Nmin*60:], zeros(3600), hanner[:Nmin*60]]
#     #filter_array = filter_array/(filter_array+filter_array2[::-1])
#     def __init__(self, nhours = 200, mean_speed=9.0, max_speed=30.0, n_states=30, start_time=datetime.datetime(2012,1,1), hist_length=24):
#         WindHour = WindResource()
#         [WindHour.getNext() for i in arange(nhours)]
#         turbWind = TurbulentWind(n = 60*(60 + WindSimulator.Nmin) + 2) 
#         self.WindHour = WindHour
#         shape_t = shape(turbWind.calcTimeSeries(self.WindHour.speed_series[0]))
#         turbCollection = [turbWind.calcTimeSeries(speed) for speed in WindHour.speed_series]
#         windCollection = [ones(shape_t)*speed for speed in WindHour.speed_series]
#         self.windSpeeds = zeros(nhours*3600)
#         self.windSpeedsNT = zeros(nhours*3600)
#         for i in arange(1, nhours-1):
#             print(shape(turbCollection[i]))
#             print(shape(WindSimulator.filter_array))
#             self.windSpeeds[i*3600:(i+1)*3600+WindSimulator.Nmin*60]+= turbCollection[i]*WindSimulator.filter_array
#             self.windSpeedsNT[i*3600:(i+1)*3600+WindSimulator.Nmin*60]+= windCollection[i]*WindSimulator.filter_array

#     def genNew():
#         turbWind = TurbulentWind2(n = 60*(60 + WindSimulator.Nmin) + 2) 
#         WindHour = self.WindHour
#         shape_t = shape(turbWind.calcTimeSeries(self.WindHour[0]))
#         turbCollection = [turbWind.calcTimeSeries(speed) for speed in WindHour.speed_series]
# #        windCollection = [ones(shape_t)*speed for speed in WindHour.speed_series]
#         self.windSpeeds = zeros(nhours*3600)
#         self.windSpeedsNT = zeros(nhours*3600)
#         for i in arange(1, nhours-1):
#             print(shape(turbCollection[i]))
#             print(shape(WindSimulator.filter_array))
#             self.windSpeeds[i*3600:(i+1)*3600+WindSimulator.Nmin*60]+= turbCollection[i]*WindSimulator.filter_array
# #            self.windSpeedsNT[i*3600:(i+1)*3600+WindSimulator.Nmin*60]+= windCollection[i]*WindSimulator.filter_array
     
class WindResource(object):
    def __init__(self, mean_speed=9.0, max_speed=30.0, n_states=30, start_time=datetime.datetime(2020,3,1), hist_length=24):
        self.mean_speed=mean_speed
        self.max_speed=max_speed
        self.n_states=n_states
        self.time=start_time
        self.hist_length=hist_length
        self.time_series = []
        #setup matrix
        n_rows=n_states                             
        n_columns=n_states          
        self.l_categ=float(max_speed)/float(n_states)    #position of each state
        
        #weibull parameters
        weib_shape=2.0
        weib_scale=2.0*float(mean_speed)/math.sqrt(math.pi);
        
        #Vector of wind speeds
#        self.WindSpeed = numpy.arange(self.l_categ/2.0,float(max_speed)+self.l_categ/2.0,self.l_categ)
        pdf_file = load('C:/Users/gutierrez/OneDrive - UNED/PythonThesis/noaa/pdf_wind.npz')
        self.WindSpeed = pdf_file['u']
        fdpWind = pdf_file['pdf'][15,:] ##OJO
        #distribution of probabilities, normalised
#        fdpWind = weibullpdf(self.WindSpeed,weib_scale,weib_shape)
#        fdpWind = fdpWind/sum(fdpWind)
        
        #decreasing function
        G = numpy.empty((n_rows,n_columns,))
        for x in range(n_rows):
            for y in range(n_columns):
                G[x][y] = 4**float(-abs(x-y))
            
        #Initial value of the P matrix
        P0 = numpy.diag(fdpWind)
        
        #Initital value of the p vector
        p0 = fdpWind
        
        #below comment from R source code:
        #"The iterative procedure should stop when reaching a predefined error.
        #However, for simplicity I have only constructed a for loop. To be improved!"
        P,p=P0,p0
        for i in range(50):
            r=matmult4(P,matmult4(G,p))
            r=r/sum(r)
            p=p+0.5*(p0-r)
            P=numpy.diag(p)
            
        N=numpy.diag([1.0/i for i in matmult4(G,p)])
        MTM=matmult4(N,matmult4(G,P))
        self.MTM = MTM
        self.MTMcum = numpy.cumsum(MTM,1)
        
        #initialise series
        self.state=0
        self.states_series=[]
        self.speed_series=[]
        self.power_series=[]
        self.randoms1=[]
        self.randoms2=[]
        
        #tick over to first value (decrement time accordingly)
        self.time=self.time+datetime.timedelta(hours=-1)
        self.getNext()
        
    #show current value without incrementing    
    def getCurrent(self):
        wind_counter = Counter([int(round(x,0)) for x in self.speed_series])
        return {'time': self.time,
                'data':{ 
                    'wind_speed': self.speed_series[-1],
                    'wind_speed_av' : sum(self.speed_series)/float(len(self.speed_series)),
                    'wind_hist': dict(wind_counter),
                    }
                }
        
    #increment time by one hour and return new value
    def getNext(self):
        self.randoms1.append(random.uniform(0,1))
        self.randoms2.append(random.uniform(0,1))
        self.state=next(j for j,v in enumerate(self.MTMcum[self.state]) if v > self.randoms1[-1])
        self.states_series.append(self.state)
        self.time_series.append(self.time)
        self.speed_series.append(self.WindSpeed[self.state]-0.5+(self.randoms2[-1]*int(self.l_categ)))
        self.time=self.time+datetime.timedelta(hours=1)
        return self.getCurrent()



from statsmodels.graphics.tsaplots import plot_acf

if(__name__=='__main__'):
    
    
    style.use('seaborn-paper')
    title('Simulación del viento')
    ylabel(r'Intensidad del viento (m/s)')
    xlabel(r'$y$ (km)')
    figure()
    
    lul = TurbulentWind()
    lul.calcTimeSeries(5)
    wind_base = WindResource()
    [wind_base.getNext() for i in arange(1000)]
    plot(wind_base.time_series, wind_base.speed_series,'-k')
    
    lul = TurbulentWind()
    lul.calcTimeSeries(5)
    wind_base = WindResource()
    [wind_base.getNext() for i in arange(1000)]
    plot(wind_base.time_series, wind_base.speed_series,'--r')
    
    lul = TurbulentWind()
    lul.calcTimeSeries(5)
    wind_base = WindResource()
    [wind_base.getNext() for i in arange(1000)]
    plot(wind_base.time_series, wind_base.speed_series, ':b')
    xlabel('Fecha')
    ylabel('Intensidad del viento (m/s)')
    # lul = WindSimulator()
    # plot(lul.windSpeeds)
    # plot(lul.windSpeedsNT)
    # show()
    
    
    # lul = WindTurbSimulator(5*sin(2*pi*arange(10)/4))
    # plot(lul.windSpeeds,'*')
    # plot(lul.windSpeedsNT)
    # show()

    # #lul = load('C:/Users/gutierrez/OneDrive - UNED/PythonThesis/noaa/pdf_wind.npz')