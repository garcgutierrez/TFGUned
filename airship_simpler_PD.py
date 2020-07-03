from pylab import *
from scipy import linalg as ll
from scipy import interpolate
m = 5.6*10**3
delta = 7.4*10**6
rho = 0.089
# xcg =5
# zcg = 15
xcg = 0
zcg = 0
# xp = 4
xp = 0
yp = 0
zp = 0
zp = 0
Ix = 5*10**7
Iy = 2.9*10**8
Iz = 2.9*10**8
Ixz = -6*10**4
from wind import *
eta = pi/6
g = 9.81
k1 = 0.17
k2 = 0.83
k3 = 0.52

Cl1 = 2.4 * 10**3
Cm1 = 7.7*10**3
Cn1 = 7.7*10**3
Cm2 = 7.7*10**3
Cn2 = 7.7*10**3
Cm3 = 7.7*10**3
Cn3 = 7.7*10**3
Cm4 = 7.7*10**3
Cn4 = 7.7*10**3
Cx1 = 657
Cx2 = 657
Cy1 = 657
Cz1 = 657
Cy2 = 657
Cz2 = 657
Cy3 = 657
Cz3 = 657
Cy4 = 657
Cz4 = 657
from numba import jit
rd = rho*delta

@jit(nopython=True)
def norm(a):
    return(sqrt(a[0]**2+a[1]**2+a[2]**2+a[3]**2))

@jit(nopython=True)
def calcydot( A, N3, N, G, B, u, ydot):
        # print('A:{}'.format(A))
        # print('N:{}'.format(sel))
        # ydot[:] = linalg.solve(A,N3 + 2*N +G + B@u)[:,0]
        ydot[:] = linalg.solve(A,N3+G+B@u)[:,0]
        ydot[1] = 0
        ydot[-1] = 0
        # print('udot:{}'.format(self.B@u, self))
        # print('B:{}', self.B)
        # ydot = linalg.solve(self.A, self.N3)
        # ydot = linalg.solve(self.A,self.B@u)
        # ydot.shape = (-1)
        # print('ydot:{}',ydot)
        # print(u)
        # print(self.B)
        # print(self.B@u)
#        print('ydot {}    {}'.format(shape(u), shape(self.B)))
        return(ydot)

@jit(nopython=True)
def set_R(Rp1, omega1,omega2,omega3):

        Rp1[:,:] = array([
            [0, omega3,-omega2,omega1],
            [-omega3, 0, omega1, omega2],
            [omega2, -omega1, 0, omega3],
            [-omega1, -omega2,-omega3,0]
            ])
        Rp1[:,:] *= 0.5
        return(Rp1)

@jit(nopython=True)
def resolve_eq(Rp1, y, qdot):
        return(Rp1@y)

@jit(nopython=True)
def set_N(N, u, v, w , p, q, r, ):
        N[2] =-(Iy + rd*k3 - Ix)*p*q-Ixz*q*r - m*xcg*(u*r-w*p) 
        N[3] =-(m + rd*k2)*(w*q-v*r) - m*zcg*p*r + m*xcg*(q**2+r**2)
        N[4] =(m + rd*k2)*w*p - (m + rd*k1)*u*r - m*xcg*p*q - m*zcg*q*r 
        return(N)
        # print(self.N)
@jit(nopython=True)
def set_G( G, m, g, xcg, zcg, theta, phi, Bg ):
    G[:] = array([
        [0],
        [0],
        [xcg*m*g*cos(theta)*sin(phi)],
         [(Bg-m*g)*sin(theta)],
         [-(Bg-m*g)*cos(theta)*sin(phi)],
         [0]
    ])
    return(G)

# @jit(nopython=True)
def set_B( B, eta, Q, cm4,u):
        B[:,:] = array([
                 [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2*Q*Cy4/10, 0],
            [0, 0, 0, 0, 0, 2*Q*Cz4/10],
            [ 2*cos(eta), 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        return(B)


@jit(nopython=True)
def quaternion_multiply(quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

@jit(nopython=True)
def set_N3(N3, u, v, w, alpha, betha, Q, omega1,omega2,omega3):
        # sbetha = sin(betha)
        # abetha = abs(betha)
        # aalpha = abs(alpha)
        # print('alppha {}'.format(alpha))
        # s2betha = sin(2*betha)
        # s2alpha = sin(2*alpha)
        # c_2betha = cos(betha/2)
        # c_2alpha = cos(alpha/2)
        # calpha = cos(alpha)
        # cbetha = cos(betha)
        N3[2] = -100*Q*omega3
        N3[3] = -2*Cx1*u
        N3[4] = -2*Cy1*v
        return(N3)
    
@jit(nopython=True)
def fromQuaternionToEuler(a,b,c,d):

        phi = arctan2(2*(a*b+c*d), a**2-b**2-c**2+d**2)
        theta = -arcsin(2*(b*d-a*c))
        theta = arctan2(tan(theta),1)
        xi = arctan2(2*(a*d+b*c),(a**2+b**2-c**2-d**2))
        return([phi, theta, xi])


class airship():
    def __init__(self):
        A = zeros([3, 3])
        B = zeros([3, 3])
        self.x = zeros(6)
        self.q = zeros(7)
        self.y = zeros(6)
        self.q[0]=1
        self.M = zeros([3, 3])
        self.C = zeros([3, 3])
        self.D = zeros([3, 3])
        self.tau = zeros([3, 3])
        self.J = zeros([3, 3])
        self.V = zeros([3, 3])
        self.R = zeros([8, 8])
        self.Rp1 = zeros([4,4])
        self.ucontrol = zeros([6,1])
        self.set_A(Ix, Ixz, m, zcg, Iy, rho, xcg, Iz, delta, k1, k2, k3)
        self.Wind = [5,0]
        self.t = 0
        self.Bg = m*g
        self.calcF = lambda t: 0
        self.calcDeltar = lambda t: 0
        self.calcDeltae = lambda t: 0
        self.N3 =zeros([6,1])
        self.N =zeros([6,1])
        self.B = zeros([6,6])
        self.G=zeros([6,1])
        self.creepy = zeros(4)
        self.creepy2 = array([1,-1,-1,-1])
        self.k1q = zeros(7)
        self.k2q = zeros(7)
        self.k3q = zeros(7)
        self.k4q = zeros(7)
        self.k1y = zeros(6)
        self.k2y = zeros(6)
        self.k3y = zeros(6)
        self.k4y = zeros(6)
        self.Cost = 0
        self.creepyQDOT = zeros(7)
        
    def calcRHS(self, q, y, Uwind,Vwind,Wwind, qdot, ydot):
#        print('x:  {}'.format(x))
#        print('y:  {}'.format(y))
        q[:4] = q[:4]/norm(q[:4])
        x = fromQuaternionToEuler(q[0],
            q[1], q[2], q[3])
        # print(x)
        phi = x[0]
        theta = 0#x[1]
        xi = x[2]
        
        # xp =x[3]
        # yp = x[4]
        # z = x[5]
        p = 0
        qq = 0
        r = y[2]
        u = y[3]#-self.Wind[0]
        v = y[4]#-self.Wind[1]
        w = 0
        self.N = set_N(self.N,  u, v, w, p, qq, r,)
        self.Rp1 = set_R(self.Rp1, r, p, qq)
        alpha = arctan2(w-Wwind, u-Uwind)
        betha = arctan2((v-Vwind)*cos(alpha), u-Uwind)
        V = sqrt((u-Uwind)**2 + (v-Vwind)**2 + (w-Wwind)**2)
        Q = rho*V**2/2
        # print('Alpha:{}   beta {}'.format(alpha, betha))
        # print('Wind: ',V,u,v,w)
        self.N3 = set_N3(self.N3, u-Uwind, v-Vwind, w-Wwind, alpha, betha, Q, p,qq,r)
        # self.G = set_G(self.G, m, g, xcg, zcg, theta, phi, rho*g)
        self.B = set_B(self.B, eta, Q, Cm4, u)
        self.creepyQDOT[:4] = q[:4]
        self.creepyQDOT[4:] = y[3:]
        qdot = self.calcxdot(self.creepyQDOT, qdot)
        ydot = calcydot( self.A, self.N3, self.N, self.G, self.B, self.ucontrol, ydot)
        self.Cost = abs(self.ucontrol[0]*Q)+abs(self.ucontrol[-2]*Q)
        # print(qdot)
        # qdot[]
        return qdot, ydot
    
    
    
    def RK4(self, dt):
        self.calcControl()
        x = self.x
        y = self.y
        q = self.q
        
        a,b,c,d = q[:4]/norm(q[:4])
        # Rot = array(
        #     [
        #         [a,b,c,d],
        #         [-b,a,-d,c],
        #         [-c,d,a,-b],
        #         [-d,-c,b,a]
        #     ])
        q1 = r_[a,-b,-c,-d]
        q2 = r_[0, self.Wind[0],self.Wind[1],0]
        q3 = r_[a,b,c,d]
        qu = quaternion_multiply(q1, q2)
        qu = quaternion_multiply(qu, q3)
        # print(a**2+b**2+c**2+d**2)
        # vwin = qu[1:]
        # print(qu)
        # vwin = r_[vwin[1], vwin[2]]
        # print(vwin)
        # self.y[3] -= vwin[0]
        # self.y[4] -= vwin[1]
        # self.y[5] -= vwin[2]
        Uwind,Vwind,Wwind = qu[1:]
        try:
            self.k1q, self.k1y = self.calcRHS(q, y,Uwind,Vwind,Wwind, self.k1q, self.k1y)
            # print('k1x: {} k1y:{}'.format(k1q, q))
            # self.k2q, self.k2y = self.calcRHS(q+0.5*self.k1q*dt, y + 0.5*self.k1y*dt,Uwind,Vwind,Wwind, self.k2q, self.k2y)
            # self.k3q, self.k3y = self.calcRHS(q+0.5*self.k2q*dt, y + 0.5*self.k2y*dt, Uwind,Vwind,Wwind, self.k3q, self.k3y)
            # self.k4q, self.k4y = self.calcRHS(q+self.k3q*dt, y + self.k3y*dt, Uwind,Vwind,Wwind, self.k4q,self.k4y)
            # self.q = q + 1/6*dt*(self.k1q + 2*self.k2q + 2*self.k3q + self.k4q)
            self.q = q +self.k1q*dt
            # print('y: {}'.format(self.y))
            # self.y = y + 1/6*dt*(self.k1y + 2*self.k2y + 2*self.k3y + self.k4y)
            self.y = y +self.k1y*dt
            # print('after: {}'.format(self.y))
            self.q[:4] =  self.q[:4]/norm(self.q[:4])
            self.x[:3] = fromQuaternionToEuler(self.q[0], self.q[1], self.q[2], self.q[3])
            self.x[3:] = self.q[4:]
            self.x[1] = 0
            self.x[-1] = 0
        except Exception as e:
            print(e)
            self.y[:] = ones(6)*9999999
            self.x[:] = ones(6)*9999999

        self.t +=dt
        
    def calcxdot(self, y, qdot):
        # qdot = zeros(7)
        qdot[:4] = resolve_eq(self.Rp1, y[:4], qdot[:4])
        # qdot[:4] = linalg.solve(self.Rp1, y[:4])
        a,b,c,d = self.q[:4]
        self.creepy[1:] = y[4:]
        qdot[4:] = quaternion_multiply(quaternion_multiply(self.q[:4], self.creepy), 
                                 self.q[:4]*self.creepy2)[1:]
        

        
        return(qdot)
        
    
#
    def setJ(phi):
        self.J[0,0] = cos(phi)
        self.J[0,1] = -sin(phi)
        self.J[1,0] = sin(phi)
        self.J[1,1] = cos(phi)
        self.J[2,2] = 1
    
    
    def set_A(self, Ix, Ixz, m, zcg, Iy, rho, xcg,Iz,delta, k1, k2, k3):
        self.A = array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, Iz+rho*delta*k3, 0, m*xcg, 0],
            [0, 0, 0, m+rho*delta*k1, 0, 0],
            [0, 0, m*xcg, 0, m+rho*delta*k2, 0],
            [0, 0, 0, 0, 0,1]
        ])

    def setControl(self, Ffunc, rFunc):
        self.F = Ffunc
        self.deltar = rFunc
        # self.calcDeltae = dFunc

    def calcControl(self):
        F = self.F
        deltar = self.deltar
        # deltae = self.calcDeltae(self.t)
        self.ucontrol[0] = F
        self.ucontrol[1] = F
        self.ucontrol[2] = 0
        self.ucontrol[3] = 0
        self.ucontrol[4] = deltar
        self.ucontrol[5] = deltar/2
        # self.ucontrol = array([F, F, 0, 0, deltar, deltae])
        # self.ucontrol.shape = (6,1)
        
    def set_u(self, mul, mur, Fl, Fr, deltar, deltae):
        self.ucontrol = array([Fl*cos(mul), Fr*cos(mur), Fl*sin(mul), Fl*sin(mur), deltar, deltae])
        self.ucontrol.shape = (6,1)
        
    # def set_N3(self, u, v, w, alpha, betha, Q):
    #     # sbetha = sin(betha)
    #     abetha = abs(betha)
    #     aalpha = abs(alpha)
    #     # print('alppha {}'.format(alpha))
    #     s2betha = sin(2*betha)
    #     s2alpha = sin(2*alpha)
    #     c_2betha = cos(betha/2)
    #     c_2alpha = cos(alpha/2)
    #     calpha = cos(alpha)
    #     cbetha = cos(betha)
    #     # print('Alpha {}  betha{}'.format(alpha, betha))
    #     self.N3[0] = -(Q*Cl1*sin(betha)*sin(abetha))
    #     self.N3[1] = -(Q*(Cm1*c_2alpha*s2alpha + Cm2*s2alpha + Cm3*sin(alpha)*sin(aalpha)))
    #     self.N3[2] = -(Q*(Cn1*c_2betha*s2betha + Cn2*s2betha + Cn3*sin(betha)*sin(abetha)))
    #     self.N3[3] = -(Q*(Cx1*calpha**2*cbetha*abs(cos(betha)) + Cx2*s2betha*sin(alpha/2)))
    #     self.N3[4] = -(Q*(Cy1*c_2betha*s2betha + Cy2*s2betha + Cy3*sin(betha)*sin(abetha)))
    #     self.N3[5] = -(Q*(Cz1*c_2alpha*s2alpha + Cz2*s2alpha + Cz3*sin(alpha)*sin(aalpha)))# self.N3 *= Q
    #     # print(self.N3)
    #     # self.N3.shape = (6,-1)

    def set_S1(self, theta, phi):
        self.S1 = array([
            [1, 0, -sin(theta)],
            [0, cos(phi), -sin(phi)*cos(theta)],
            [0, -sin(phi), cos(phi)*cos(theta)]
        ])

    def set_S2(self, theta, phi):
        self.S2 = array([
            [cos(theta)*cos(xi), cos(theta)*sin(xi), -sin(theta)],
            [sin(theta)*cos(xi)*sin(phi) - sin(xi)*cos(phi), sin(theta)*cos(xi)*sin(phi) + cos(xi)*cos(phi), cos(theta)*sin(phi)],
            [sin(theta)*cos(xi)*cos(phi) + sin(phi)*sin(xi), sin(theta)*cos(xi)*cos(phi) - cos(xi)*sin(phi), cos(theta)*cos(phi)]
        ])

    def set_S(self):
        self.set_S1()
        self.set_S2()
        self.S = r_[c_[self.S1, zeros(3,3)] ,c_[zeros(3,3), self.S2]]

    def set_Sdot(self, r, q):
        self.Sdot =array([ [0, -r, q],
                     [r, 0, -p],
                     [-q, p, 0]])
                    
#    def set_Ba(self, Qinf, eta):
#        self.Ba = array([[cos()]])
#

def calcControlPID(zep, k):
    r_[zep.x[3]
    zep.x[4]
    zep.x[5]
    [zep.x[0]
    zep.x[1]
    zep.x[2]]
    zep.setControl
    

def runSim(tcontrol = lambda t:0, deltatcontrol =lambda t:0, fwind = lambda x:0, fwind2 = lambda x:0): 
    zep = airship()
    zep.Wind = array([-5,0.1]) 
    controlCost = 0                   
    for i in arange(nt):
            # if(i%100==0):
            #     print(i)
            zep.Wind[0] = -fwind(i*dt)
            zep.Wind[1] = -fwind2(i*dt)
            # zep.setControl(tcontrol , deltatcontrol)
            zep.RK4(dt)
            xs[i] = zep.x[3]
            ys[i] = zep.x[4]
            zs[i] = zep.x[5]
            phi[i] = zep.x[0]
            theta[i] = zep.x[1]
            xi[i] = zep.x[2]
            p[i] = zep.y[0]
            q[i] = zep.y[1]
            r[i] = zep.y[2]
            us[i] = zep.y[3]
            vs[i] = zep.y[4]
            ws[i] = zep.y[5]
            controlCost += abs(zep.Cost*dt)
    print('Control: {}     {}'.format(controlCost,max(sqrt(xs**2+ys**2+zs**2))))
    # return(max(sqrt(xs**2+ys**2+zs**2))/100+controlCost/10000000)
    return(max(sqrt(xs**2+ys**2+zs**2))/1000+sqrt(xs**2+ys**2+zs**2)[-1]/1000+controlCost/10**8)
      
def controlProblem(u, fwind, fwind2):
    npoints = int(len(u)/3)
    T = u[:npoints]
    deltat = u[npoints:]
    # deltae = u[2*npoints:]
    t_c = arange(nt)*dt
    t = linspace(0,(nt-1)*dt, npoints)
    tt = linspace(0,(nt-1)*dt, len(u)-int(len(u)/3))
    tcontrol = interpolate.interp1d(t, T*10000, kind = 'cubic', fill_value='extrapolate')
    deltatcontrol = interpolate.interp1d(tt, deltat, kind = 'cubic', fill_value='extrapolate')
    # deltaecontrol = interpolate.interp1d(t, T, kind = 'cubic')
    fCost = runSim( tcontrol, deltatcontrol, fwind, fwind2)
    print(fCost)
    return(fCost)

if(__name__=='__main__'):
    from scipy import optimize
    nt = 1800
    xs = zeros(nt)
    ys = zeros(nt)
    zs = zeros(nt)
    dt = 2
    us = zeros(nt)
    vs = zeros(nt)
    ws = zeros(nt)
    p = zeros(nt)
    q = zeros(nt)
    r = zeros(nt)
    phi = zeros(nt)
    theta = zeros(nt)
    xi = zeros(nt)
    
    values = []
    # calcMean fake          8.6
    # for i in arange(5000):
    #     if(i%100==0):
    #         print(i)
    #     lul.state = argwhere(lul.WindSpeed<vmean)[-1][0]
    #     lul.getNext()
    #     values.append(lul.speed_series[-1])
    
    for inp in arange(3):
        Nh = 3
        Nint = 1
        t = arange(0, (Nh-2)*3600)[::Nint]
        lul = WindResource()
        vmean = 9
        lul.state = argwhere(lul.WindSpeed<vmean)[-1][0]
        lul.getNext()
        turb = WindTurbSimulator((vmean*(cos(2*pi*(arange(Nh))/4)+1)+(1-(cos(2*pi*(arange(Nh))/4)+1))*8.6)[::-1])#lul.speed_series[-1])[::-1])
        wind = turb.windSpeeds[3600:-3600:Nint][::-1]
        
        # plot(t,wind)
        turb.genNew()
        wind = turb.windSpeeds[3600:-3600:Nint][::-1]
        plot(t,wind,'--', linewidth=3.0)
        
        
        lul2 = turb.getPSD()
        N_t = 1
        psds2 =  array(lul2)[:,:3]
        rand_cf = array([[numpy.random.normal(0, sqrt(0.5*abs(psd)), 2) for psd in psds] for psds in psds2])
        rand_cf = (rand_cf[:,:,0]+1j*rand_cf[:,:,1]).reshape(-1)
        print('outer' ,abs(rand_cf[1]))
        turb.genNewCoef(rand_cf, 3)
        # wind = turb.windSpeeds[3600:-3600:Nint]
        wind = turb.windSpeeds[3600:-3600:Nint][::-1]
        plot(t,wind,'--', linewidth=3.0)
    
    turb2 = WindTurbSimulator([0.1,0.1,0.1])#lul.speed_series[-1])[::-1])
    wind2 = turb2.windSpeedsNT[3600:-3600:Nint][::-1]
    windNT = turb.windSpeedsNT[3600:-3600:Nint][::-1]
    # zep.set_u(0, 0, 10**1,10**1, 0, 0.1)
    
    fwind = interpolate.interp1d(t, windNT)
    fwind2 = interpolate.interp1d(t, wind2)
    # zep = runSim(zep)
    # costF = controlProblem( zeros([30]), zep)
    con = optimize.LinearConstraint(eye(40), r_[zeros([10]), -1*ones([30])],  r_[1*ones([10]), 1*ones([30])])
    # costF = controlProblem( ones([30]))
    result = optimize.minimize(controlProblem, zeros([40]), args = (fwind, fwind2),constraints = con,method = 'SLSQP', options = {'disp':True, 'maxiter':10000, 'eps':0.01})
    figure()   
    plot(dt*arange(nt), xs,label='x')
    plot(dt*arange(nt),ys, label='y')
    plot(dt*arange(nt),zs, label = 'z')
    legend()
    figure()   
    plot(dt*arange(nt),phi,label='phi')
    plot(dt*arange(nt),theta, label='theta')
    plot(dt*arange(nt),xi, label = 'xi')
    legend()
    
    figure()
    plot(dt*arange(nt),us,label='u')
    plot(dt*arange(nt),vs,label='v')
    plot(dt*arange(nt),ws,label='w')
    legend()
    
    figure()
    plot(dt*arange(nt),p, label='p')
    plot(dt*arange(nt),q, label='q')
    plot(dt*arange(nt),r, label='r')
    legend()
    # yscale('log')
    
    
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    style.use('seaborn-paper')
    title('Trayectorial dirigible')
    figure()
    plot(xs/1000,ys/1000,'-k')
    xlabel(r'$x$ (km)')
    ylabel(r'$y$ (km)')
    results2 ={'pos':r_[xs,ys,zs], 'vel':r_[us,vs,ws], 'vel2':r_[p,q,r], 'angles':r_[phi, theta, xi], 'results' : result, 'fwind':fwind, 't' : dt*arange(nt)}
    savez_compressed('Rdet', results = results2)
    
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # style.use('seaborn-paper')
    # title('Control')
    # figure()
    # plot(xs/1000,ys/1000,'-k', label = 'Caso deterministico')
    # xlabel(r'$x$ (km)')
    # ylabel(r'$y$ (km)')
    
    # # windNT = turb.windSpeeds[3600:-3600:Nint]
    # # # zep.set_u(0, 0, 10**1,10**1, 0, 0.1)
    
    # # fwind = interpolate.interp1d(t, windNT)
    
    # # costF = controlProblem( result['x'], fwind)
    
    # # # figure()   
    # # # plot(dt*arange(nt), xs,label='x')
    # # # plot(dt*arange(nt),ys, label='y')
    # # # plot(dt*arange(nt),zs, label = 'z')
    # # # legend()
    # # # figure()   
    # # # plot(dt*arange(nt),phi,label='phi')
    # # # plot(dt*arange(nt),theta, label='theta')
    # # # plot(dt*arange(nt),xi, label = 'xi')
    # # # legend()
    
    # # # figure()
    # # # plot(dt*arange(nt),us,label='u')
    # # # plot(dt*arange(nt),vs,label='v')
    # # # plot(dt*arange(nt),ws,label='w')
    # # # legend()
    
    # # # figure()
    # # # plot(dt*arange(nt),p, label='p')
    # # # plot(dt*arange(nt),q, label='q')
    # # # plot(dt*arange(nt),r, label='r')
    # # # legend()
    # # # yscale('log')
    
    
    # # import matplotlib.pyplot as plt
    # # from mpl_toolkits.mplot3d import Axes3D
    # # style.use('seaborn-paper')
    # # title('Control')
    # # figure()
    # # plot(xs/1000,ys/1000,'-k')
    # # xlabel(r'$x$ (km)')
    # # ylabel(r'$y$ (km)')
    # lul.state = argwhere(lul.WindSpeed<vmean)[-1][0]
    # lul.getNext()
    # turb = WindTurbSimulator((vmean*(cos(2*pi*(arange(Nh))/4)+1)+(1-(cos(2*pi*(arange(Nh))/4)+1))*8.6)[::-1])#lul.speed_series[-1])[::-1])
    # wind = turb.windSpeeds[3600:-3600:Nint][::-1]
    
    # windNT = wind
    # zep.set_u(0, 0, 10**1,10**1, 0, 0.1)
    
    
    
    
    #####
    #   Montecarlo
    #####
    results_mon = []
    for i in arange(1000):
        print(i)
        turb2.genNew()
        turb.genNew()
        wind2 = turb2.windSpeeds[3600:-3600:Nint][::-1]
        windNT = turb.windSpeeds[3600:-3600:Nint][::-1]
        plot(wind2)
        fwind2 = interpolate.interp1d(t, wind2)
        fwind = interpolate.interp1d(t, windNT)
        costF = controlProblem( result['x'], fwind, fwind2)
        result_m ={'costF':costF, 'pos':r_[xs,ys,zs], 'vel':r_[us,vs,ws], 'vel2':r_[p,q,r], 'angles':r_[phi, theta, xi], 'results' : result, 'fwind':fwind, 't' : dt*arange(nt)}
        results_mon.extend(result_m)
    savez_compressed('Montecarlo_det', results = results2)
    
    # style.use('seaborn-paper')
    # title('Control')
    # figure()
    
    
    
    # plot(xs/1000,ys/1000,'--r',label='Caso aleatorio')
    # legend()
    # xlabel(r'$x$ (km)')
    # tight_layout()
    # ylabel(r'$y$ (km)')
    