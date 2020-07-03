from pylab import *
from numba import jit
import numba as nb
from numba import jitclass
m = 5.6*10**4
delta = 7.4*10**5
rho = 0.089
# xcg = 5
# zcg = 15
xcg = 0
zcg = 0
xp = 4
yp = 0
zp = 40
Ix = 5*10**7
Iy = 2.9*10**8
Iz = 2.9*10**8
Ixz = -6*10**4

eta = pi/6
g = 9.81
k1 = 0.17
k2 = 0.83
k3 = 0.52

Cl1 = 2.4 * 10**4
Cm1 = 7.7*10**4
Cn1 = 7.7*10**4
Cm2 = 7.7*10**4
Cn2 = 7.7*10**4
Cm3 = 7.7*10**4
Cn3 = 7.7*10**4
Cm4 = 7.7*10**4
Cn4 = 7.7*10**4
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


@jitclass()
class airship():
    def __init__(self):
        A = zeros([3, 3])
        B = zeros([3, 3])
        self.x = zeros(6)
        self.y = zeros(6)
        self.M = zeros([3, 3])
        self.C = zeros([3, 3])
        self.D = zeros([3, 3])
        self.tau = zeros([3, 3])
        self.J = zeros([3, 3])
        self.V = zeros([3, 3])
        self.R = zeros([6, 6])
        self.ucontrol = zeros([6,1])
        self.set_A(Ix, Ixz, m, zcg, Iy, rho, xcg, Iz, delta, k1, k2, k3)
        self.Wind = [0,0]
        self.t = 0
        self.Bg = m*g
        self.calcF = lambda t: 0
        self.calcDeltar = lambda t: 0
        self.calcDeltae = lambda t: 0
        self.N3 =zeros([6,1])
        self.B = zeros([6,6])
        
    def calcRHS(self, x, y):
#        print('x:  {}'.format(x))
#        print('y:  {}'.format(y))
        phi = x[0]
        theta = x[1]
        xi = x[2]
        # xp =x[3]
        # yp = x[4]
        # z = x[5]
        p = y[0]
        q = y[1]
        r = y[2]
        u = y[3]#-self.Wind[0]
        v = y[4]#-self.Wind[1]
        w = y[5]
        self.set_N(Ix, Iy, Iz, u, v, w, p, q, r, rho, delta, xcg, zcg, m)
        self.set_R(phi, theta, xi)
        alpha = arctan2(w, u)
        betha = arctan2(v*cos(alpha), u)
        V = sqrt(u**2 + v**2 + w**2)
        Q = rho*V**2/2
        self.set_N3(u, v, w, alpha, betha, Q, Cl1)
        self.set_G(m, g, xcg, zcg, theta, phi)
        self.set_B(eta, Q, Cm4)
        xdot = self.calcxdot(y)
        ydot = self.calcydot(self.ucontrol)
        # print(ydot)
        return xdot, ydot
    
    def RK4(self, dt):
        self.calcControl()
        x = self.x
        y = self.y
        self.y[3] = self.y[3]+self.Wind[0]
        self.y[4] = self.y[4]+self.Wind[1]
        k1x, k1y = self.calcRHS(x, y)
#        print('k1x: {} k1y:{}'.format(k1x, k1y))
        k2x, k2y = self.calcRHS(x+0.5*k1x*dt, y + 0.5*k1y*dt)
        k3x, k3y = self.calcRHS(x+0.5*k2x*dt, y + 0.5*k2y*dt)
        k4x, k4y = self.calcRHS(x+k3x*dt, y + k3y*dt)
        self.x = x + 1/6*dt*(k1x + 2*k2x + 2*k3x + k4x)
        self.y = y + 1/6*dt*(k1y + 2*k2y + 2*k3y + k4y)
        self.y[3] = self.y[3]-self.Wind[0]
        self.y[4] = self.y[4]-self.Wind[1]
        self.t +=dt
        
    def calcxdot(self, y):
        xdot = self.R@y
        return(xdot)
        
    def calcydot(self, u):
        # print('A:{}'.format(A))
        # print('N:{}'.format(sel))
        ydot = linalg.solve(self.A,self.N3 + self.N + self.G + self.B@u)
        # ydot = linalg.solve(self.A,self.B@u)
        # print(u)
        # print(self.B)
        # print(self.B@u)
#        print('ydot {}    {}'.format(shape(u), shape(self.B)))
        return(ydot.reshape(-1))
#
    def setJ(phi):
        self.J[0,0] = cos(phi)
        self.J[0,1] = -sin(phi)
        self.J[1,0] = sin(phi)
        self.J[1,1] = cos(phi)
        self.J[2,2] = 1
        
    def set_R(self, phi, theta, xi):
        # self.set_Rgamma(phi, theta)
        # self.set_Rb(theta, phi, xi)
        ttheta = tan(theta)
        cphi = cos(phi)
        sphi = sin(phi)
        ctheta = cos(theta)
        stheta = sin(theta)
        cxi = cos(xi)
        sxi = sin(xi)
        self.R[:3,:3] = [
            [1, ttheta*sphi, ttheta*cphi],
            [0, cphi, -sphi],
            [0, sphi/ctheta, cphi/ctheta]
            ]
        self.R[3:,3:] = [
            [ctheta*cxi, stheta*cxi*sphi-sphi*cphi, stheta*cxi*cphi+sphi*sphi],
            [ctheta*sphi, stheta*sphi*sphi+cxi*cphi, stheta*cxi*cphi-cxi*sphi],
            [-stheta, ctheta*sphi, ctheta*cphi]
            ]
        # self.R[:3,:3] = self.Rgamma
        # self.R[3:,3:] = self.Rb

#     def set_Rgamma(self, phi, theta):
# #        print('{}    {}'.format(phi, theta))
#         self.Rgamma = [
#             [1, tan(theta)*sin(phi), tan(theta)*cos(phi)],
#             [0, cos(phi), -sin(phi)],
#             [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]
#             ]

#     def set_Rb(self, theta, phi, xi):
#         self.Rb = [
#             [cos(theta)*cos(xi), sin(theta)*cos(xi)*sin(phi)-sin(xi)*cos(phi), sin(theta)*cos(xi)*cos(phi)+sin(xi)*sin(phi)],
#             [cos(theta)*sin(xi), sin(theta)*sin(xi)*sin(phi)+cos(xi)*cos(phi), sin(theta)*cos(xi)*cos(phi)-cos(xi)*sin(phi)],
#             [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)]
#             ]

    def set_A(self, Ix, Ixz, m, zcg, Iy, rho, xcg,Iz,delta, k1, k2, k3):
        self.A = array([
            [Ix, 0, -Ixz, 0, -m*zcg, 0],
            [0, Iy+rho*delta*k3, 0, m*zcg, 0, -m*xcg],
            [-Ixz, 0, Iz+rho*delta*k3, 0, m*xcg, 0],
            [0, m*zcg, 0, m+rho*delta*k1, 0, 0],
            [-m*zcg, 0, m*xcg, 0, m+rho*delta*k2, 0],
            [0, -m*xcg, 0, 0, 0, m+rho*delta*k2]
        ])
        # self.A = array([
        #      [0, m*zcg, 0, m+rho*delta*k1, 0, 0],
        #     [-m*zcg, 0, m*xcg, 0, m+rho*delta*k2, 0],
        #     [0, -m*xcg, 0, 0, 0, m+rho*delta*k2],
        #    [Ix, 0, -Ixz, 0, -m*zcg, 0],
        #     [0, Iy+rho*delta*k3, 0, m*zcg, 0, -m*xcg],
        #     [-Ixz, 0, Iz+rho*delta*k3, 0, m*xcg, 0]
        # ])
        
        

    def set_N(self, Ix, Iy, Iz, u, v, w , p, q, r, rho, delta, xcg, zcg, m):
        self.N = array([
            [-(Iz - Iy)*q*r + Ixz*p*q + m*zcg*(u*r-w*p) ],
            [-(Ix-Iz-rho*delta*k3)*p*r-Ixz*(p**2-r**2) - m*zcg*(w*q-v*r) + m*xcg*(v*p-u*q) ],
            [-(Iy + rho*delta*k3 - Ix)*p*q-Ixz*q*r - m*xcg*(u*r-w*p) ],
            [-(m + rho*delta*k2)*(w*p-v*r) - m*zcg*p*r + m*xcg*(q**2+r**2)],
            [(m + rho*delta*k2)*w*p - (m + rho*delta*k1)*u*r - m*xcg*p*q - m*zcg*q*r ],
            [(m + rho*delta*k1)*u*q - (m + rho*delta*k2)*v*p - m*xcg*r*p + m*zcg*(p**2 + q**2) ]
        ])

    def set_G(self, m, g, xcg, zcg, theta, phi):
        self.G = array([
            [-zcg*m*g*cos(theta)*sin(phi)],
            [-zcg*m*g*sin(theta)-xcg*m*g*cos(theta)*cos(phi)],
            [xcg*m*g*cos(theta)*sin(phi)],
             [(self.Bg-m*g)*sin(theta)],
             [-(self.Bg-m*g)*cos(theta)*sin(phi)],
             [-(self.Bg-m*g)*cos(theta)*cos(phi)]
        ])
    
    def set_B(self, eta, Q, cm4):
#        print('{} {} {}'.format(eta, Q, cm4))
        self.B[:,:] = array([
                [zp*sin(eta), -zp*sin(eta), yp, -yp, 0, 0],
            [zp*cos(eta), zp*cos(eta), -xp, -xp, 2*Q*Cy4, 0],
            [xp*sin(eta) - yp*cos(eta), xp*sin(eta) + yp*sin(eta), 0, 0, 0, -2*Q*Cz4],
            [ cos(eta), cos(eta), 0, 0, 0, 0],
            [ sin(eta), -sin(eta), 0, 0, 0, -2*Q*Cm4],
            [0, 0, 1, 1, -2*Q*Cn4, 0]
            
        ])
    def setControl(self, Ffunc, rFunc, dFunc):
        self.calcF = Ffunc
        self.calcDeltar = rFunc
        self.calcDeltae = dFunc

    def calcControl(self):
        F = self.calcF(self.t)
        deltar = self.calcDeltar(self.t)
        deltae = self.calcDeltae(self.t)
        self.ucontrol[0] = F
        self.ucontrol[1] = F
        self.ucontrol[2] = 0
        self.ucontrol[3] = 0
        self.ucontrol[4] = deltar
        self.ucontrol[5] = deltae
        # self.ucontrol = array([F, F, 0, 0, deltar, deltae])
        # self.ucontrol.shape = (6,1)
        
    def set_u(self, mul, mur, Fl, Fr, deltar, deltae):
        self.ucontrol = array([Fl*cos(mul), Fr*cos(mur), Fl*sin(mul), Fl*sin(mur), deltar, deltae])
        self.ucontrol.shape = (6,1)
        
    def set_N3(self, u, v, w, alpha, betha, Q, Cl1):
        # sbetha = sin(betha)
        abetha = abs(betha)
        aalpha = abs(alpha)
        self.N3[0] = Q*Cl1*sin(betha)*sin(abetha)
        s2betha = sin(2*betha)
        s2alpha = sin(2*alpha)
        c_2betha = cos(betha/2)
        c_2alpha = cos(alpha/2)
        calpha = cos(alpha)
        cbetha = cos(betha)
        
        self.N3[1] = -Q*(Cm1*c_2alpha*s2alpha + Cm2*s2alpha + Cm3*sin(alpha)*sin(aalpha))
        self.N3[2] = Q*(Cn1*c_2betha*s2betha + Cn2*s2betha + Cn3*sin(betha)*sin(abetha))
        self.N3[3] = -Q*(Cx1*calpha**2*cbetha**2 + Cx2*s2betha*sin(alpha/2))
        self.N3[4] = -Q*(Cy1*c_2betha*s2betha + Cy2*s2betha + Cy3*sin(betha)*sin(abetha))
        self.N3[5] = -Q*(Cz1*c_2alpha*s2alpha + Cz2*s2alpha + Cz3*sin(alpha)*sin(aalpha))
        # self.N3 *= Q
        # self.N3.shape = (6,-1)

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


if(__name__=='__main__'):
    nt = 160*10**2
    xs = zeros(nt)
    ys = zeros(nt)
    zs = zeros(nt)
    
    us = zeros(nt)
    vs = zeros(nt)
    ws = zeros(nt)
    
    
    zep = airship()
    # zep.set_u(0, 0, 10**1,10**1, 0, 0.1)
    zep.Wind = array([5, 0])
    for i in arange(nt):
        zep.RK4(1)
        xs[i] = zep.x[3]
        ys[i] = zep.x[4]
        zs[i] = zep.x[5]
        us[i] = zep.y[3]
        vs[i] = zep.y[4]
        ws[i] = zep.y[5]
    
    figure()   
    plot(xs,label='x')
    plot(ys, label='y')
    plot(zs, label = 'z')
    legend()
    figure()
    plot(us,label='u')
    plot(vs,label='v')
    plot(ws,label='w')
    legend()