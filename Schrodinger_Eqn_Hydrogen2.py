from scipy import *
from numpy import *
from scipy import integrate
from scipy import optimize
from matplotlib import pyplot as plt

def wave_function_derv(x, r, l, En):                        
    u, up = x
    return array([up, (l*(l+1)/r**2-2/r-En)*u]) 


def forward_int():                                                              # Function calculates wave function with u(a) as IC
    urf = integrate.odeint(wave_function_derv, x0, R, args=(l,E0))
    u = urf[:,0][::-1]
    norm=integrate.simps(u**2,x=R)
    u *= 1./sqrt(norm)
    return u

def backward_int():                                                             # Function calculates wave function with u(b) as IC
    Rb = R[::-1]                                                                # invert the mesh
    urb = integrate.odeint(wave_function_derv, [0.0, -1], Rb, args=(l,E0))
    u = urb[:,0][::-1]                                                          # Invert u(r)  it in R.
    norm=integrate.simps(u**2,x=R)
    u *= 1./sqrt(norm)
    return u

def Numerovc(f, x0, dx, dh):                                                    # Calculate Wave function using the Numerov algorithm - Check Notes.
    x = zeros(len(f))
    x[0]=x0
    x[1]=x0+dh*dx
    h2 = dh*dh
    h12 = h2/12.
    w0 = x[0]*(1-h12*f[0])
    w1 = x[1]*(1-h12*f[1])
    xi = x[1]
    fi = f[1]
    for i in range(2,f.size):
        w2 = 2*w1-w0+h2*fi*xi  # here fi=f1
        fi = f[i]              # fi=f2
        xi = w2/(1-h12*fi)
        x[i]=xi
        w0 = w1
        w1 = w2
    return x

def fSchrod(En, l, R):                                                          
    return l*(l+1.)/R**2-2./R-En

def SolveSchroedinger(En,l,R):
    """" Use this function to evaluate the wave function. Supply any algorithm within functiom to calculate ur. Currently using Numerov. Can also be calculated 
    using odeint """""
    f = fSchrod(En,l,R[::-1])
    ur = Numerovc(f,0.0,-1e-7,-R[1]+R[0])[::-1]
    norm = integrate.simps(ur**2,x=R)
    ur *= 1/sqrt(abs(norm))
    return ur

def Shoot(En,R,l):
    f = fSchrod(En,l,R[::-1])
    ur = Numerovc(f,0.0,-1e-7,-R[1]+R[0])[::-1]
    norm = integrate.simps(ur**2,x=R)
    ur*1/sqrt(abs(norm))
    ur = ur/R**l
    f0 = ur[0]
    f1 = ur[1]
    f_at_0 = f0 + (f1-f0)*(0.0-R[0])/(R[1]-R[0])
    return f_at_0

def Shoot2(En,R,l):
    f = fSchrod(En,l,R[::-1])
    ur = Numerovc(f,0.0,-1e-7,-R[1]+R[0])[::-1]
    norm = integrate.simps(ur**2,x=R)
    ur*1/sqrt(abs(norm))
    ur = ur/R**l
    poly = polyfit(R[:4], ur[:4], deg = 3)
    return polyval(poly, 0.0)



def FindBoundStates(R,l,nmax,Esearch):
    """" Find all bound states by optimization between energy levels in the defined space """""
    n=0
    Ebnd=[]
    u0 = Shoot2(Esearch[0],R,l)
    for i in range(1,len(Esearch)):
        u1 = Shoot2(Esearch[i],R,l)
        if u0*u1<0:
            Ebound = optimize.brentq(Shoot2,Esearch[i-1],Esearch[i],xtol=1e-16,args=(R,l))
            Ebnd.append((l,Ebound))
            if len(Ebnd)>nmax: break
            n+=1
            print('Found bound state at E=%14.9f E_exact=%14.9f l=%d' % (Ebound, -1.0/(n+l)**2,l))
        u0=u1
    
    return Ebnd

def cmpKey(x):
    return x[1]*1000 + x[0]  # energy has large wait, but degenerate energy states are sorted by l

#urf = forward_int()
#urb = backward_int()
Esearch = -1.2/arange(1,20,0.2)**2

R = linspace(1e-8,100,2000)

nmax=7
Bnd=[]
for l in range(nmax-1):
    Bnd += FindBoundStates(R,l,nmax-l,Esearch)
    
Bnd = sorted(Bnd, key=cmpKey)                       # Sort bounded states by l

Z=28                                               # Specify atomic number 
N=0
rho=zeros(len(R))                                   # Initialize charge density vector
plt.figure()
for (l,En) in Bnd:
    ur = SolveSchroedinger(En,l,R)
    dN = 2*(2*l+1)
    if N+dN<=Z:
        ferm = 1.
    else:
        ferm = (Z-N)/float(dN)
    drho = ur**2 * ferm * dN/(4*pi*R**2)
    rho += drho
    N += dN
    print('adding state (%2d,%14.9f) with fermi=%4.2f and current N=%5.1f' % (l,En,ferm,N))    
    plt.plot(R,rho*(4*pi*R**2),label='charge density with fermi= ' + str(N))
    
    if N>=Z: break

# plt.plot(R,urb)
# plt.plot(R,R*exp(-R)*2)
plt.legend()
plt.xlim([0,40])
plt.xlabel('R')
plt.ylabel('Charge Density')
plt.show()