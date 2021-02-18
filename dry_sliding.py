import math
import time
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.interpolate import interp1d
import multiprocessing as mp
import os

start_time = time.time() #Time recording before calculation

#Initial conditions
m = 58.6934 #mass of one Ni atom in amu
converter = 9648.53329 #unit converter
e = 1.84 #(eV) for Lennard-Jones
r0 = 2.56 #(Angstrom) for Lennard-Jones
subsc = 1001 #Number of substrate atoms in the system (This must be an odd number!!)
dt = 0.01 #Time interval
Steps = 10000 #Step Count
t = [0] #Time
coordinates = int((subsc-1)*3/2) #to calculate starting coordinates of the substrates
distance_substrate = 3 #Distance between each substrate atom (In Angstrom)

xt=[0] ; zt=[6] #Inıtial coordinates of the sliding object (In Angstrom)
vt=[0] #Inıtıal speed of the sliding object
xd=[0] ; zd=[6] #Inıtıal coordinates of the moving agent (In Angstrom)
vD=0.3 #(Angstrom/ps) Velocity of the moving agent

xi=list(range(-coordinates,coordinates+1,distance_substrate))
xi_0=list(range(-coordinates,coordinates+1,distance_substrate))
xi_temporary=[]
zi=[2]*subsc #Inıtıal z coordinates of the substrate atoms
zi_0=[2]*subsc
vi=[0]*subsc #Inıtıal velocities of the substrate atoms in x axis
vi_z=[0]*subsc #Initial velocities of the substrate atoms in z axis

Fn=0 #Normal Force
Ff=[0]

#Spring Constants (eV/Angstrom)
kt=0.4
kx=4.3
kx_prime=5.8
kz=10

Fid1 = open("i_coordinates.xyz","a")
Fid1.truncate(0)

"""
#Training part of the spline Interpolation
y_list = []
x = np.arange(1.5,5.1,0.1)
y = x**3
#y = 4*e*((6*r0**6/x**7)-(12*r0**12/x**13))
f = interp1d(y,x, kind="cubic")
print(y)

z = f(8)

print(z)

plt.plot(x,y)
plt.show()
"""


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

printProgressBar(0,Steps,prefix="Progress",suffix="Complete",length=50)

#Lennard-Jones Potential Force Between single substrate and sliding onject
def VLJ_Single(a,xt):
    r = math.sqrt(math.pow((xt[-1]-a),2)+math.pow(3,2))
    V_ti = 4*e*(-((12*r0**12)/r**14)+((6*r0**6)/r**8))*(a-xt[-1])
    return V_ti

#Lennard-Jones Potential Force between closest 61 atoms and the sliding object
def VLJ_Cluster(xtf,xi,xt_a):
    V_ti_2 = 0
    for l in range(range_min,range_max):
        r = math.sqrt(math.pow((xtf-xi[l]),2) + math.pow(3,2))
        V_ti_2 = V_ti_2 + 4*e*(-((12*r0**12)/r**14)+((6*r0**6)/r**8))*(xt[-1]-xi[l])
    return V_ti_2

def Heat(range_min,range_max,vi):
    total_heat = 0 #Total heat loss of the system
    for n in range(range_min,range_max):
        total_heat = total_heat + (0.5)*m*(vi[n]**2)
    return total_heat

for l in range(Steps):
    #time
    t.append(t[-1]+dt)

    xi_temporary = xi[:]

    #Determining the closest 61 substrate atoms
    xt_a = int(xt[-1]-30)
    range_min = int(((subsc-1)/2)+(xt_a))
    range_max = int(((subsc-1)/2)+61+(xt_a))

    for n in range(range_min,range_max):
        a = xi_temporary[n]
        vi[n] = (vi[n]-(((VLJ_Single(a,xt)+kx*(xi[n]-xi_0[n])+2*kx_prime*xi[n]-kx_prime*(xi_temporary[n+1]+xi_temporary[n-1]))*dt)/m)*converter)
        xi[n] = vi[n]*dt + xi[n]

    """pool = mp.Pool(3) #initilizing pool for multiprocessing
    toral_heat = [pool.apply(Heat, args=(range_min,range_max,vi))]
    pool.close()"""
    total_heat = Heat(range_min,range_max,vi)

    Ff.append(kt*(-xd[-1]+xt[-1])+0.1*vt[-1]) #calculation of friction force with damping

    xd.append(vD*dt+xd[-1]) #calculation of the next position of the moving agent

    xtf = xt[-1] #last known location of the sliding object

    vt.append(vt[-1]-((VLJ_Cluster(xtf,xi,xt_a)+(Ff[-1]))*dt/m)*converter)

    xt.append(xt[l]+vt[-1]*dt)

    Fid1.write("{0:d}\n\n".format(subsc+2))
    for i in range(len(xi)):
        Fid1.write("{0:s} {1:18.17f} {2:18.17f} {3:18.17f}\n".format("Ni",xi[i],zi[i],0))
    Fid1.write("{3:s} {0:f} {1:18.17f} {2:18.17f}\n".format(xtf,6,0,"Ni"))
    Fid1.write("{3:s} {0:f} {1:18.17f} {2:18.17f}\n".format(xd[-1],6,0,"C"))

    printProgressBar(l+1,Steps,prefix="Progress",suffix="Complete",length=50)


Fid1.close()

end_time = time.time()
print("Calculation Time:",end_time-start_time,"s")

plt.plot(xd,Ff)
plt.show()
