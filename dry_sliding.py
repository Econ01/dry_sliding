import math
import time
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.interpolate import interp1d
import multiprocessing as mp
import os

start_time = time.time()

#Initial conditions
m = 58.6934 #mass of one Ni atom in amu
m_kg = 9.7462675*math.pow(10,-26)
N = 6.0221409*math.pow(10,23)
converter = 9648.53329 #unit converter
e = 0.84 #(eV) for Lennard-Jones
r0 = 2.56 #(Angstrom) for Lennard-Jones
kB = 1.38064852 * math.pow(10,-23) #Boltzmann Constant
subsc = 1001 #Number of substrate atoms in the system (This must be an odd number!!)
dt = 0.01 #Time interval
Steps = 5000 #Step Count
t = [0] #Time
heat_total = [] #To store lost energy
target_heat = 297.15 #Environment temperature in Kelvin
distance_substrate = 3 #Distance between each substrate atom (In Angstrom)
coordinates = int((subsc-1)*distance_substrate/2) #to calculate starting coordinates of the substrates

xt=[0] ; zt=[5] #Inıtial coordinates of the sliding object (In Angstrom)
vt=[0] #Inıtıal velocity of the sliding object on x axis
vt_z=[0] #Initial velocity of the sliding object on z axis
xd=[4] ; zd=[6] #Inıtıal coordinates of the moving agent (In Angstrom)
vD=0.4 #(Angstrom/ps) Velocity of the moving agent

#Inıtial coordinates and velocities of substrates
xi=list(range(-coordinates,coordinates+1,distance_substrate))
xi_0=list(range(-coordinates,coordinates+1,distance_substrate))
xi_temporary=[]
zi=[2]*subsc
zi_0=[2]*subsc
vi=[0]*subsc
vi_z=[0]*subsc

Fn=0 #Normal Force
Ff=[0] #Friction Force

#Spring Constants (eV/Angstrom)
kt=0.2
kx=4.3
kx_prime=5.8
kz=10

Fid1 = open("i_coordinates.xyz","a")
Fid1.truncate(0)

#Checks whether the user wants to use normalized heat function or not.
def NormalizationPrompt(prompt_want):
    while True:
        try:
            return {"y":True,"n":False}[input(prompt_want).lower()]
        except KeyError:
            print("Invalid input please enter y or n (Default is n)!")
Normalization_Answer=bool(NormalizationPrompt("Do you want to normalize system heat? (y/n) \n"))

#Asking user to input a step interval for determining normalization period.
def NormalizationPrompt(prompt_steps):
    while True:
        try:
            return int(input(prompt_steps))
        except ValueError:
            print("Invalid input please enter an integer!")
if(Normalization_Answer==True):
    Normalization_Steps=int(NormalizationPrompt("Normalization period (Every nth step) \n"))

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
def InteractionForce_Substrate_X(xi_temp,xt,zi_temp,zt):
    r = math.sqrt(math.pow((xt[-1]-xi_temp),2)+math.pow(zt[-1]-zi_temp,2))
    IF_Substrate_X = 4*e*(-((12*r0**12)/r**14)+((6*r0**6)/r**8))*(xi_temp-xt[-1])
    return IF_Substrate_X

def InteractionForce_Substrate_Z(xi_temp,xt,zi_temp,zt):
    r = math.sqrt(math.pow((xt[-1]-xi_temp),2)+math.pow((zt[-1]-zi_temp),2))
    IF_Substrate_Z = 4*e*(-((12*r0**12)/r**14)+((6*r0**6)/r**8))*(zi_temp-zt[-1])
    return IF_Substrate_Z
#--------------------------------------------------------------------------

#Lennard-Jones Potential Force between closest 61 atoms and the sliding object
def InteractionForce_Object_X(xtf,xi,ztf,zi):
    IF_Object_X = 0
    for l in range(range_min,range_max):
        r = math.sqrt(math.pow((xtf-xi[l]),2) + math.pow(ztf-zi[l],2))
        IF_Object_X = IF_Object_X + 4*e*(-((12*r0**12)/r**14)+((6*r0**6)/r**8))*(xtf-xi[l])
    return IF_Object_X

def InteractionForce_Object_Z(xtf,xi,ztf,zi):
    IF_Object_Z = 0
    for l in range(range_min,range_max):
        r2 = (xtf-xi[l])*(xtf-xi[l])+(ztf-zi[l])*(ztf-zi[l])
        IF_Object_Z = IF_Object_Z + 4*e*6*(r0**6)*((r2**3-(2*r0**6))/(r2**7))*(ztf-zi[l])
    return IF_Object_Z
#--------------------------------------------------------------------------

#Calculating the z coordinate for the object that makes the net force 0 in z axis
def Object_Z_Calculator(xi,zi,xtf,ztf):
    while(True):
        TempForce_Z = 0
        for l in range(range_min,range_max):
            r = math.sqrt(math.pow((xtf-xi[l]),2) + math.pow(ztf-zi[l],2))
            TempForce_Z = TempForce_Z + 4*e*(-((12*r0**12)/r**14)+((6*r0**6)/r**8))*(ztf-zi[l])
        TempForce_Z = TempForce_Z + Fn
        if(TempForce_Z>=0 and TempForce_Z<=0.1):
            return ztf
        else:
            if(TempForce_Z<0):
                ztf+=0.001
            elif(TempForce_Z>0):
                ztf-=0.001

def Heat(range_min,range_max,vi,vi_z):
    total_energy = 0
    for n in range(range_min,range_max):
        total_energy = total_energy + (0.5)*m_kg*((vi[n]*100)**2) + (0.5)*m_kg*((vi_z[n]*100)**2)
    heat = (2*(total_energy/(range_max-range_min)))/(3*kB)
    #heat = (total_energy / (1.5*kB*(range_max-range_min))) * ((math.pow(10,-24))/(6.022))
    return heat

#Rescaling the velocities according to the normalized heat
def Rescale(vi,vi_z,heat_total):
    vi_rescale = vi*math.sqrt(target_heat/heat_total)
    vi_z_rescale = vi_z*math.sqrt(target_heat/heat_total)
    return vi_rescale,vi_z_rescale

for l in range(Steps):
    #time
    t.append(t[-1]+dt)

    xi_temporary = xi[:]
    zi_temporary = zi[:]

    #Determining the closest 61 substrate atoms
    xt_a = int(xt[-1]-30)
    range_min = int(((subsc-1)/2)+(xt_a))
    range_max = int(((subsc-1)/2)+61+(xt_a))

    #Calculation of velocities and next coordinates of substrates
    for n in range(range_min,range_max):
        zi_temp = zi_temporary[n]
        xi_temp = xi_temporary[n]
        vi[n] = (vi[n]-(((InteractionForce_Substrate_X(xi_temp,xt,zi_temp,zt)+kx*(xi[n]-xi_0[n])+2*kx_prime*xi[n]-kx_prime*(xi_temporary[n+1]+xi_temporary[n-1]))*dt)/m)*converter)
        vi_z[n] = (vi_z[n]-(((kz*(zi[n]-zi_0[n])+InteractionForce_Substrate_Z(xi_temp,xt,zi_temp,zt)+0.1*vi_z[n])*dt)/m)*converter)
        zi[n] = vi_z[n]*dt + zi[n]
        xi[n] = vi[n]*dt + xi[n]
    #--------------------------------------------------------------------------

    #Nomalization of the velocity values for substrades
    Step_Counter=False
    if(Normalization_Answer==True):
        Remainder = l%(Normalization_Steps)
        Step_Counter = (Remainder == 0)
    if(Step_Counter==True and Normalization_Answer==True and l!=0):
        for n in range(range_min,range_max):
            vi_rescale,vi_z_rescale = Rescale(vi[n],vi_z[n],heat_total[-1])
            vi[n]=vi_rescale
            vi_z[n]=vi_z_rescale

    #Indexing the total heat output of the system step by step
    heat_total.append(Heat(range_min,range_max,vi,vi_z))

    Ff.append(kt*(xd[0]-xd[-1]+xt[-1])) #calculation of friction force

    xd.append(vD*dt+xd[-1]) #calculation of the next position of the moving agent

    #last known location of the sliding object
    xtf = xt[-1]
    ztf = zt[-1]

    #Calculation of velocities and next coordinates of the object
    vt.append(vt[-1]-((InteractionForce_Object_X(xtf,xi,ztf,zi)+(Ff[-1])+0.1*vt[-1])*dt/m)*converter)
    xt.append(xt[l]+vt[-1]*dt)

    z_true = Object_Z_Calculator(xi,zi,xtf,ztf)

    vt_z.append(vt_z[-1]-((Fn + InteractionForce_Object_Z(xtf,xi,ztf,zi) + 4.3*(zt[-1]-(z_true)) + 0.1*vt_z[-1])*dt/m)*converter)
    zt.append(zt[l]+vt_z[-1]*dt)
    #--------------------------------------------------------------------------

    Fid1.write("{0:d}\n\n".format(subsc+2))
    for i in range(len(xi)):
        Fid1.write("{0:s} {1:18.17f} {2:18.17f} {3:18.17f}\n".format("Ni",xi[i],zi[i],0))
    Fid1.write("{3:s} {0:f} {1:18.17f} {2:18.17f}\n".format(xtf,ztf,0,"Ni"))
    Fid1.write("{3:s} {0:f} {1:18.17f} {2:18.17f}\n".format(xd[-1],ztf,0,"C"))

    printProgressBar(l+1,Steps,prefix="Progress",suffix="Complete",length=50)


Fid1.close()

end_time = time.time()
print("Calculation Time:",end_time-start_time,"s")

plt.plot(xd,Ff)
plt.show()
