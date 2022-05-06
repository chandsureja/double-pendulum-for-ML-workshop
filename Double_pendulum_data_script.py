import numpy as np
import sympy as smp
import pandas as pd
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter



#function to get cartesian coordinates of the 2 pendulum bobs
def get_cartesian_values(t, the1, the2, L1, L2):
    return (L1*np.sin(the1),
            -L1*np.cos(the1),
            L1*np.sin(the1) + L2*np.sin(the2),
            -L1*np.cos(the1) - L2*np.cos(the2))

#We need to define all the variables as symbols in SYMPY module
t, g = smp.symbols ('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols ('L1, L2')
the1, the2 = smp.symbols (r'\theta_1, \theta_2', cls=smp. Function)
the1 = the1(t)
the2 = the2(t)

#differentiating the1 w.r.t 't' 
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

#defining all the trigonometric fucntions 
x1 = L1*smp.sin(the1)
y1 = -L1*smp.cos(the1)
x2 = L1*smp.sin(the1)+L2*smp.sin(the2)
y2 = -L1*smp.cos(the1)-L2*smp.cos(the2)

#knietic
T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1+T2
# Potential
V1 = m1*g*y1
V2 = m2*g*y2
V = V1 + V2
# Lagrangian
L = T-V


LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

sols = smp.solve([LE1, LE2], (the1_dd, the2_dd),simplify=False, rational=False)

dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the1_dd])
dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d), sols[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)


#Reading the Intial conditiions from .csv file
intial_conditions = np.genfromtxt('Intial_conditions.csv',delimiter=',',skip_header=1,max_rows = 1)
t_step =int((intial_conditions[0]*25)+1)


t = np.linspace(0, int(intial_conditions[0]), t_step)
#g = 9.81
#m1=2
#m2=1
#L1 = 2
#L2 = 1
# y0_1 is the intial state of the double pendulum 
# which are theta1,angluar rate of change of theta1, theta2, angluar rate of change of theta2 respectively

y0_1 = list()
for i in range(1,5):
    y0_1.append(intial_conditions[i])

def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2= S
    return [
        dthe1dt_f(intial_conditions[2]),
        dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, intial_conditions[2], intial_conditions[4]),
        dthe2dt_f(intial_conditions[4]),
        dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, intial_conditions[2], intial_conditions[4]),
    ]



#args are g,m1,m2,L1,L2 respectively in that order
ans = odeint(dSdt, y0_1, t=t, args=(intial_conditions[5],intial_conditions[6],intial_conditions[7],intial_conditions[8],intial_conditions[9]))

#creates a csv file with the 2 columns of theta1 and theta2 which would be used later to model the pendulum
the1 = ans.T[0]
the2 = ans.T[2]
theta_final = np.stack((the1,the2),axis =1)
np.savetxt("Double_pendulum_theta.csv",theta_final,delimiter =',')

#similarly creates a csv file with the 4 columns of (x1,y1),(x2,y2) used for modelling and creating animation
x1, y1, x2, y2 = get_cartesian_values(t, ans.T[0], ans.T[2],intial_conditions[8],intial_conditions[9])
x_y_cartesian = np.stack((x1,y1,x2,y2),axis=1)
x_y_cartesian.T
np.savetxt("Double_pendulum_cartesian.csv", x_y_cartesian, delimiter=",")