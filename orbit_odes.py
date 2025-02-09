
"""
                   Probe orbitting the Moon orbiting the Earth

                   
8 variables: xm, ym, vxm, vym (components of position and velocity of the moon)
             xp, yp, vxp, vyp (components of the position and velocity of the probe)

Initial Conditions: xm0 = 3.83e8 m, ym0 = 0.0 m, vxm0 = 0.0 m/s, vym0 = sqrt(GM xm0) m/s
                    xp0 = xm0 - xpm , yp0 = 0.0 m, vxp0 = 0.0 m/s, vyp0 = sqrt(GM xpm/xp0^2 + 1350G/xpm) m/s
                    xpm = 1e7 m (will be varied)

Elliptical Motion (moon) x0 = 3.6e8 m, y0 = 0.0 m, vx0 = 0.0  m/s, vy0 = sqrt(GM(2/x0 - 1/a)) m/s

Equations of motion: 

dvxm/dt = (-MG/(xm^2+ym^2)^3/2) xm (f3),           dxm/dt = vxm (f1)
dvym/dt = (-MG/(xm^2+ym^2)^3/2) ym (f4),           dym/dt = vym (f2)

dxp/dt = vxp (f5)
dyp/dy = vyp (f6)
dvxp/dt = (-MG/(xp^2+yp^2)^3/2) xp -(mG/((xp-xm)^2+(yp-ym)^2)^3/2) (xp-xm) (f7)
dvyp/dt = (-MG/(xp^2+yp^2)^3/2) yp -(mG/((xp-xm)^2+(yp-ym)^2)^3/2) (yp-ym) (f8)
"""

import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
def derivatives_moon(time, state, M,G):
    """
    Function to compute the derivatives for a single moving body (the moon) in 
    a stationary gravitational (central) potential (the Earth)
    ----------
    time : independent variable, floating point, not used.
    state : tuple of floats containing x, y, vx, vy.
    M : mass of the Earth in kg.
    G : Gravitational constant in Nm^2/kg^2
    -------
    Returns: tuple of derivatives (dx/dt, dy/t, dvx/dt, dvy/dt)
    """
    x,y,vx,vy = state
    f1 = vx
    f2 = vy
    r = np.sqrt(x**2+y**2)
    f3 = -M*G/(r**3)*x
    f4 = -M*G/(r**3)*y
    return (f1,f2,f3,f4)

def derivatives_probe(time, state, M,m,G):
    """
    Function to compute the derivatives for two moving bodies (the moon and a probe)
    in a stationary gravitational (central) potential (the Earth)
    ----------
    time : independent variable, floating point, not used.
    state : tuple of floats containing xm, ym, vxm, vym,xp, yp, vxp, vyp.
    m : mass of the Moon in kg.
    M : mass of the Earth in kg.
    G : Gravitational constant in Nm^2/kg^2
    -------
    Returns: tuple of derivatives (dxp/dt, dyp/t, dvxp/dt, dvyp/dt,dxm/dt, dym/t, dvxm/dt, dvym/dt)
    """
    xm,ym,vxm,vym, xp, yp, vxp, vyp = state
    rm = np.sqrt(xm**2+ym**2)
    rp = np.sqrt(xp**2+yp**2)
    rpm = np.sqrt((xp-xm)**2 + (yp-ym)**2)
    
    f1 = vxm
    f2 = vym
    f3 = -M*G*xm/(rm**3)
    f4 = -M*G*ym/(rm**3)
    f5 = vxp
    f6 = vyp
    f7 = -M*G*xp/(rp**3) - m*G*(xp-xm)/(rpm**3)
    f8 = -M*G*yp/(rp**3) - m*G*(yp-ym)/(rpm**3)
    return (f1,f2,f3,f4,f5,f6,f7,f8)


# Initial Conditions and Constants

G = 6.67e-11 # Gravitational constant (Nm^2/kg^2)
M = 5.972e24 # Mass of the Earth (kg)
m = 7.347e22 # Mass of the moon (kg)
p = 1350 # Mass of probe (kg)


# Initial position and velocity of the moon (Circular orbit)
xm0 = 3.83e8 # m - average distance between the centre of the moon and the Earth
ym0 = 0.0 # m 
vxm0 = 0.0 #m/s
vym0 = np.sqrt(G*M/xm0) # m/s

# Initial position and velocity of the probe
xpm = 1.9e7 # m - vary altitudes to observe effects
xp0 = xm0 - xpm # m, moon and probe start on the x-axis with probe between Earth and moon
yp0 = 0.0
vxp0 = 0.0
vyp0 = np.sqrt((G*M*xpm/xp0**2)+(G*m/xpm)) 


# Initial position and velocity of the moon (Elliptial Orbit)
a = 3.844e8 # m, semi-major axis of moon's orbit
y0 = 0.0 # m
x0 = 4.05e8 # m
vy0 = np.sqrt(G*M*((2/x0) - (1/a)))  # m/s, Vis Viva equation
vx0 = 0.0 # m/s


t0 = 0.0 #s, initial time
tmax =  2358720 # s, final time - 27.3 days, ie: 1 orbit of the moon
numpoints = 1000
times = np.linspace(t0,tmax, numpoints)

# Tolerances
rtol=1e-14 # relative tolerance of solve_ivp method
atol=1e-6 # absolute tolerance of solve_ivp method


# Describes initial states for desired result using tuples
initial_state_ellipse = (x0,y0,vx0,vy0)
initial_state_probe = (xm0,ym0,vxm0,vym0, xp0, yp0, vxp0, vyp0) 
      # only need this as plotting just moon's position will give circular orbit

# Solving the equations of motion for an elliptical orbit
results_ellipse = si.solve_ivp(derivatives_moon, (t0,tmax), initial_state_ellipse, method='RK45',t_eval=times, rtol=rtol, atol=atol, args = (G,M))
x = results_ellipse.y[0,:]
y = results_ellipse.y[1,:]
vx = results_ellipse.y[2,:]
vy = results_ellipse.y[3,:]

# Sovling the equations of motion for the moon-probe system / circular orbit
results_probe = si.solve_ivp(derivatives_probe, (t0,tmax), initial_state_probe, method='RK45',t_eval=times, rtol=rtol, atol=atol, args = (M,m,G))
xm = results_probe.y[0,:]
ym = results_probe.y[1,:]
vxm = results_probe.y[2,:]
vym = results_probe.y[3,:]

xp = results_probe.y[4,:]
yp = results_probe.y[5,:]
vxp = results_probe.y[6,:]
vyp = results_probe.y[7,:]

# Energy of moon
v_m = np.sqrt(vxm**2+vym**2)
KEm = m/2*(v_m**2)
expect_KE = np.array([m/2*(vym0**2)]*numpoints)

rm = np.sqrt(xm**2+ym**2)
GPEm = -1*G*M*m/rm
expect_GPE = np.array([-1*G*M*m/xm0]*numpoints)

expect_E = expect_KE + expect_GPE

E_tot = KEm+GPEm
perc_E = abs((E_tot-expect_E)*100/expect_E)



# Energy of Probe
v_squared = (vxp**2+vyp**2)
KEp = p*v_squared/2

rp = np.sqrt(xp**2+yp**2)
rpm = np.sqrt((xp-xm)**2 + (yp-ym)**2)
GPEp = (-1*G*M*p/rp) + (-1*G*m*p/rpm)

tot_E = KEp+GPEp



choice ="0" # variable to allow user to choose what they see
while choice != "q" :
    choice = input ( "Enter a choice, 1, 2, 3 or q to quit: ")
    print("You entered the choice:", choice )
    if choice == "1" :
        print( "You have chosen part (1): simulation of a circular lunar orbit")
        ax = plt.axes()
        ax.set_aspect(1.0)

        ax.set_xlabel("x coordinate (m)")
        ax.set_ylabel("y coordinate (m)")
        ax.set_title("Circular Orbit of the moon around the Earth.")

        ax.plot(xm,ym)
        ax.plot(0.0,0.0, marker=".", markersize=30, markeredgecolor="green")
        ax.legend(["Moon","Earth"],loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.show()
        
        
        
    elif choice == "2" :
        print( "You have chosen part (2): simulation for elliptical lunar orbit" )

        ax = plt.axes()
        ax.set_aspect(1.0) 
        ax.set_xlabel("x coordinate (m)")
        ax.set_ylabel("y coordinate (m)")
        ax.set_title(" Elliptical orbit of the moon around the Earth")
        ax.plot(x,y)
        ax.plot(0.0,0.0, marker=".", markersize=30, markeredgecolor="orange")
        ax.legend(["Moon","Earth"],loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.show()


    elif choice == "3":
        print( "You have chosen part (3): earth-moon-probe system" )
        ax = plt.axes()
        ax.set_aspect(1.0)

        ax.set_xlabel("x coordinate (m)")
        ax.set_ylabel("y coordinate (m)")
        ax.set_title("Orbit of the moon around the Earth and Probe around Moon")

        ax.plot(xm,ym)
        ax.plot(xp,yp)
        ax.plot(0.0,0.0, marker=".", markersize=30, markeredgecolor="green")
        ax.legend(["Moon","Probe","Earth"],loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.show()
        
    elif choice != "q" :
        print( "This is not a valid choice" )
print("You have chosen to finish this section." )









choice2 = "0"
while choice2 != "q" :
    choice2 = input ( "Enter a choice, 1 (moon - total energy), 2 (moon - KE and GPE) 3 (probe energy) or q to quit:")
    print("You entered the choice:" , choice2 )
    if choice2 == "1" :
        print("You have chosen part (1): total energy of the moon" )
        plt.figure()
        plt.suptitle("Evaluation of Energy Conservation for Moon's Circular Orbit")

        plt.subplot(211)
        plt.plot(times, E_tot, label = "Calculated total energy")
        plt.plot(times, expect_E, label = "Expected total energy")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.legend(loc='center left', bbox_to_anchor=(0.5, 1.0))

        plt.subplot(212)
        plt.plot(times, perc_E, label = "Percentage Error", color = "turquoise")
        plt.legend()

        plt.tight_layout()
        plt.show()


    elif choice2 == "2":
        print("You have chosen part (2): kinetic and gravitational potential energy of the moon" )
        plt.figure()
        plt.suptitle("Comparison of Expected and Calculated energies of the Moon's orbit.")

        plt.subplot(211)
        plt.plot(times, KEm, label = "Calculated KE", color = "purple")
        plt.plot(times, expect_KE, label = "Expected KE", color = "blue")
        plt.legend(loc='center left', bbox_to_anchor=(0.5, 1.2))

        plt.subplot(212)
        plt.plot(times, GPEm, label = "Calculated GPE", color = "purple")
        plt.plot(times, expect_GPE, label = "Expected GPE", color = "blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.legend(loc='center left', bbox_to_anchor=(0.5, 1.2))

        plt.tight_layout()
        plt.show()

    elif choice2 == "3" :
        print("You have chosen part (2): energy of probe" )
        plt.figure()
        plt.suptitle("Evaluation of Energy Transfer for Probe")

        plt.plot(times, tot_E, label = "Calculated total energy")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.legend(loc='center left', bbox_to_anchor=(0.5, 1.2))

        plt.tight_layout()
        plt.show()

    elif choice2 != "q" :
        print( "This is not a valid choice." )
        print("You have chosen to finish - goodbye.")
















