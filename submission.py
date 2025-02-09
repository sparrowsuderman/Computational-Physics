"""
               Fresnel and Fruanhofer Diffraction 
using a scipy methods of numerical integration and the Monte Carlo method

Fresnel  (near-field): aperture width = 2e-4m; screen distance = 0.005m
Fraunhofer (far-field): aperture width = 2e-5; screen distance = 0.05m

Created on Fri Nov 15 14:30:12 2024
@author: Sparrow Suderman 
"""
# IMPORT MODULES ==============================================================
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
import time

# FUNCTIONS ===================================================================
# INTEGRAND FUNCTIONS----------------------------------------------------------
def Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen):
    """
    The real part of the integrand or the electric field at a point on 
    screen (x,y,z) from (x',y',0) on the aperture.
    ----------
    x_aperture, y_aperture : variable; coordinates of a point on the aperture
    x_screen, y_screen : variable; coordinates of a point on the screen
    z_screen : variable; distance to the screen
    -------
    value of the real part of the integrand for a given point in the aperture 
    and point on the screen
    """
    A = ((x_screen - x_aperture)**2 + (y_screen - y_aperture)**2)*k/(2*z_screen)
    B = k/(2*np.pi*z_screen)
    return B*np.cos(A)

def Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen):
    """
    The imaginary part of the integrand for the electric field at a point on 
    screen ((x,y,z) from (x',y',0) on the aperture.
    ----------
    x_aperture, y_aperture : variable; coordinates of a point in the aperture
    x_screen, y_screen : variable; coordinates of a point on the screen
    z_screen : variable; distance to the screen
    -------
    The value of the imaginary part of the integrand for a given point in the 
    aperture and point on the screen.
    """
    A = ((x_screen - x_aperture)**2 + (y_screen - y_aperture)**2)*k/(2*z_screen)
    B = k/(2*np.pi*z_screen)
    return B*np.sin(A)

# 1D FUNCTIONS-----------------------------------------------------------------
def solve_1D(xvals_screen, num_points, aperture_dimensions, epsrel, epsabs, z_screen):
    """
    Return intensity and percentage error of diffraction pattern due to a 
    rectangular aperture.
    ----------
    xvals_screen : 1D array; values for intensity to be calculated
    num_points : integer; number of points integral is calculated at
    aperture_dimensions :  tuple; maximum and minimum x and y values
    epsrel, epsabs : float; tolerances for numerical integration
    -------
    Returns intensity and percentage error for a range of x values.
    """
    initial_exact = time.time() # start time for complete computation
    intensity = np.zeros(num_points) # empty array to assign values for intensity
    perc_error = np.zeros(num_points) # emtpy array to assign values for percentage error
    (xmin_aperture, xmax_aperture, ymin_aperture, ymax_aperture) = aperture_dimensions
    y_screen = 0.0 # m, 1D pattern
    for i in range(num_points): # iterate across all x-values
        if i == 0: # in first runthrough loop
            initial_sample = time.time() # start time for first loop
        
        x_screen = xvals_screen[i] # m, iterate through x values across screen
        realpart, realerror = dblquad(Fresnel_2d_real,
                                      xmin_aperture, xmax_aperture, ymin_aperture, ymax_aperture,
                                      args = (y_screen, x_screen, z_screen),
                                      epsabs=epsabs, epsrel=epsrel)
        imagpart, imagerror = dblquad(Fresnel_2d_imag,
                                      xmin_aperture,xmax_aperture, ymin_aperture, ymax_aperture,
                                      args = (y_screen, x_screen, z_screen),
                                      epsabs=epsabs, epsrel=epsrel)
        E_squared = realpart**2 + imagpart**2 # (V/m)^2, the squared magnitude of the E-field
        intensity[i] = E_squared # (V/m)^2, relative intensity of light
        dI = np.sqrt((2*realpart*realerror)**2 + (2*imagpart*imagerror)**2) # (V/m)^2, propogation of errors
        perc_error[i] = dI/E_squared*100 # percentage error of the intensity
        
        if i == 0: # in first runthrough loop
            final_sample = time.time() # end time for first loop
            time_taken_estimate = num_points*(final_sample - initial_sample) # estimate for total time based on first runthrough
            print("This should take at least {:4.2f} seconds.".format(time_taken_estimate))
        
    rel_intensity = np.zeros(num_points) # sets all 0<I<1.
    max_value = intensity.max()
    for i in range(num_points):
        rel_intensity[i] = intensity[i]/max_value
        
    final_exact = time.time() # end time for complete computation
    time_taken_exact = (final_exact - initial_exact)
    percentage_time = time_taken_exact/time_taken_estimate *100
    print("This process took {:4.2f} seconds, {:4.2f}% of expected time.".format(time_taken_exact, percentage_time))   
    return rel_intensity, perc_error
 
def plot_1D(xvals_screen, intensity, perc_error, xmax_aperture, title):
    """
    Plots one dimensional dffraction pattern for a square/rectangular aperture
    ----------
    xvals_screen : array; values of x that intensity was found for
    intensity : array; values of intensity at x values across a screen
    perc_error : array; percentage error of the intensity
    xmax_aperture : float; half the aperture width
    title: string; title for graph
    -------
    Plot of intensity and percentage error against x value.
    """
    plt.figure() # Plotting multiple graphs on one figure
    
    plt.suptitle(title)

    plt.subplot(211) # Plotting the intensity against x
    plt.plot(xvals_screen*(10**3), intensity)
    plt.xlabel("Screen coordinate (mm)")
    plt.ylabel("Relative Intensity")

    plt.subplot(212) # Plotting the percentage error against x
    plt.plot(xvals_screen*(10**3), perc_error)
    plt.xlabel("Screen coordinate (mm)")
    plt.ylabel(" Percentage Error")

    plt.tight_layout()
    plt.show()

#2D FUNCTIONS------------------------------------------------------------------
def solve_2D(num_points, xvals_screen, yvals_screen, z_screen, aperture_dimensions, epsrel, epsabs):
    """
    Uses scipy.dblquad to solve the 2D integral for diffraction from an aperture
    ----------
    num_points : integer
    xvals_screen : array; x values across the screen
    yvals_screen : array; y values across the screen
    z_screen : float; distance of screen from aperture
    aperture_dimensions : tuple; the boundaries of the aperture
    epsrel, epsabs : float; tolerances of integration
    -------
    Returns array of intensity.
    """
    initial_exact = time.time() # initial time for complete computation
    intensity = np.zeros((num_points, num_points))
    (xmin_aperture, xmax_aperture, ymin_aperture, ymax_aperture) = aperture_dimensions
    for i in range(num_points):
        x_screen = xvals_screen[i] # iterates through all x values
        for j in range(num_points): 
            y_screen = yvals_screen[j] # interates through all y values
            if (i, j) == (0,0): # first runthrough loop
                initial_sample = time.time()
            realpart, realerror = dblquad(Fresnel_2d_real, 
                                          xmin_aperture, xmax_aperture, ymin_aperture, ymax_aperture, 
                                          args = (y_screen, x_screen, z_screen), 
                                          epsabs=epsabs, epsrel=epsrel)  
            imagpart, imagerror = dblquad(Fresnel_2d_imag,
                                          xmin_aperture, xmax_aperture, ymin_aperture, ymax_aperture, 
                                      args = (y_screen, x_screen, z_screen),
                                      epsabs=epsabs, epsrel=epsrel) 
            E_squared = realpart**2 + imagpart**2 # valus of intensity
            intensity[i][j] = E_squared
            if (i,j) == (0,0): # first runthrough loop
                final_sample = time.time()
                time_taken_estimate = (num_points*(final_sample - initial_sample)) # estimate for time taken based on first loop
                print("This should take at least {:4.2f} minutes.".format(time_taken_estimate))
    final_exact = time.time()
    time_taken_exact = ((final_exact - initial_exact)/60) # total time taken for full computation
    percentage_time = time_taken_exact/time_taken_estimate *100
    print("This process took {:4.2f} seconds, {:4.2f}% of expected time.".format(time_taken_exact, percentage_time))  
    return intensity
 
def monte(myfunc, aperture_dimensions, x_screen, y_screen, z_screen, N, area): 
    """
    Monte Carlo Method to Solve an integral
    ----------
    myfunc : function; Integrand
    aperture_dimensions : tuple; limits of aperture
    (x_screen, y_screen, z_screen): float; coordinate on the screen
    N : int; number of MC points
    R : float; radius of circular aperture (can be set to None for some apertures)
    -------
   Returns (value, error) - value of the integral, error on the value of the integral
    """
    (xmin_aperture, xmax_aperture, ymin_aperture, ymax_aperture) = aperture_dimensions
    values = np.zeros(N, dtype = complex) # empty array for values of integrand for points on the aperture
    x_aperture = np.random.uniform(xmin_aperture,xmax_aperture, N) # x values across the aperture
    y_aperture = np.random.uniform(ymin_aperture,ymax_aperture, N) # y values across the aperture
    for i in range(N):    
        values[i]= myfunc(y_aperture[i], x_aperture[i], y_screen, x_screen, z_screen, R)
    mean = values.sum()/N # <f>
    meansq = (values**2).sum()/N # <f^2>
    
    integral = area*mean # A x <f>
    error = area*np.sqrt(abs(meansq - (mean**2))/N) # f x sqrt(<f^2>-<f>^2 /N)
    return integral, error    
 
def solve_MC(aperture_function, num_points, xvals, yvals, aperture_dimensions, z_screen, num_samples, R, area): 
    """
    Uses Monte Function and iterates across the screen calculating a value for 
    intensity and its error at each point, comnbining it into an array.
    ----------
    aperture_function : Function; assigns value for input (x,y) on the aperture
                        based on choice of aperture.
    num_points : integer; number of axis divisions
    xvals, yvals : array; description of grid values for plot.
    aperture_dimensions : tuple; extremal values of aperture span.
    z_screen : float; aperture-screen distance.
    num_samples : integer; number of random points to generate on aperture
    R : float (or None); key dimension (radius) of aperture.
    area : float; area of aperture
    -------
    Returns array of intensities and errors.
    """
    initial_exact = time.time()
    intensity = np.zeros((num_points,num_points))
    errors = np.zeros((num_points,num_points))

    for i in range(num_points): # loop across all x-values of plot (screen)
        x_screen = xvals[i] # value of x on the screen
        for j in range(num_points):
            y_screen = yvals[j]
            if (i,j) == (0,0): # first runthrough loop
                initial_sample = time.time()
            integral, error = monte(aperture_function, aperture_dimensions, x_screen, y_screen, z_screen, num_samples, area)
        
            value_intensity = integral*integral.conjugate()
            intensity[i][j] = value_intensity.real
        
            value_error = np.sqrt((2*integral.real*error.real)**2 +(2*integral.imag*error.imag)**2)
            errors[i][j] = value_error
            if (i,j) == (0,0): # first runthrough loop
                final_sample = time.time()
                time_taken_estimate = (num_points**2)*(final_sample - initial_sample) # estimate for time taken based on first runthough
                print("This should take {:4.2f} seconds.".format(time_taken_estimate))  
    
    final_exact = time.time()
    time_taken_exact = (final_exact - initial_exact) # actual time taken for complete computation
    percentage_time = time_taken_exact/time_taken_estimate *100
    print("This process took {:4.2f} seconds, {:4.2f}% of expected time.".format(time_taken_exact, percentage_time)) 
    return intensity, errors
    
def plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title):
    """
    Plots a 2D diffraction pattern.
    ----------
    intensity : array; intensity values to be plotted.
    xvals_screen, yvals_screen : array; describe grid of values
    z_screen : float; aperture-screen distance.
    title : string; title text for the graph.
    -------
    Returns plot of diffraction.
    """
    # Plots realtive intensity on a scale of 0-1 so that comparison across multiple plots is straight-forward
    num_points = len(xvals_screen)
    rel_intensity = np.zeros((num_points, num_points)) # plots intensity on a scale with max 1.0, easier to compare
    max_value = intensity.max()
    for i in range(num_points):
        for j in range(num_points):
            rel_intensity[i][j] = intensity[i][j]/max_value


    plt.imshow(rel_intensity, vmin = 0.0, vmax = 1.0*rel_intensity.max(), extent = screen_dimensions, \
                origin = "lower", cmap = "nipy_spectral_r")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(str(title), pad = 10) 
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# APERTURE FUNCTIONS FOR SCIPY
def ymin_circ_aperture(xmin_aperture):
    """
    Returns lower limit of circular aperture for y as a function of xp1
    ----------
    xmin_aperture : lower x limit of aperture
    """
    R = 0.5e-5 # m, radius of circular aperture
    ymin_aperture = -1*np.sqrt(R**2 - xmin_aperture**2)
    return ymin_aperture

def ymax_circ_aperture(xmax_aperture):
    """
    Returns upper limit of circular aperture for y as a function of xp2 
    ----------
    xmax_aperture : upper x limit of aperture
    """
    R = 0.5e-5 # m, radius of circular aperture
    ymax_aperture = np.sqrt(R**2 - xmax_aperture**2)
    return ymax_aperture
 
# APERTURE FUNCTIONS FOR MONTE CARLO METHOD - ensure all have the same input
def MC_circ(y_aperture,x_aperture,y_screen,x_screen,z_screen,R): 
    """
    Value of function from (x,y) on the aperture at point (x,y,z) on the screen
    ----------
    x_aperture, y_aperture : float; coordinates of point selected on aperture
    x_screen, y_screen, z_screen : float; coordinates of point on screen
    R : float; base length of triangle
    -------
    function value
    """
    r = np.sqrt(x_aperture**2 + y_aperture**2)
    if r>R: # if the point selected is outside the circle described it is given value 0.0
        return 0.0
    else: # if point is within circle the integrand function values are assigned
        func_real = Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen) 
        func_imag = Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen)
        func_value = complex(func_real,func_imag) 
        return func_value
    
def MC_tri(y_aperture,x_aperture,y_screen,x_screen,z_screen,R):
    """
    Value of function at points on the aperture, based on a triangular aperture shape
    ----------
    x_aperture, y_aperture : float; coordinates of point selected on aperture
    x_screen, y_screen, z_screen : float; coordinates of point on screen
    R : float; base length of triangle
    -------
    function value
    """
    # if value is in triangle assign integrand values, if not return 0.0
    if x_aperture>0:
        if y_aperture>((-2*x_aperture)+R): 
            return 0.0
        elif y_aperture<((-2*x_aperture)+R): 
            func_real = Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen) 
            func_imag = Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen)
            func_value = complex(func_real,func_imag) 
            return func_value
    elif x_aperture<0:
        if y_aperture>((2*x_aperture)+R): 
            return 0.0
        elif y_aperture<((2*x_aperture)+R): 
            func_real = Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen) 
            func_imag = Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen)
            func_value = complex(func_real,func_imag) 
            return func_value

def MC_rectangle(y_aperture,x_aperture,y_screen,x_screen,z_screen,R):
    """
    Value of the function of (x,y) from the aperture at (x,y,z) on the screen
    ----------
    x_aperture, y_aperture : float; point on the aperture
    x_screen, y_screen, z_screen: float; point on the screen
    R : included for symmetry across MC aperure functions
    -------
    returns value of function.
    """
    # here since aperture is same as space for possible values being selected, no need for if statements.
    func_real = Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen) 
    func_imag = Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen)
    func_value = complex(func_real,func_imag) 
    return func_value

def MC_flower(y_aperture,x_aperture,y_screen,x_screen,z_screen,R):
    """
    Value of function for a point on the flower shaped aperture (x',y') is returned.
    ----------
    x_aperture, y_aperture : float; point on the aperture
    x_screen, y_screen, z_screen: float; point on the screen
    R : included for symmetry across MC aperure functions
    -------
    returns value of function.
    """
    # if point selected lies within the flower shape assign value of integrand function
    # if not return 0.0
    x_scale = x_aperture*(10**4)
    y_scale = y_aperture*(10**4)
    if y_scale>0:
        if float((x_scale**2+y_scale**2)**3)< (x_scale**2)*(y_scale**3):
            func_real = Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen) 
            func_imag = Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen)
            func_value = complex(func_real,func_imag) 
            return func_value
        else: 
            return 0.0
    elif y_scale<0:
        if float((x_scale**2+y_scale**2)**3)< -1*(x_scale**2)*(y_scale**3):
            func_real = Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen) 
            func_imag = Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen)
            func_value = complex(func_real,func_imag) 
            return func_value
        else: 
            return 0.0

def MC_heart(y_aperture,x_aperture,y_screen,x_screen,z_screen,R):
    """
    Value of function for a point on the heart shaped aperture (x',y') is returned.
    ----------
    x_aperture, y_aperture : float; point on the aperture
    x_screen, y_screen, z_screen: float; point on the screen
    R : included for symmetry across MC aperure functions
    -------
    returns value of function.
    """
    x_scale = x_aperture*(10**5)
    y_scale = y_aperture*(10**5)
    if float((x_scale**2+(y_scale/2)**2-0.5)**3)< (x_scale**2)*(y_scale**3):
        func_real = Fresnel_2d_real(y_aperture, x_aperture, y_screen, x_screen, z_screen) 
        func_imag = Fresnel_2d_imag(y_aperture, x_aperture, y_screen, x_screen, z_screen)
        func_value = complex(func_real,func_imag) 
        return func_value
    else: 
        return 0.0

# CONSTANTS ===================================================================

# Properties of the light wave
WAVELENGTH = 589e-9 # m, wavelength of wave
k = 2*np.pi/WAVELENGTH # m^(-1), wavenumber


#  Introducing choice variables
choice_initial = 0 # 1d or 2d diffraction
choice_1D = 0 # fresnel or fraunhofer pattern
choice_2D = 0 # monte carlo or scipy
choice_scipy = 0 # type of aperture for scipy
choice_mc = 0 # type of aperture for monte carlo

# Text to print to show user options
options_initial = "Select the type of diffraction you would like to see:\n 1) 1D difffraction  \n 2) 2D diffraction \n 3) Quit Program.\n-> "
options_1D = "Select the type of 1D diffraction you would like to see:\n 1) Fresnel  (near-field):  aperture width = 2e-4m; screen distance = 0.005m\n 2) Fraunhofer (far-field): aperture width = 2e-5m; screen distance = 0.05m\n 3) Go back.\n-> "
options_2D = "Select how you would like your pattern to be generated:\n 1) Scipy numerical integration (dblquad)\n 2) Monte Carlo Method\n 3) Go back.\n-> "
options_scipy = "Select the shape of the aperture:\n 1) Rectangle \n 2) Circular\n 3)Go back\n ->" 
options_mc = "Select the shape of the aperture:\n 1) Rectangle\n 2) Circle\n 3) Triangle\n 4) Flower\n 5) Heart\n 6) Go back.\n-> " 



print("DIFFRACTION PATTERN PRINTER")
print(" You will be asked a series of questions,")
print(" which will allow you to choose the type of pattern you would like to see:")
print("==============================")
while choice_initial != 3: # while user doesn't want to quit program
    choice_initial = int(input(options_initial))
    if choice_initial == 1:#===================================================
        print("==============================")
        while choice_1D != 3:
            # 1D Diffraction Pattern
            choice_1D = int(input(options_1D)) # user chooses type of 1D diff.
            
            num_points = 200 # number of divisions along axes
            epsrel = epsabs = 1e-6 # tolerances for integration method
            print("==============================")
            if choice_1D == 1:
                print("You have selected to see a 1D Fresnel (near field) diffraction pattern.")
                
                xmax_screen = 0.5e-3 # m, maximum x-coordinate of plot
                z_screen = 5.0e-3 #m, distance between aperture and screen
                
                xmax_aperture = 1.0e-4 # m, extremal length of aperture
                aperture_dimensions = (-1*xmax_aperture,xmax_aperture ,-1*xmax_aperture, xmax_aperture) # m
                
                xvals_screen = np.linspace(-1*xmax_screen,xmax_screen, num_points) 
                # array of xvalues to plot intensity for
                
                # Solve Integral
                intensity, perc_error = solve_1D(xvals_screen, num_points, aperture_dimensions, epsrel, epsabs, z_screen)
                # Plot results
                title = "1D Fresnel Diffraction from a 2D square aperture({:.1e}m)".format(xmax_aperture*2)
                plot_1D(xvals_screen, intensity, perc_error, xmax_aperture, title)
                
            elif choice_1D == 2:
                print("You have selected to see a 1D Fraunhofer (far field) diffraction pattern.")
                diffraction_type = "Fraunhofer"
                
                xmax_screen = 5.0e-3 # m, maximum x-coordinate of plot
                z_screen = 5.0e-2 #m, distance between aperture and screen
                
                xmax_aperture = 1.0e-5
                aperture_dimensions = (-1*xmax_aperture,xmax_aperture ,-1*xmax_aperture, xmax_aperture) # m
    
                xvals_screen = np.linspace(-1*xmax_screen,xmax_screen, num_points) # array of xvalues to plot intensity for
                
                # Solve Integral
                intensity, perc_error = solve_1D(xvals_screen, num_points, aperture_dimensions, epsrel, epsabs, z_screen)
                # Plot results
                title = "1D Fraunhofer Diffraction from a 2D square aperture({:.1e}m)".format(xmax_aperture*2)
                plot_1D(xvals_screen, intensity, perc_error, xmax_aperture, diffraction_type)
                
    elif choice_initial == 2:#=================================================
        print("==============================")
        # choose shape of aperture for scipy integration
        while choice_2D != 3:
            print("==============================")
            choice_2D = int(input(options_2D))
            if choice_2D == 1:
                # dbl quad scipy method
                while choice_scipy != 3:
                    choice_scipy = int(input(options_scipy)) # user chooses type of aperture
                    num_points = 150
                    epsrel = epsabs = 1.0e-10 # tolerance
                   
                    # Screen dimensions - shared for all Scipy plots
                    ymax_screen = 0.01 # m, maximum x and y-coordinate on screen/plot
                    xmax_screen = ymax_screen # m, scaled to accomodate aperture ratio
                    screen_dimensions = (-1*xmax_screen, xmax_screen, -1*ymax_screen, ymax_screen)
                    z_screen = 0.10 # m, screen distance
                    xvals_screen = np.linspace(-1*xmax_screen, xmax_screen, num_points)
                    yvals_screen = np.linspace(-1*ymax_screen, ymax_screen, num_points)
                    
                    if choice_scipy == 1:
                        print("==============================")
                        print("You have selected a 2D diffraction pattern from a Rectangular aperture.")
                        scale = float(input(" State the ratio of length vs height of the rectangular aperture (use 1 for a suare aperture): "))
                        
                        
                        # Aperture dimensions - RECTANGULAR
                        xmax_aperture = 1.0e-5 * scale # m, half the width of the aperture
                        ymax_aperture = 1.0e-5 # m, half the height of the aperture
                        aperture_dimensions = (-1*xmax_aperture, xmax_aperture, -1*ymax_aperture, ymax_aperture)
                        print("You have selected an aperture of dimension {:.1e}m by {:.1e}m.".format(2*xmax_aperture, 2*ymax_aperture))
                        
                        # Solve the integral
                        intensity = solve_2D(num_points, xvals_screen, yvals_screen, z_screen, aperture_dimensions, epsrel, epsabs)
                        # Plot the results
                        title = "Rectangular Diffraction({:.1e}x{:.1e})(z = {:4.2f}m).".format(2*xmax_aperture, 2*ymax_aperture, z_screen)
                        plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title)
                    
                    
                    elif choice_scipy == 2:
                        print("==============================")
                        print("You have selected a 2D diffraction pattern from a circular aperture.")
                        
                        # Aperture Dimensions - CIRCULAR
                        R = 0.5e-5 # m, radius of circular aperture
                        xmin_aperture = -1*R
                        xmax_aperture = R
                        aperture_dimensions = (xmin_aperture, xmax_aperture, ymin_circ_aperture, ymax_circ_aperture)
                        
                        # Solve the integral
                        intensity = solve_2D(num_points, xvals_screen, yvals_screen, z_screen, aperture_dimensions, epsrel, epsabs)
                        # Plot the results
                        title = "Circular Diffraction (R = {:.1e})(z = {:.1e}).".format(R, z_screen)
                        plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title)
                    
   
            elif choice_2D == 2:#==============================================
                 print("==============================")
                 # same values for all Monte Carlo plots
                 num_points = 300 # number of points across each axis / divide screen into a grid
                 num_samples = 250 # sample points taken across aperture for MC method
            
                 while choice_mc != 6:
                     choice_mc = int(input(options_mc)) # user chooses aperture type
                     if choice_mc == 1: 
                         print("You have selected a rectangular diffraction pattern using the Monte Carlo method.")
                         scale = float(input("State the ratio of length vs height of the rectangular aperture (select 1 for a square aperture): "))
                         
                        
                         # Screen dimensions
                         ymax_screen = 0.01 # m, maximum x and y-coordinate on screen/plot
                         xmax_screen = ymax_screen # m 
                         z_screen = 1.0e-1 # m, screen distance
                         screen_dimensions = (-1*xmax_screen, xmax_screen, -1*ymax_screen, ymax_screen)
                         xvals_screen = np.linspace(-1*xmax_screen, xmax_screen, num_points)
                         yvals_screen = np.linspace(-1*ymax_screen, ymax_screen, num_points)
                        
                         # Aperture dimensions - RECTANGULAR
                         R = None
                         xmax_aperture = 1.0e-5 * scale # m, half the width of the aperture
                         ymax_aperture = 1.0e-5 # m, half the height of the aperture
                         aperture_dimensions = (-1*xmax_aperture, xmax_aperture, -1*ymax_aperture, ymax_aperture)
                         area = 4*xmax_aperture*ymax_aperture
                         
                         print("You have selected an aperture of dimension {:.1e}m by {:.1e}m.".format(2*xmax_aperture, 2*ymax_aperture))
                        
                         # Solve the integral
                         intensity, errors = solve_MC(MC_rectangle, num_points, xvals_screen, yvals_screen, aperture_dimensions, z_screen, num_samples, R, area)
                         # Plot the results
                         title = "Rectangular Diffraction({:.1e}x{:.1e})(z = {:4.2f}m).".format(2*xmax_aperture, 2*ymax_aperture, z_screen)
                         plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title)
                         
                         
                     elif choice_mc == 2:
                         print("You have selected a 2D circular diffraction pattern using the Monte Carlo Method.")
                         
                         # Screen dimensions
                         xmax_screen = ymax_screen = 2.0e-3 # m
                         screen_dimensions = (-1*xmax_screen, xmax_screen, -1*ymax_screen, ymax_screen)
                         z_screen = 1.0e-2 # m
                         xvals_screen = np.linspace(-1*xmax_screen, xmax_screen, num_points)
                         yvals_screen = np.linspace(-1*ymax_screen, ymax_screen, num_points)
                         
                         # Aperture Dimensions - CIRCLE
                         R = 5.0e-6 # m, radius of circular aperture
                         area_aperture = np.pi*R**2
                         xmax_aperture = ymax_aperture = R # m
                         aperture_dimensions = (-1*xmax_aperture, xmax_aperture, -1*ymax_aperture, ymax_aperture)
                         
                         # Solve the integral
                         intensity, errors = solve_MC(MC_circ, num_points, xvals_screen, yvals_screen, aperture_dimensions, z_screen, num_samples, R, area_aperture)
                         
                         # Plot the results
                         title = "Circular Diffraction using Monte Carlo method (R = {:.1e}m)(z = {:.1e}m)".format(R, z_screen)
                         plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title)
                         
                         
                     elif choice_mc == 3:
                         print("You have selected a 2D triangular diffraction pattern using the Monte Carlo Method.")
                         
                         # Screen dimensions
                         xmax_screen = ymax_screen = 4.0e-3 # m
                         screen_dimensions = (-1*xmax_screen, xmax_screen, -1*ymax_screen, ymax_screen)
                         z_screen = 1.0e-2 # m
                         xvals_screen = np.linspace(-1*xmax_screen, xmax_screen, num_points)
                         yvals_screen = np.linspace(-1*ymax_screen, ymax_screen, num_points)
                         
                         # Aperture Dimensions - TRIANGLE
                         R = 1.0e-5 # m, half width of square aperture fits in.
                         area = 2*(R**2) # m^2 (2R*2R/2) = base*height/2
                         xmax_aperture = ymax_aperture = R # m
                         aperture_dimensions = (-1*xmax_aperture, xmax_aperture, -1*ymax_aperture, ymax_aperture)
                         
                         # Solve the integral
                         intensity, errors = solve_MC(MC_tri, num_points, xvals_screen, yvals_screen, aperture_dimensions, z_screen, num_samples, R, area)
                         
                         # Plot the results
                         title = "Triangular Diffraction using Monte Carlo method (Height = {:.1e}m)(z = {:.1e}m)".format(R, z_screen)
                         plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title)
                         
                     elif choice_mc == 4:
                         print("You have selected a 2D Flower diffraction pattern using the Monte Carlo Method.")
                         
                         # Screen Dimensions
                         xmax_screen = ymax_screen = 5.0e-3 # m
                         screen_dimensions = (-1*xmax_screen,xmax_screen,-1*ymax_screen, ymax_screen)
                         z_screen = 0.05 # m, screen distance
                         xvals_screen = np.linspace(-1*xmax_screen, xmax_screen, num_points)
                         yvals_screen = np.linspace(-1*ymax_screen, ymax_screen, num_points)
                         
                         # Aperture Dimensions - FLOWER
                         R = 1.5e-5 # m, half width of square aperture fits in.
                         area = 1.0 # m^2 - set to 1 since intensity is relative.
                         xmax_aperture = ymax_aperture = R # m
                         aperture_dimensions = (-1*xmax_aperture, xmax_aperture, -1*ymax_aperture, ymax_aperture)
                         
                         # Solve the integral
                         intensity, errors = solve_MC(MC_flower, num_points, xvals_screen, yvals_screen, aperture_dimensions, z_screen, num_samples, R, area)
                         
                         # Plot the results
                         title = "Diffraction from a flower shaped aperture using Monte Carlo method (z = {:.1e}m)".format(z_screen)
                         plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title)
                         
                     elif choice_mc == 5:
                         print("You have selected a 2D heart Diffraction pattern using the Monte Carlo Method.")
                         
                         # Screen Dimensions
                         xmax_screen = ymax_screen = 4.0e-3 # m
                         screen_dimensions = (-1*xmax_screen, xmax_screen, -1*ymax_screen, ymax_screen)
                         z_screen = 0.05 # m, aperture screen distance
                         xvals_screen = np.linspace(-1*xmax_screen, xmax_screen, num_points)
                         yvals_screen = np.linspace(-1*ymax_screen, ymax_screen, num_points)
                         
                         # Aperture Dimensions
                         R = None
                         area = 1.0 # m^2, area is scaling factor so no impact on relative intensity
                         xmin_aperture = ymin_aperture = -1.5e-5 # m
                         xmax_aperture = -1*xmin_aperture
                         ymax_aperture =  3.5e-5 # m
                         aperture_dimensions = (xmin_aperture, xmax_aperture, ymin_aperture, ymax_aperture)
                        
                         # Solve the integral
                         intensity, errors = solve_MC(MC_heart, num_points, xvals_screen, yvals_screen, aperture_dimensions, z_screen, num_samples, R, area)
                         
                         # Plot the results
                         title = "Diffraction from a heart shaped aperture using Monte Carlo method (z = {:.1e}m)".format(z_screen)
                         plot_2D(intensity, xvals_screen, yvals_screen, screen_dimensions, z_screen, title)
