!curl -O https://raw.githubusercontent.com/Alex-Levine/my-physics-lib/main/physics_tools.py

from physics_tools import propagate_error
import numpy as np
import sympy as sp

r_s, m_s, g_s, t_s, theta_s = sp.symbols('r m g t theta')
my_f = m_s*r_s*g_s*t_s**2/(2*theta_s) - m_s * r_s**2

# Define your data maps
vals = {
    r_s: (25e-3 + 4*5*0.02e-3)/2,
    m_s: np.array([7.22,10.58,15.75,19.12,38.94])*1e-3 + 1.77e-3,
    g_s: 9.7949,
    t_s: np.array([14.16,11.05,9,7.81,6.20]),
    theta_s: 2*np.pi
}

errs = {
    r_s: 0.02e-3/2,
    m_s: 0.01e-3,
    g_s: 0.1e-2,
    t_s: 400e-3,
    theta_s: 5/360 * 2*np.pi
}

# Run the function
I_vals,dI_result = propagate_error(my_f, vals, errs,3)
print(I_vals,dI_result)

#plot_and_save_regression_with_errors example:

x_test = [1.4, 2.43, 3.57, 5.04, 7.91, 7.37, 7.94, 10.96]
y_test = [7.3, 8.92, 20.03, 21.76, 32.71, 28.43, 30.11, 40.79]

plot_and_save_regression_with_errors(
    x=x_test, 
    y=y_test, 
    x_err=0.5, 
    y_err=2, 
    x_label="x axis", 
    y_label="y axis",
    filename="linear_fit_uncertainty.png"
)
