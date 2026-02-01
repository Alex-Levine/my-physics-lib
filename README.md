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
