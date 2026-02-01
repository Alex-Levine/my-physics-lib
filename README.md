# r_s, m_s, v_s, I_s = sp.symbols('r m v I')
# my_f = r_s * m_s * v_s / (I_s + m_s * r_s**2)

# # Define your data maps
# vals = {
#     r_s: 13.9e-2,
#     m_s: 66.8e-3,
#     v_s: np.array([1.44, 1.21, 1.64, 1.01]), # Works with arrays!
#     I_s: 0.011
# }

# errs = {
#     r_s: 1e-3,
#     m_s: 0.01e-3,
#     v_s: np.array([0.01]*4),
#     I_s: 0.001
# }

# # Run the function
# w_vals,dw_result = propagate_error(my_f, vals, errs,3)
# print(w_vals,dw_result)
