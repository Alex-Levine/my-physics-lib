import numpy as np
import sympy as sp
from math import floor, log10

def propagate_error(formula, variables_map, errors_map, precision):
    """
   Calculates values and propagated error, rounded to a specific precision.
    
    Parameters:
    - formula: Sympy expression.
    - variables_map: Dict of symbols and numerical values/arrays.
    - errors_map: Dict of symbols and numerical uncertainties.
    - precision: Integer, number of decimal places to round to (default=4).
                     
    Returns:
    - nominal_values: Rounded result of the formula.
    - total_error: Rounded propagated uncertainty.
    """

    # 1. Identify all symbols in the formula
    symbols = list(variables_map.keys())
    
    # This allows us to calculate the nominal values (e.g., 'w')
    formula_func = sp.lambdify(symbols, formula, "numpy")
    
    # 2. Calculate partial derivatives for each symbol
    derivatives = {s: sp.diff(formula, s) for s in symbols}
    
    # 3. Convert symbolic derivatives to fast NumPy functions
    # We use the list of symbols as the argument order for lambdify
    funcs = {s: sp.lambdify(symbols, derivatives[s], "numpy") for s in symbols}
    
    # 4. Prepare the numerical inputs in the correct order
    ordered_values = [variables_map[s] for s in symbols]

    nominal_values = formula_func(*ordered_values)
    
    # 5. Calculate (df/dx * dx)^2 for each variable
    squared_terms = []
    for s in symbols:
        # Evaluate the derivative at the given points
        deriv_value = funcs[s](*ordered_values)
        # Multiply by the error and square it
        term = (deriv_value * errors_map[s])**2
        squared_terms.append(term)
    
    # 6. Sum terms and take the square root
    total_error = np.sqrt(sum(squared_terms))
    
    rounded_values = np.round(nominal_values, precision)
    rounded_error = np.round(total_error, precision)
    
    return rounded_values, rounded_error


def propagate_error_2(formula, variables_map, errors_map, sig_figs):
    """
    Calculates propagated error using symbolic differentiation and 
    rounds results based on experimental physics significant figure rules.
    """
    symbols = list(variables_map.keys())
    
    # 1. Calculate partial derivatives
    derivatives = {s: sp.diff(formula, s) for s in symbols}
    
    # 2. Lambdify the formula and its derivatives
    f_func = sp.lambdify(symbols, formula, "numpy")
    df_funcs = {s: sp.lambdify(symbols, derivatives[s], "numpy") for s in symbols}
    
    # 3. Prepare numerical inputs and handle broadcasting
    max_len = max(len(np.atleast_1d(v)) for v in variables_map.values())
    ordered_values = []
    ordered_errors = []
    
    for s in symbols:
        v = np.atleast_1d(variables_map[s])
        e = np.atleast_1d(errors_map[s])
        if len(v) == 1 and max_len > 1:
            v, e = np.tile(v, max_len), np.tile(e, max_len)
        ordered_values.append(v)
        ordered_errors.append(e)
    
    # 4. Calculate Nominal Values
    nominal_values = f_func(*ordered_values)
    
    # 5. Calculate Propagated Error (Square Root of Sum of Squares)
    squared_error_sum = np.zeros(max_len)
    for i, s in enumerate(symbols):
        deriv_val = df_funcs[s](*ordered_values)
        squared_error_sum += (deriv_val * ordered_errors[i])**2
    
    total_error = np.sqrt(squared_error_sum)

    # 6. Smart Rounding Logic
    def smart_round(val, err):
        if err == 0 or not np.isfinite(err):
            return val, err
        # Calculate the decimal place to round to based on error magnitude
        precision = -int(floor(log10(abs(err)))) + (sig_figs - 1)
        return round(val, precision), round(err, precision)

    # 7. Final Processing
    if max_len == 1:
        return smart_round(nominal_values[0], total_error[0])
    
    rounded_pairs = [smart_round(v, e) for v, e in zip(nominal_values, total_error)]
    return np.array([r[0] for r in rounded_pairs]), np.array([r[1] for r in rounded_pairs])
