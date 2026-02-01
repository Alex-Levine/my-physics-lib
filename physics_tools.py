import numpy as np
import sympy as sp

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
