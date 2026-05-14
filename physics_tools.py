import numpy as np
import sympy as sp
from math import floor, log10
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import AutoMinorLocator 
from matplotlib.ticker import ScalarFormatter 


def propagate_error(formula, variables_map, errors_map, sig_figs):
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


def plot_and_save_regression_with_errors(x, y, x_err, y_err, x_label, y_label, filename="regression_analysis.png"):
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Perform linear regression using linregress
    # res contains: slope, intercept, rvalue, pvalue, stderr, intercept_stderr
    res = linregress(x, y)
    
    slope = res.slope
    intercept = res.intercept
    slope_err = res.stderr
    intercept_err = res.intercept_stderr
    r_squared = res.rvalue**2
    
    # Generate points for the regression line
    x_line = np.linspace(0.95*np.min(x), 1.05*np.max(x), 100)
    y_line = slope * x_line + intercept
    
    # Create the plot
    #fig,ax = plt.subplots(figsize=(12, 7))
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    # Plot data points with error bars
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', capsize=5, 
                 label='Data Points', color='black', ecolor = 'grey', markersize=5, alpha=0.8)
    
    # Format the legend text with uncertainty: "slope ± error" and "intercept ± error"
    # We use LaTeX formatting for the plus-minus sign (\pm)
    sign = "+" if intercept >= 0 else "-"
    label_text = (f"Fit: $y = ({slope:.3f} \pm {slope_err:.3f})x$ "
                  f"$ {sign} ({abs(intercept):.3f} \pm {intercept_err:.3f})$\n"
                  f"$R^2 = {r_squared:.3f}$")
    
    # Plot the solid regression line
    plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label=label_text)
    
    # Aesthetics
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # 2. Main major grid lines (darker, dashed)
    ax.grid(True, which="major", color="grey", linestyle="--", alpha=0.7)

    # 3. Finer minor grid lines (lighter, dotted, high density)
    ax.grid(True, which="minor", color="lightgrey", linestyle=":", alpha=0.7)
    # 4. minor grid visual parameters
    ax.tick_params(
        which="minor",
        direction="in",
        length=4,
        width=1.0,
        color="black",
        top=True,
        right=True,
        labelsize="x-small",
    )
    #5. adding scalar values for minor ticks 
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())

    #plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize='medium', frameon=True)
    
    # Save the figure
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Plot saved as {filename}")
    plt.show()

# --- Test Execution ---
