import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def sin_model(x, a, b, c, d, e, f, g):
    return a*np.sin(b*np.deg2rad(x) + c) + d + e*np.cos(f*np.deg2rad(x) + g)


def periodic_fit_plot(x_variable, y_variable, initial, plot=False):

    params, covariance = optimize.curve_fit(
        sin_model, x_variable, y_variable, p0=initial, maxfev=10000
    )

    errs = np.sqrt(np.diag(covariance))

    residuals = y_variable - sin_model(x_variable, *params)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_variable - np.mean(y_variable))**2)
    r_squared = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

    ## Rounding frequencies, ensuring single values at 0/360
    params_sin[1] = np.round(params_sin[1], decimals=0)
    params_sin[4] = np.round(params_sin[4], decimals=0)
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2,1]},
            sharex=True, figsize=(10,6)
        )

        dummy = np.linspace(np.min(x_variable), np.max(x_variable), 360)

        ax1.scatter(x_variable, y_variable, label="Raw Data")
        ax1.plot(dummy, sin_model(dummy, *params), 'r', label="Fit")
        ax1.legend(title=f"R² = {r_squared:.4f}")
        ax1.grid()

        ax2.plot(x_variable, residuals, 'or')
        ax2.set_xlabel("Angle")
        ax2.set_ylabel("Residuals")
        ax2.grid()

        plt.tight_layout()
        plt.show()

    return params, errs, r_squared
