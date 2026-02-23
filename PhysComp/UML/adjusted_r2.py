import numpy as np

def adjusted_r2(y_true, y_pred, n_params):
    """
    Compute adjusted R-squared for a model.

    Parameters
    ----------
    y_true : array - True Data
    y_pred : array - Fit Predictions
    n_params : int - Number of parameters in fit

    Returns
    -------
    r2_adj : float - Adjusted R-squared.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    
    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Standard R^2
    r2 = 1 - ss_res / ss_tot
    
    # Adjusted R^2
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    
    return r2_adj
