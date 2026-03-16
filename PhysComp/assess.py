import numpy as np

def adjusted_r2(y_true, y_pred, n_params):
    """
    Compute adjusted R-squared for a model.

    Parameters
    ----------
    y_true : array-like - True, raw data
    y_pred : array-like - Fit predictions
    n_params : int - Number of parameters in fit

    Returns
    -------
    r2_adj : float - Adjusted R-squared.
    """
    #---- Validation ---------
    if not all(isinstance(v,np.ndarray) for v in (y_true,y_pred)):
        raise TypeError(f"y_true and y_pred should both be numpy arrays")

    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred should both be 1-D and the same size")

    if not isinstance(n_params,int) or n_params < 0:
        raise ValueError(f"n_params must be a positive integer")
    
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
