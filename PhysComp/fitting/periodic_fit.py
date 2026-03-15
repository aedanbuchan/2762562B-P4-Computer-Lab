import numpy as np
from scipy import optimize

def periodic(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """Evaluate a two-component periodic model at position(s) x.
    The model is the sum of two sine harmonics plus a vertical offset:

        f(x) = a * sin(x + b) + c * sin(2*(x + d)) + e

    where all angle arguments are converted from degrees to radians
    before evaluation.

    Parameters
    ----------
    x : array-like
        Input position(s) in degrees.
    a : float
        Amplitude of the sin(x) term.
    b : float
        Phase offset of the sin(x) term, in degrees.
    c : float
        Amplitude of the sin(2x) term.
    d : float
        Phase offset of the sin(2x) term, in degrees.
    e : float
        Vertical offset.

    Returns
    -------
    numpy.ndarray
        Model values at each point in *x*."""
    
    ## Input Validation
    if not all(isinstance(v, (int,float)) for v in (a,b,c,d,e)):
        raise ValueError("Params a, b, c, d, e must be single value floats")

    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy ndarray, got {type(x).__name__!r}")

    return (a * np.sin(np.deg2rad(x + b)) + c * np.sin(2.0 * np.deg2rad(x + d)) + e)

def periodic_fit(x_variable: np.ndarray, y_variable: np.ndarray, initial = None,bounds= None):
    """Fit the periodic model to data via curve_fit.

    Uses scipy.optimize.curve_fit to find best-fit parameters [a, b, c, d, e]
    for the model: f(x) = a*sin(x + b) + c*sin(2*(x + d)) + e

    Parameters
    ----------
    x_variable : array-like of shape (N,)
        Independent variable (in degrees). Requires at least 5 points.
    y_variable : array-like of shape (N,)
        Dependent variable. Must be the same length as x_variable.
    initial : list of 5 floats, optional
        Initial parameter guesses [a, b, c, d, e]. Defaults to [30, 5, 5, 0, 70].
    bounds : optional
        Lower and upper bounds per parameter:
        Defaults to ([0, -180, 0, -90, 0], [1e6, 180, 1e6, 90, 1e6]).

    Returns
    -------
    params : numpy.ndarray of shape (5,)
        Best-fit parameters [a, b, c, d, e].
    errors : numpy.ndarray of shape (5,)
        One-sigma uncertainties derived from the covariance matrix diagonal."""
        
    # --- Validate x and y --------------------------------------------------
    x_variable = np.asarray(x_variable, dtype=float)
    y_variable = np.asarray(y_variable, dtype=float)

    if x_variable.ndim != 1 or y_variable.ndim != 1:
        raise ValueError("x_variable and y_variable must both be 1-D arrays.")
    if len(x_variable) != len(y_variable):
        raise ValueError(
            f"x_variable and y_variable must have the same length, "
            f"got {len(x_variable)} and {len(y_variable)}."
        )
    if len(x_variable) < 5:
        raise ValueError(
            f"At least 5 data points are required to fit 5 parameters, "
            f"got {len(x_variable)}."
        )

    # --- Validate bounds and initial ---------------------------------------
    if bounds is not None:
        if not (isinstance(bounds, tuple) and len(bounds) == 2
                and len(bounds[0]) == 5 and len(bounds[1]) == 5):
            raise ValueError(
                "bounds must be a 2-tuple of length-5 sequences: "
            )
        if any(lo >= hi for lo, hi in zip(bounds[0], bounds[1])):
            raise ValueError("Each lower bound must be less than its upper bound.")

    if initial is not None:
        if len(initial) != 5 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                        for v in initial):
            raise ValueError("initial must be a list of 5 values.")

    if bounds is not None:
        if any(v < lo or v > hi for v, lo, hi in zip(initial, bounds[0], bounds[1])):
            raise ValueError(f"Initial guesses {list(initial)} contain values outside the "f"specified bounds {bounds}.")

    # --- Defaults ----------------------------------------------------------
    b0 = bounds if bounds is not None else ([0, -180, 0, -90, 0], [1e6, 180, 1e6, 90, 1e6])
    p0 = initial if initial is not None else [30, 5, 5, 0, 70]

    # --- Fit ---------------------------------------------------------------
    try:
        params_sin, params_sin_covariance = optimize.curve_fit(
            periodic, x_variable, y_variable, p0=p0, bounds=b0, maxfev=30000
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "curve_fit did not converge within 30 000 function evaluations. "
            "Try better initial guesses, relaxed bounds, or more data points. "
            f"Original error: {exc}"
        ) from exc

    errs = np.sqrt(np.diag(params_sin_covariance))
    return params_sin, errs

def periodic_fit_whole(dataset: np.ndarray, initial: list, bounds= None, iterative_fitting = False, degrees = 360) -> tuple[np.ndarray, np.ndarray]:
    
    """Fit the periodic model to every pixel in a 3-D dataset.

    Iterates over all (x, y) spatial pixels in dataset, fits
    'periodic` to the angular profile at each pixel using 'perodic_fit'.

    Parameters
    ----------
    dataset : numpy.ndarray of shape (nx, ny, nz)
        3-D array where the first two axes are spatial and the third axis
        contains the angular intensity profile sampled at evenly-spaced
        degrees equal to 'degrees'.
    initial : list of 5 floats
        Initial parameter guesses [a, b, c, d, e] for the first fit.
        When iterative_fitting is True, subsequent pixels reuse the
        previous pixel's best-fit parameters as the new initial guess.
    bounds : 2-tuple of sequences, optional
        Parameter bounds passed directly to :func:`periodic_fit`.
        See that function for the expected format.
    iterative_fitting : bool, optional
        If True, the best-fit parameters from each pixel are used as the
        initial guess for the next, improves computational time needed by ~30%.

    Returns
    -------
    params_reshaped : numpy.ndarray of shape (nx, ny, 5)
        Best-fit parameters [a, b, c, d, e] for every pixel.
    errors_reshaped : numpy.ndarray of shape (nx, ny, 5)
        Corresponding one-sigma uncertainties for every pixel.
    """
    # --- Validate dataset --------------------------------------------------
    if not isinstance(dataset, np.ndarray) or dataset.ndim != 3:
        raise ValueError(
            f"dataset must be a 3-D numpy array of shape (nx, ny, nz), "
            f"got {type(dataset).__name__} with "
            f"{'shape ' + str(dataset.shape) if isinstance(dataset, np.ndarray) else 'no shape'}."
        )
    if dataset.shape[2] != degrees:
        raise ValueError(
            f"The third axis of dataset must have exactly {degrees} elements "
            f", got {dataset.shape[2]}."
        )
    if not isinstance(iterative_fitting, bool):
        raise TypeError(
            f"iterative_fitting must be a bool, got {type(iterative_fitting).__name__!r}."
        )
    if bounds is not None:
        if any(v < lo or v > hi for v, lo, hi in zip(initial, bounds[0], bounds[1])):
            raise ValueError(f"Initial guesses {list(initial)} contain values outside the "f"specified bounds {bounds}.")

    # --- Setup -------------------------------------------------------------
    degrees = np.linspace(0, degrees,degrees)
    nx, ny = dataset.shape[0], dataset.shape[1]
    params_array = []
    errors_array = []
    current_params = initial

    # --- Fit every pixel ---------------------------------------------------
    with tqdm(total=nx * ny, desc="Processing slices") as pbar:
        for x in range(nx):
            for y in range(ny):
                p0 = current_params if iterative_fitting else initial
                current_params, errors = periodic_fit(
                    degrees, dataset[x, y, :], initial=p0, bounds=bounds
                )
                params_array.append(current_params)
                errors_array.append(errors)
                pbar.update(1)

    # --- Reshape and return ------------------------------------------------
    params_reshaped = np.array(params_array).reshape(nx, ny, 5)
    errors_reshaped = np.array(errors_array).reshape(nx, ny, 5)
    return params_reshaped, errors_reshaped
