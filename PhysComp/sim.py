"""
The functions contained within this file are all related to simulating an
azimuthal scattering dataset.

Each function includes a description of the method alongside the required
inputs and outputs. Demonstrations of each function can be found in the 
sim demo notebook in the Demos folder
"""

"""
--------- Required imports for this file ---------
"""
import numpy as np
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

"""
-------- Functions --------
"""
def sim_dataset(x_size: int, y_size: int, bins: int, gaussian_noise_std: float, lam: float, filesave: bool = False, save_path: Optional[str] = None, seed: Optional[int] = None):
    """
    Simulate a 3D detector dataset with azimuthal sinusoidal signals and noise.

    Parameters
    ----------
    x_size : int
        Number of pixels in x dimension.
    y_size : int
        Number of pixels in y dimension.
    bins : int
        Number of azimuthal bins.
    gaussian_noise_std : float
        Standard deviation of Gaussian noise.
    lam : float
        Lambda parameter for Poisson noise.
    filesave : bool, optional
        If True, save dataset to disk.
    save_path : str, optional
        Path to save file if filesave=True.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    detector_data : np.ndarray
        Simulated noisy detector data (nx, ny, nz).
    signal_pattern: np.ndarray
        Ground truth signal without noise.
    params : dictionary
        Dict of parameters used.
    """

    ## Input Validation
    if not all(isinstance(v, int) and v > 0 for v in (x_size, y_size, bins)):
        raise ValueError("x_size, y_size, and bins must be positive integers.")

    if gaussian_noise_std < 0:
        raise ValueError("gaussian_noise_std must be float >= 0")

    if lam < 0:
        raise ValueError("lam must be float >= 0")

    if filesave is True and save_path is None:
        raise ValueError("save_path must be provided if filesave=True")
    
    if seed is not None:
        if not isinstance(seed, int) or not seed > 0:
            raise ValueError("Seed must be positive integer.")
    # --------------------------

    # Generating randomness
    rng = np.random.default_rng(seed)

    # Defining dimension
    nx, ny, nz = x_size, y_size, bins

    # Generating angles, works best with 180,360,480 etc
    angles = np.linspace(0, (nz)/180 * np.pi, nz)

    # Generating phase shift to progate signal 2 dimenstionally
    phase_shift_x = np.linspace(0, 2*np.pi, nx)[:, None, None]
    phase_shift_y = np.linspace(0, 2*np.pi, ny)[None, :, None]

    # Generating random parameters for ground truth
    amp1 = rng.integers(1, 100)
    amp2 = rng.integers(1, 100)

    phase1 = np.deg2rad(rng.integers(1, bins))
    phase2 = np.deg2rad(rng.integers(1, bins))

    offset = amp1 + amp2

    # Ground truth signal generation
    signal_pattern = (amp1 * np.sin(angles[None, None, :] + phase_shift_x + phase_shift_y + phase1) + amp2 * np.sin(2 * angles[None, None, :] + phase_shift_x + phase_shift_y + phase2) + offset)

    # Generating noise based upon inputs
    gaussian_noise = rng.normal(loc=0, scale=gaussian_noise_std, size=(nx, ny, nz))
    poisson_noise = rng.poisson(lam=lam, size=(nx, ny, nz))
    
    # Combining noise and ground truth to generate detector data
    detector_data = signal_pattern + gaussian_noise + poisson_noise

    # File saving if needed
    if filesave:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, detector_data)

    params = {
        "amp1": float(amp1),
        "phase1": float(phase1),
        "amp2": float(amp2),
        "phase2": float(phase2),
        "offset": float(offset),}

    return detector_data, signal_pattern, params
