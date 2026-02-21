import numpy as np

def sim_dataset(x_size, y_size, max_intensity, gaussian_noise_std, lam):
    """
    Simulate a 3D detector dataset with sinusoidal patterns and noise.

    Parameters:
    - x_size, y_size : int : spatial dimensions
    - max_intensity : float : maximum intensity to clip
    - gaussian_noise_std : float : standard deviation for Gaussian noise
    - lam : float : lambda parameter for Poisson noise

    Returns:
    - detector_data : np.ndarray : final noisy dataset
    - signal_pattern : np.ndarray : base sinusoidal signal
    """

    # Detector parameters
    nx, ny, nz = x_size, y_size, 360  # pixels x, pixels y, azimuthal bins

    # Azimuthal angle (0 to 2*pi)
    angles = np.linspace(0, 2*np.pi, nz)

    # Shifting sinusoid spatially
    phase_shift_x = np.linspace(0, 2*np.pi, nx)[:, np.newaxis, np.newaxis]
    phase_shift_y = np.linspace(0, 2*np.pi, ny)[np.newaxis, :, np.newaxis]

    # Amplitudes
    amp1 = np.random.randint(0, 100)
    amp2 = np.random.randint(0, 100)
    
    # Phases (use uniform random floats)
    phase1 = np.random.randint(0, 2*np.pi)
    phase2 = np.random.randint(0, 2*np.pi)

    # Creating base signal pattern
    signal_pattern = (
        amp1 * np.sin(angles[np.newaxis, np.newaxis, :] + phase_shift_x + phase_shift_y + phase1) +
        amp2 * np.sin(2 * angles[np.newaxis, np.newaxis, :] + phase_shift_x + phase_shift_y + phase2)
    )

    # Generate noise
    gaussian_noise_array = np.random.randn(nx, ny, nz) * gaussian_noise_std
    poisson_noise = np.random.poisson(lam=lam, size=(nx, ny, nz))

    # Combine signal and noise
    detector_data = signal_pattern + gaussian_noise_array + poisson_noise

    # Clip to valid intensity range
    detector_data = np.clip(detector_data, 0, max_intensity)
    
    return detector_data, signal_pattern
