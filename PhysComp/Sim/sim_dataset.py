import numpy as np
import matplotlib.pyplot as plt

def sim_dataset(x_size,y_size,max_intensity, gaussian_noise, lam):
# Detector parameters
    nx, ny, nz = x_size, y_size, 360  # pixels x, pixels y, azimuthal bins

    # Azimuthal angle (0 to 2*pi)
    angles = np.linspace(0, 2*np.pi, nz)

    # Shifting sinsuisoid spatially
    phase_shift_x = np.linspace(0, 2*np.pi, nx)[:, np.newaxis, np.newaxis]
    phase_shift_y = np.linspace(0, 2*np.pi, ny)[np.newaxis, :, np.newaxis]

    # Amplitudes
    amp1 = np.random.randint(0,100)
    amp2 = np.random.randint(0,100)
    # Phases
    phase1 = np.random.randint(0,2*np.pi)
    phase2 = np.random.randint(0,2*np.pi)

    # Creating Base signal pattern
    signal_pattern = amp * np.sin(angles[np.newaxis, np.newaxis, :] + phase_shift_x + phase_shift_y + phase1) + amp2 * np.sin(2*angles[np.newaxis, np.newaxis, :] + phase_shift_x + phase_shift_y + phase2)



    ## Generating noise to simulate realistic dataset
    # Gaussian noise
    gaussian_noise = np.random.randn(nx, ny, nz) * gaussian_noise
    # Poisson noise
    poisson_noise = np.random.poisson(lam=lam, size=(nx, ny, nz))

    # Combine signal and noise to create final data set
    detector_data = signal_pattern + gaussian_noise + poisson_noise

    # Clip to valid intensity range
    detector_data = np.clip(detector_data, 0, max_intensity)
    
    return detector_data, signal_pattern
