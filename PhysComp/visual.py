"""
The functions contained within this file are all related to assessing 
the visualastion of a signal or dataset.

Each function includes a description of the method alongside the required
inputs and outputs. Demonstrations of each function can be found in the 
visual demo notebook in the Demos folder
"""

"""
--------- Required imports for this file ---------
"""

import matplotlib.pyplot as plt
import numpy as np
import PhysComp.fitting as fit

"""
-------- Functions ---------
"""

def visualise_dataset(raw,decomped,degree,colour_map="magma"):
    """
    Allows visualsation of a single degree slice for raw and decomposed data
    
    Parameters
    ----------
    raw: array-like
        Raw parameter dataset, shape (x, y, n_params), second output of fit.periodic_fit_whole
    decomped: array-like
        Decomposed parameter dataset, shape (x, y, n_params), first output of fit.periodic_fit_whole
    degree: int
        Degree slice to show
    colour_map: str
        Chosen colour map for visualisation, defaults to "magma"
    
    Returns
    -------
    Plot of single slice of raw and decomposed data
        
    """
    
    cmap= colour_map
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    im0 = axes[0].imshow(raw[:,:,degree], cmap=cmap)

    plt.suptitle(f"Single Degree Slice at {degree} of Dataset Before and After Decomposition", fontweight='bold', fontsize=13, y=0.85)

    axes[0].set_title("Raw Data", fontsize=11, pad=8)

    im1 = axes[1].imshow(decomped[:,:,degree], cmap=cmap)
    axes[1].set_title("Decomposed Data", fontsize=11, pad=8)
    
    for ax in axes:
        ax.set_xlabel("Pixels", fontsize=10)
        ax.set_ylabel("Pixels", fontsize=10)
        ax.tick_params(labelsize=9)

    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.03)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.03)
    cb0.set_label("Intensity", fontsize=10)
    cb1.set_label("Intensity", fontsize=10)
    cb0.ax.tick_params(labelsize=9)
    cb1.ax.tick_params(labelsize=9)

    plt.tight_layout()
    
def visualise_params(raw, decomped, num, angle_map = None, amp_map = None, angle_amp = None):
    """
    Allows visualsation of fitting parameters map calculated for raw and decomposed data
    
    Parameters
    ----------
    raw: array-like
        Raw parameter dataset, shape (x, y, n_params), second output of fit.periodic_fit_whole
    decomped: array-like
        Decomposed parameter dataset, shape (x, y, n_params), first output of fit.periodic_fit_whole
    num: int
        Number of parameters in fit
    angle_map,amp_map : str
        Colour map applied to the angle and amplitude parameters respectively
        Defaults to "twilight" and "inferno"
    angle_amp: array-like
        Defines which parameters are angles and amplitude, 1 = amplitude 0 = angle
        Defaults to [0,1,0,1,0], which corresponds to the periodic model used in this package

    Returns
    -------
    Plot of raw and decomposed param maps
        
    """
    cmap1 = "twilight"
    cmap2 = "inferno"
    angle_or_amp = [0,1,0,1,0]
    
    if angle_map:
        cmap1 = angle_map

    if amp_map:
        cmap2 = amp_map

    if angle_amp:
        angle_or_amp = angle_amp
    maps = []
    
    for k in range(num):
        if angle_or_amp[k] == 0:
            maps.append(cmap2)
        else:
            maps.append(cmap1)
        
    fig, axes = plt.subplots(num, 2,figsize=(10,20))

    for n in range(num):

    # Raw
        im0 = axes[n, 0].imshow(raw[:, :, n],cmap=maps[n])
        axes[n, 0].set_title(f'Raw Parameter {n}')
        fig.colorbar(im0, ax=axes[n, 0])

    # Decomposed
        im1 = axes[n, 1].imshow(decomped[:, :, n],cmap=maps[n])
        axes[n, 1].set_title(f'Decomposed Parameter {n}')
        fig.colorbar(im1, ax=axes[n, 1])

def visualise_residuals(raw,decomped,num):
    """
    Allows visualsation of residuals of parameters maps calculated for raw and decomposed data
    
    Parameters
    ----------
    raw: array-like
        Raw parameter dataset, shape (x, y, n_params), second output of fit.periodic_fit_whole
    decomped: array-like
        Decomposed parameter dataset, shape (x, y, n_params), first output of fit.periodic_fit_whole
    num: int
        Number of parameters in fit
    Returns
    -------
    Plot of raw and decomposed param maps residuals
    """
    
    fig, axes = plt.subplots(num, 1,figsize=(10,20))
    for n in range(num):
    # Residuals
        im0 = axes[n].imshow(raw[:, :, n]-decomped[:, :, n],cmap='magma')
        axes[n].set_title(f'Residuals Parameter {n}')
        fig.colorbar(im0, ax=axes[n])
    
    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()

def test_point(raw, decomped, raw_params, decomp_params, test_point, degrees):
    """Visualise the raw and decomposed data and the corresponding fits for a given pixel.

    Parameters
    ----------
    raw: array-like
        Raw dataset, shape (x, y, degrees)
    decomped: array-like
        Decomposed dataset, shape (x, y, degrees)
    raw_params: array-like
        Raw params dataset, shape (x, y, n_params)
    decomp_params: array-like
        Decomposed params dataset, shape (x, y, n_params)
    test_point: array-like,int
        Chosen pixel to visualise, form [x,y]
    degrees: int
        Number of degrees in pixel signal

    Returns
    -------
    Plot of raw and decomposed fits and data
        
    """
    X = np.arange(degrees)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(X, raw[test_point[0], test_point[1]],                           
            label="Raw data",          color="tab:blue",   alpha=0.5)
    ax.plot(X, decomped[test_point[0], test_point[1]],                       
            label="Denoised data",     color="tab:orange", alpha=0.5)
    ax.plot(X, fit.periodic(X, *raw_params[test_point[0], test_point[1]]),   
            label="Raw fit",           color="tab:blue",   linestyle="--")
    ax.plot(X, fit.periodic(X, *decomp_params[test_point[0], test_point[1]]),
            label="Denoised fit",      color="tab:orange", linestyle="--")

    ax.set_xlabel("Degrees")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Test point ({test_point[0]}, {test_point[1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
