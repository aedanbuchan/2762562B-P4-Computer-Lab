"""
The functions contained within this file are all related to decomposing 
a dataset into components and reconstructing using a number of components.

Each function includes a description of the method alongside the required
inputs and outputs. Demonstrations of each function can be found in the 
uml demo notebook in the Demos folder
"""

"""
--------- Required imports for this file ---------
"""
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

"""
------- Functions --------
"""
def decomp_data(path: str, algo: str, plot: bool = False, components: Optional[int]=None):
    """
    Decomposes file given by path, using UML into given number of components for given algorithm

    Parameters:
    - path : str : path of file for decomposition
    - algo : str : desired algorithm for decomposition (e.g. 'SVD', 'NMF')
    - plot : bool : whether to display scree plot (default False)
    - components : int or None : number of components for reconstruction.
                    If None, automatically uses scree plot to determine optimal number of comps.

    Returns:
    - decomposed_data : np.ndarray : reconstructed data with given (or calculated) number of components
    - raw_data_end : np.ndarray : raw data array
    """

    # --- Input validation -----
    if not all(isinstance(v, str) for v in (path, algo)):
      raise TypeError(f"file path and algorithm need to be strings, got {type(path).__name__!r}")
    if not isinstance(plot, bool):
      raise TypeError(f"plot must be a bool, got {type(plot).__name__!r}.")
    if components is not None and not isinstance(components, int):
      raise TypeError(f"Components must be integer or left as None for automatic computation")
    
    raw_data = hs.load(path)
    raw_data_end = np.load(path)
    data_hs_func = hs.signals.Signal1D(raw_data.data)
    data_hs_func.decomposition(algorithm=algo)

    sp = scree_plot(data_hs_func, plot=plot)

    if components:
        sc = data_hs_func.get_decomposition_model(components)
    else:
        sc = data_hs_func.get_decomposition_model(sp[0])

    decomposed_data = sc.data
    return decomposed_data, raw_data_end

def get_components(decomposed_obj, components: int, degree: int = 0, pixel: list = [0,0], size: list = [10, 10], plot: bool =False):
    """
    Extract and optionally plot individual decomposition components.

    Parameters:
    - decomposed_obj : hyperspy signal : decomposed hyperspy object
    - components : list of int : component indices to extract (can use np.arange(X) to produce first X comps)
    - degree : int : degree for slice of 2d image plot (default 0, shows degree 0 slice)
    - pixel : list of int : [row, col] pixel coordinates for single pixel plot (default [0,0], shows first pixel)
    - size : list of int : figure size [width, height] (default [10, 10])
    - plot : bool : whether to plot components (default False)

    Returns:
    - components_array : list of np.ndarray : data arrays for each component
    """
    components_array = []

    if plot:
        fig, axes = plt.subplots(len(components), 2, figsize=(size[0], size[1]))

    for c in range(len(components)):
        components_data = decomposed_obj.get_decomposition_model([components[c]]).data

        if plot:
            # Spatial map at given degree
            im0 = axes[c, 0].imshow(components_data[:, :, degree], cmap="magma")
            axes[c, 0].set_title(f'Component {c}, Degree Slice {degree}')
            fig.colorbar(im0, ax=axes[c, 0])

            # Spectrum at given pixel
            axes[c, 1].plot(components_data[pixel[0], pixel[1], :])
            axes[c, 1].set_title(f'Component {c}, Pixel {pixel}')

        components_array.append(components_data)

    if plot:
        plt.tight_layout()
        plt.show()

    return components_array

def scree_plot(decomposed_obj, plot: bool =False):
    """
    Calculate and optionally plot a scree plot for decomposition results.

    Parameters:
    - decomposed_obj : hyperspy signal : decomposed hyperspy object
    - plot : bool : whether to display the scree plot (default False)

    Returns:
    - num_relevant_comps : int : number of relevant components found at elbow
    - relevant_comps : np.ndarray : array of relevant component indices
    - EVR : np.ndarray : explained variance ratio for all components
    - component_num : np.ndarray : array of component indices
    """
    EVR = decomposed_obj.get_explained_variance_ratio().data
    relevant_comps = find_elbow(EVR)
    num_relevant_comps = len(relevant_comps)

    component_num = np.arange(len(EVR))

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Scree Plot")
        ax.semilogy()
        ax.scatter(component_num, EVR)
        plt.tight_layout()
        plt.show()

    print(f"Scree Plot: calculated {num_relevant_comps} relevant components")

    return num_relevant_comps, relevant_comps, EVR, component_num

def find_elbow(variance: list):
    """
    Find elbow point in scree plot data using maximum distance from line method.

    Parameters:
    - variance : array-like : variance explained values from decomposition (can be founs via scree_plot)

    Returns:
    - relevant_components : np.ndarray : components to keep (up to and including elbow, use len() to find number of components)
    """
    # ------- Input Validation ----------
    if variance.ndim != 1:
      raise ValueError(f"Varience must be 1-D, got {varience.ndim}-D")
      
    # Setup
    x = np.linspace(1, len(variance), len(variance))
    y = np.array(variance)

    # Normalize data
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Line vector between first and last points
    line_vec = np.array([x_norm[-1] - x_norm[0],
                         y_norm[-1] - y_norm[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)

    # Compute perpendicular distance from each point to the line
    distances = []
    for i in range(len(x_norm)):
        point_vec = np.array([x_norm[i] - x_norm[0],
                              y_norm[i] - y_norm[0]])
        proj_len = np.dot(point_vec, line_vec)
        proj_vec = proj_len * line_vec
        dist_vec = point_vec - proj_vec
        distances.append(np.linalg.norm(dist_vec))

    elbow_index = np.argmax(distances)
    relevant_components = x[:elbow_index + 1]

    return relevant_components
