import hyperspy.api as hs

def decomp_data(path, components, algo):
  """
    Decomposes given file with UML for a specified number of components using given algorithm.

    Parameters:
    - path : string : path of file for decomposing 
    - components : int : defines up to number of components for the reconstruction
    - algo : string : desired algorithm for decomp

    Returns:
    - decomposed_data : np.ndarray : data set with given number of components
    - raw_data_end : np.ndarray : raw data set
    """

    raw_data = hs.load(path)
    raw_data_end = np.load(path)
    data_hs_func = hs.signals.Signal1D(raw_data.data)
    data_hs_func.decomposition(algorithm=algo)
    sc = data_hs_func.get_decomposition_model(components)
    decomposed_data = sc.data

    return decomposed_data, raw_data_end
