"""
The functions contained within this file are all related to a master function
which can decompose, fit, visualise and assess a dataset.

Each function includes a description of the method alongside the required
inputs and outputs. Demonstrations of the master pipeline function can be found in
the "pipeline" demo notebook. This is conducted on real data to also demonstrate the 
outcomes of this project.
"""


"""" ------- Required Imports -----"""
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy import optimize
import logging

""""------ Cross Imports-------"""
import PhysComp.assess as assess
import PhysComp.fitting as fit
import PhysComp.uml as uml
import PhysComp.visual as vs

"""------ Functions------"""

def decomp_fit_pipeline(path,algo,scree_plot=False,param_visual = False,error_visual=False,residual=False ,components=None,initial= None,r2=False,signalleak=False, bounds= None, angle_map = None, amp_map=None,angle_amp = None,verbose=False):

    """
    Master function which decomposes dataset and fits periodic function to each pixel. Various other features including visualastion and assessment.
    Works as orchestrator function calling upon all other functions within project.
    
    Required Parameters:
    ----------
    path: str
        path to the data file
    algo: str
        name of algorithm for decomposition e.g. "SVD", "NMF", "sklearn"

    Optional Params:
    ----------
    components: int
        Number of components for reconstruction, if left as None number of components decided via scree plot
    inital: list
        initial guess for parameters, default set if left blank
    scree_plot: bool
        Whether to visualise scree plot
    param_visual: bool
        Whether to visualise fitting parameter map
    error_visual: bool
        Whether to visualise error map
    residual: bool
        Whether to visualise residual map
    r2: bool
        Whether to calculate adjusted r2 value
    signalleak: bool
        Whether to calculate signal leakage
    angle_map, amp_map: str
        Fed into visualise functions
    amp_map: list
        Fed into visualise functions
    verbose: bool
        Wheter each step is logged and signaled for debugging
    
    Returns
    -------
    dict with keys:
    "decomped_data":    arrray : decomposed data set
    "raw_data":         array : raw data set
    "decomped_params":  array : decomposed paramater map,
    "raw_params":       array : raw paramater map,
    "raw_errors":       array : raw errors of paramater map,
    "decomped_errors":  array : decomposed errors of paramater map,
    "raw_r2s":          array : raw adjusted r2 valiues map,
    "decomped_r2s":     array : decomposed adjusted r2 valiues map,
    "signal_leak":      float: signal leakage value
        
    """
    
    
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)    
    if verbose == True:
        logger.setLevel(logging.DEBUG)
    ## ------- Input Validation ----------
    logger.debug("Validating Inputs...")
    validation(path,algo,scree_plot,param_visual,error_visual,residual,components,initial,r2,signalleak,bounds)

    ## -------- Decomposition -----------
    logger.debug("Decomposing Data...")
    try:
        decomped_data, raw_data = uml.decomp_data(path,algo,plot=scree_plot,components=components)
    except Exception  as e:
        raise RuntimeError(f"Decomposition Failed: {e}") from e
    X = np.arange(len(raw_data[0][0]))

    ## ---------- Periodic fitting --------------
    logger.debug("Fitting Decomped Data...")
    
    try:
        decomped_params, decomped_errors = fit.periodic_fit_whole(decomped_data,initial,bounds)
    except Exception as e:
        raise RuntimeError(f"Periodic fit failed on Decomposed data: {e}") from e
    logger.debug("Fitting Raw Data...")

    try:
        raw_params, raw_errors = fit.periodic_fit_plot_whole(raw_data,initial,bounds)
    except Exception as e:
        raise RuntimeError(f"Periodic fit failed on raw data: {e}") from e

    ## ------- r2 calculation -----------
    raw_r2s = no_calc_warn("raw_r2s") if not r2 else []
    decomped_r2s = no_calc_warn("decomped_r2s") if not r2 else []
    
    if r2 == True:
        logger.debug("Calculating R2 Values...")
        
        for nx in range(np.shape(raw_data)[0]):
            for ny in range(np.shape(raw_data)[1]):
                try:
                    raw_r2s.append(assess.adjusted_r2(raw_data[nx][ny],perodic(X,*raw_params[nx][ny])))
                    decomped_r2s.append(assess.adjusted_r2(decomped_data[nx][ny],perodic(X,*decomped_params[nx][ny])))
                except Exception as e:
                    raise RuntimeError(f"R2 calculation failed at pixel ({nx} , {ny}): {e}") from e

    ## ---------- Signal Leakage --------------
    signal_leak = no_calc_warn("signal_leakage")
    if signalleak == True:
        logger.debug("Calculating Signal Leakage...")
        
        try:
            signal_leak = assess.signal_leakage(raw_data,decomped_data)
        except Exception as e:
            raise RuntimeError(f"Signal Leakage calculation failed: {e}") from e

   ## ---------- Visualisation -------------
    if param_visual == True:
        logger.debug("Visualising Paramater...")
        vs.visualise_params(raw_params,decomped_params,"Parameter",5,angle_map=angle_map,amp_map=amp_map,angle_amp=angle_amp)

    if error_visual == True:
        logger.debug("Visualising Errors...")
        vs.visualise_params(raw_errors,decomped_errors,"Parameter Error",5,angle_map=angle_map,amp_map=amp_map,angle_amp=angle_amp)

    if residual == True:
        logger.debug("Visualising Residuals...")
        vs.visualise_residuals(raw_params,decomped_params,5,"Parameter")
        vs.visualise_residuals(raw_errors,decomped_errors,5,"Residuals")

    return {
        "decomped_data":    decomped_data,
        "raw_data":         raw_data,
        "decomped_params":  decomped_params,
        "raw_params":       raw_params,
        "raw_errors":       raw_errors,
        "decomped_errors":  decomped_errors,
        "raw_r2s":          raw_r2s,
        "decomped_r2s":     decomped_r2s,
        "signal_leak":      signal_leak,
        }

def no_calc_warn(param_name: str):
    logger = logging.getLogger(__name__)
    Not_calc = {"raw_r2s": "Not calculated - re-run with r2=True, or call adjusted_r2() directly",
        "decomped_r2s": "Not calculated - re-run with r2=True, or call adjusted_r2() directly",
        "signal_leakage": "Not calculated - re-run with signalleak=True, or call signal_leakage() directly"
    }
    return Not_calc[param_name]

def validation(path,algo,scree_plot,param_visual,error_visual,residual,components,initial,r2,signalleak,bounds):
    if not isinstance(path,str):
        raise TypeError(f"'path' must be a string, got {type(path)}")
    if not isinstance(algo,str):
        raise TypeError(f"'algo' must be a string, got {type(algo)}")
    if components is not None and (not isinstance(components,int ) or components < 1):
        raise ValueError(f"components must be a positive integer or None")
    if initial is not None and not isinstance(initial,(list,np.ndarray)):
        raise ValueError(f"inital must be a list or array or None")
    if bounds is not None:
        if not (isinstance(bounds,tuple) and len(bound) == 2):
            raise ValueError(f"'bounds' must be a tuple of (lower_bounds,upper_bounds)")
