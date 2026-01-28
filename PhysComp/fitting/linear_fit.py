import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def linear(x, m, c):
    """
       Simple linear function for fitting purposes

        Parameters
        ----------
        x : array
            x axis data for the straight line
        m : number
            Gradient of straight line
        c : number
            Y -intercept of straight line

        Returns
        -------
        m*x + c : An array of the y values related to the x-data by the straight line equatio defined by the given
        gradient and y-intercept
        """
    return m*x + c

## Definig linear plot function to complete all tasks from excercise 1 for any give x and y data set
def linear_fit_plot(x_variable, y_variable):
    """
      Uses optimize.curvefit to fit and plot a linear function to raw x and y data.

        Parameters
        ----------
        x_variable : array
            x axis data
        y_variable : array
            y_axis data
        
        Returns
        -------
        Plots including raw data, fitted linear function and residuals.
        
        m : number
            Fitted parameter defining the gradient of the straight line found by curve.fit
        merr : number
            Caculated error on gradient parameter
        c : number
            Fitted parameter defining the y-intercept of the straight line found by curve.fit
        cerr : number
            Caculated error on y-intercept parameter
        r_squared :
            Calculated r squared value to asses goodness of fit
    """
## Using optimize.curve_fit to fit a linear funtion to our x and y data
    params_linear, params_linear_covariance = optimize.curve_fit(linear,x_variable,y_variable)

## Finding and Defining Parameter Values
    m = params_linear[0]
    c = params_linear[1]

## Finding estimated uncertainties for each parameter
    merr , cerr = np.sqrt(np.diag(params_linear_covariance))

## creating a dummy variable for plotting purposes using linspace, parameters of linspace ensure it varies from lowest to highest x value regarless of order in array
    dummy = np.linspace(np.min(x_variable),np.max(x_variable),1000)

## Calculating residuals
    difference = linear(x_variable,m,c) - y_variable

## Calculating r squared for goodness of fit

## Squared sum of residuals
    ss_res = np.sum(difference**2)

## Squared sum of data - mean
    ss_tot = np.sum((y_variable- np.mean(y_variable))**2)

## Calculate r_squared per equation
    r_squared = 1 - (ss_res/ss_tot)

## Setting up subplots
    plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]},sharex=True,figsize=[10,6]) # height ratios make sizes right, sharex makes plots share x axis

## Subplot 1 with fit and raw data
    plt.subplot(211)
    plt.scatter(x_variable, y_variable,label="Raw Data")
    plt.plot(dummy,linear(dummy,m,c),color="r",label="Fitted Function") ## using curvefit parameters and linear function to plot function
    plt.fill_between(dummy,linear(dummy,m+merr,c+cerr),linear(dummy,m-merr,c-cerr),alpha=0.2) ## highlights linear fit within 1 SD of both parameters, too small to see for this data
    plt.plot([], [], ' ', label="m = {0:.4f} \u00B1 {1:.4f}".format(m,merr)) ## displaying m parameter with uncertainty
    plt.plot([], [], ' ', label="c = {0:.4f} \u00B1 {1:.4f}".format(c,cerr)) ## displaying c parameter with uncertainty
    plt.plot([], [], ' ', label="R^2 = {0:.4f}".format(r_squared)) ## displaying r squared value
    plt.grid()
    plt.legend()

## Subplot 2 shows residuals
    plt.subplot(212)        
    plt.plot(x_variable,difference,'or')
    plt.xlabel("Current (A)")
    plt.ylabel("Residuals")

    plt.grid()
    plt.tight_layout()
    return(m,merr,c,cerr,r_squared)
