import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

## Defining polynomial functions for curve fitting
def poly1(x, m, c):
    return m*x + c

def poly2(x,a,b,c):
    return a*x**2 + b*x + c

def poly3(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

def poly4(x,a,b,c,d,e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def poly5(x,a,b,c,d,e,f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f


## Definig polynomial plot function to complete all tasks from excercise 2 for any give x and y data set
def poly_fit_plot(x_variable, y_variable, polynumber, plot = False):
## Array of different polynomial functions to allow different orders to be plotted by same function
    polys = [poly1,poly2,poly3,poly4,poly5]
## Using optimize.curve_fit to fit a chosen poly funtion to our x and y data
    params_poly, params_poly_covariance = optimize.curve_fit(polys[polynumber-1],x_variable,y_variable)

## Finding estimated uncertainties for each parameter
    errs = np.sqrt(np.diag(params_poly_covariance))

## creating a dummy variable for plotting purposes using linspace, parameters of linspace ensure it varies from lowest to highest x value regarless of order in array
    dummy = np.linspace(np.min(x_variable),np.max(x_variable),1000)

## Calculating residuals
    difference = polys[polynumber-1](x_variable,*params_poly) - y_variable

## Calculating r squared for goodness of fit
## Squared sum of residuals
    ss_res = np.sum(difference**2)

## Squared sum of data - mean
    ss_tot = np.sum((y_variable- np.mean(y_variable))**2)

## Calculate r_squared per equation
    r_squared = 1 - (ss_res/ss_tot)

## Allows the option of running function with no plotting
    if plot == True:
## Setting up subplots
        plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]},sharex=True,figsize=[10,6]) # height ratios make sizes right, sharex makes plots share x axis
## Subplot 2 shows residuals
        plt.subplot(212)        
        plt.plot(x_variable,difference,'or')
        plt.xlabel("Current (A)")
        plt.ylabel("Residuals")

        plt.grid()
        plt.tight_layout()
## Subplot 1 with fit and raw data
        plt.subplot(211)
        plt.scatter(x_variable, y_variable,label="Raw Data")
        plt.plot(dummy,polys[polynumber-1](dummy,*params_poly),color="r",label="Fitted Function") ## using curvefit parameters and linear function to plot function
        plt.fill_between(dummy,polys[polynumber-1](dummy,*(params_poly+errs)),polys[polynumber-1](dummy,*(params_poly-errs)),alpha=0.2) ## highlights linear fit within 1 SD of both parameters, too small to see for this data
        plt.plot([], [], ' ', label="R^2 = {0:.4f}".format(r_squared)) ## displaying r squared value
        plt.grid()
        plt.legend()

    return(params_poly)
