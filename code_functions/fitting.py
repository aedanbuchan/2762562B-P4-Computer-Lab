## Defining linear function for curve fitting
def linear(x, m, c):
    return m*x + c

## Definig linear plot function to complete all tasks from excercise 1 for any give x and y data set
def linear_fit_plot(x_variable, y_variable):

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
