import numpy as np

def line(m=0, b=0):
    return lambda x: m*x + b

def residuals(model, x, y):
    return y - model(x)

def plot_model(model, x, ax = None):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,10))
    else:
        fig = ax.get_figure()
    
    h = ax.plot(x, model(x), 
            color = 'teal', linestyle = ':',
            label = 'Model')
    return h

def residuals_for_plot(model, x, y):
    return np.vstack([x,x]), np.vstack([y, model(x)])

def plot_residuals(model, x, y, ax = None):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize(10,10))
    # else:
    #     fig = ax.get_figure()
    
    X, Y = residuals_for_plot(model, x, y)
    
    h = ax.plot(X, Y, color = 'salmon', label = 'Residuals')
    return h

def rmse(model, x, y):
    return np.sqrt(np.mean(residuals(model, x, y)**2))