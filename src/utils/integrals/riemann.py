import torch

def riemann_sum(t0:float, tn:float, n_samples:int, i:int, j:int, func, device) -> torch.Tensor:
    '''
    Calculates the Riemann sum for the integral from t0 to tn
    based on the nodes i and j and the given function func.

    :param t0:          Start of integral interval
    :param tn:          End of integral interval
    :param n_samples:   Number of time to split the interval from t0 to tn
    :param i:           The index of node i
    :param j:           The index of node j

    :returns:           The Riemann sum of the integral
    '''
    # https://secure.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
    x = torch.linspace(t0, tn, n_samples+1)
    x_mid = (x[:-1]+x[1:])/2
    dx = (tn - t0) / n_samples
    rsum = torch.zeros(size=(1,1)).to(device)

    for x_i in x_mid:
        rsum += func(i, j, x_i) * dx
    
    return rsum