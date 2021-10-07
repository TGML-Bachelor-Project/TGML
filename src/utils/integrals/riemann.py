import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def riemann_sum(t0:float, tn:float, n_samples:int, zt, i:int, j:int, func) -> torch.Tensor:
    '''
    Calculates the Riemann sum for the integral from t0 to tn
    based on the nodes i and j and the given function func.

    :param t0:          Start of integral interval
    :param tn:          End of integral interval
    :param n_samples:   Number of time to split the interval from t0 to tn
    :param zt:          A function z(t) which gives the latent representation z to the time t
    :param i:           The index of node i
    :param j:           The index of node j

    :returns:           The Riemann sum of the integral
    '''
    # https://secure.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
    t = torch.linspace(t0, tn, n_samples+1).to(device)
    t_mid = (t[:-1]+t[1:])/2
    dt = (tn - t0) / n_samples
    rsum = torch.zeros(size=(1,1)).to(device)

    for t_i in t_mid:
        rsum += func.result(zt(t_i), i, j) * dt
    
    return rsum