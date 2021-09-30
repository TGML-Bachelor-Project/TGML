import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def riemann_sum(t0:float, tn:float, n_samples:int, func) -> torch.Tensor:
    '''
    Calculates the Riemann sum for the integral from t0 to tn
    based on the nodes i and j and the given function func.

    :param z:           The latent vector representation
    :param u:            Index of node u
    :param v:           Index of node v 
    :param t0:          Start of integral interval
    :param tn:          End of integral interval
    :param n_samples:   Number of time to split the interval from t0 to tn

    :returns:           The Riemann sum of the integral
    '''
    # https://secure.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
    t = torch.linspace(t0, tn, n_samples+1)
    t_mid = (t[:-1]+t[1:])/2
    dt = (tn - t0) / n_samples
    rsum = torch.zeros(size=(1,1)).to(device)

    for t_i in t_mid:
        rsum += func(t_i) * dt
    
    return rsum