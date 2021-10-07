import math
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
    t = torch.linspace(t0, tn, n_samples+1)
    t_mid = (t[:-1]+t[1:])/2
    dt = (tn - t0) / n_samples
    rsum = torch.zeros(size=(1,1)).to(device)

    for t_i in t_mid:
        rsum += func.result(zt(t_i), i, j) * dt
    
    return rsum

def analytical_squared_euclidean(t0:float, tn:float, zt, v:torch.Tensor, i:int, j:int, func):
    '''
    Calculates the Riemann sum for the integral from t0 to tn
    based on the nodes i and j and the given function func.

    :param t0:          Start of integral interval
    :param tn:          End of integral interval
    :param zt:          A function z(t) which gives the latent representation z to the time t
    :param v:           The constant velocity vector
    :param i:           The index of node i
    :param j:           The index of node j
    :param func:        The intensity function to compute the integral of

    :returns:           The closed form solution of the squared euclidean intensity function
    '''
    z = zt(t0)
    pos_i, pos_j = z[i, :], z[j, :]
    xi, yi, xj, yj = pos_i[0], pos_i[1], pos_j[0], pos_j[1]
    vi, vj = v[i, :], v[j, :]
    vxi, vyi, vxj, vyj = vi[0], vi[1], vj[0], vj[1]

    a = xi-xj
    m = vxi - vxj
    b = yi - yj
    n = vyi - vyj

    # **.5 is often faster than math.sqrt
    return  (- ((math.pi**.5)*math.exp( ((-b**2-a+func.beta)*m**2 - n**2*(-func.beta+a)) / (m**2 + n**2) ))/(2*(m**2 + n**2)**0.5) 
                *
                (
                    torch.erf(((m**2 + n**2)*t0+b*n)/((m**2+n**2)**0.5)) -
                    torch.erf(((m**2+n**2)*tn + b*n)/((m**2+n**2)**.5))
                )
             )