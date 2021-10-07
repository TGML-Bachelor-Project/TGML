import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def analytical_integral(t0:float, tn:float, z:torch.Tensor, v:torch.Tensor, i:int, j:int, beta:torch.Tensor) -> torch.Tensor:
    '''
    Calculates the Riemann sum for the integral from t0 to tn
    based on the nodes i and j and the given function func.

    :param t0:          Start of integral interval
    :param tn:          End of integral interval
    :param z:          The latent representation z to the time t
    :param v:           The constant velocity vector
    :param i:           The index of node i
    :param j:           The index of node j
    :param func:        The intensity function to compute the integral of

    :returns:           The closed form solution of the squared euclidean intensity function
    '''
    pos_i, pos_j = z[i, :], z[j, :]
    xi, yi, xj, yj = pos_i[0], pos_i[1], pos_j[0], pos_j[1]
    vi, vj = v[i, :], v[j, :]
    vxi, vyi, vxj, vyj = vi[0], vi[1], vj[0], vj[1]

    a = xi-xj
    m = vxi - vxj
    b = yi - yj
    n = vyi - vyj

    return  (((torch.sqrt(torch.pi))*torch.exp(((-b**2-a+beta)*m**2 - n**2*(-beta+a)) / (m**2 + n**2) ))/(2*torch.sqrt(m**2 + n**2)) 
                *
                (
                    torch.erf(((m**2+n**2)*tn + b*n)/(torch.sqrt(m**2+n**2))) -
                    torch.erf(((m**2 + n**2)*t0+b*n)/(torch.sqrt(m**2+n**2)))
                )
             )