import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def analytical_integral(t0:torch.Tensor, tn:torch.Tensor, z:torch.Tensor, v:torch.Tensor, i:int, j:int, beta:torch.Tensor) -> torch.Tensor:
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

    sqb = torch.square(b)
    sqm = torch.square(m)
    sqn = torch.square(n)
    sqrtmn = torch.sqrt(sqm + sqn)
    psqmn = sqm+sqn

    return  (((torch.sqrt(torch.pi))*torch.exp(((-sqb-a+beta)*sqm-sqn*(-beta+a))/(psqmn)))
                /(2*sqrtmn) 
                *
                (
                    torch.erf(((psqmn)*tn + b*n)/(sqrtmn)) -
                    torch.erf(((psqmn)*t0+b*n)/(sqrtmn))
                )
             )
    # Simons
    # return -torch.sqrt(torch.pi)*torch.exp(((-b**2 + beta)*m**2 + 2*a*b*m*n - n**2*(a**2 - beta))/(m**2 + n**2))*(torch.erf(((m**2 + n**2)*t0 + a*m + b*n)/torch.sqrt(m**2 + n**2)) - torch.erf(((m**2 + n**2)*tn + a*m + b*n)/torch.sqrt(m**2 + n**2)))/(2*torch.sqrt(m**2 + n**2))

def vec_analytical_integral(t0:torch.Tensor, tn:torch.Tensor, Z:torch.Tensor, V:torch.Tensor, beta:torch.Tensor):
    a = xi-xj
    m = vxi - vxj
    b = yi - yj
    n = vyi - vyj

    sqb = torch.square(b)
    sqm = torch.square(m)
    sqn = torch.square(n)
    sqrtmn = torch.sqrt(sqm + sqn)
    psqmn = sqm+sqn

    return  (((torch.sqrt(torch.pi))*torch.exp(((-sqb-a+beta)*sqm-sqn*(-beta+a))/(psqmn)))
                /(2*sqrtmn) 
                *
                (
                    torch.erf(((psqmn)*tn + b*n)/(sqrtmn)) -
                    torch.erf(((psqmn)*t0+b*n)/(sqrtmn))
                )
             )