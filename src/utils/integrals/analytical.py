import torch
import numpy as np

def analytical_integral(t0:torch.Tensor, tn:torch.Tensor, 
                        i:int, j:int,
                        z:torch.Tensor, v:torch.Tensor,  beta:torch.Tensor) -> torch.Tensor:
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

    xi, xj, yi, yj      = z[i,0], z[j,0], z[i,1], z[j,1]
    vxi, vxj, vyi, vyj  = v[i,0], v[j,0], v[i,1], v[j,1]

    a = xi - xj
    b = yi - yj
    m = vxi - vxj
    n = vyi - vyj

    ## Simons integral
    return (    -   
                (
                torch.sqrt(torch.pi) / (2*torch.sqrt(m**2 + n**2))
                )
                *
                ( 
                torch.exp(((-b**2 + beta) * m**2 + 2*a*b*m*n - n**2 * (a**2 - beta)) / (m**2 + n**2))
                ) 
                * 
                (torch.erf(((m**2 + n**2)*t0 + a*m + b*n) / torch.sqrt(m**2 + n**2)) - 
                torch.erf(((m**2 + n**2)*tn + a*m + b*n) / torch.sqrt(m**2 + n**2))) 
            )

    # ## Our integral
    # return (    -   
    #             (
    #             torch.sqrt(torch.pi) / (2*torch.sqrt(m**2 + n**2))
    #             )
    #             *
    #             ( 
    #             torch.exp(((-b**2 - a + beta) * m**2 - n**2 * (a - beta)) / (m**2 + n**2))
    #             ) 
    #             * 
    #             (torch.erf(((m**2 + n**2)*t0 + b*n) / torch.sqrt(m**2 + n**2)) - 
    #             torch.erf(((m**2 + n**2)*tn + b*n) / torch.sqrt(m**2 + n**2))) 
    #         )



def vec_analytical_integral(t0:torch.Tensor, tn:torch.Tensor, 
                            z0:torch.Tensor, v0:torch.Tensor, beta:torch.Tensor, device):
    eps = torch.tensor(np.finfo(float).eps).to(device) #Adding eps to avoid devision by 0 
    a = (z0[:,0].unsqueeze(1) - z0[:,0].unsqueeze(0)) + eps
    m = (v0[:,0].unsqueeze(1) - v0[:,0].unsqueeze(0)) + eps
    b = (z0[:,1].unsqueeze(1) - z0[:,1].unsqueeze(0)) + eps
    n = (v0[:,1].unsqueeze(1) - v0[:,1].unsqueeze(0)) + eps

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