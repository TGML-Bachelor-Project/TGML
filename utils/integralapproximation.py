import torch
import numpy as np

def riemann_sum(i, j, t0, tn, n_samples, func):
    # https://secure.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
    x = torch.linspace(t0.item(), tn.item(), n_samples+1)
    x_mid = (x[:-1]+x[1:])/2
    dx = (tn - t0) / n_samples
    rsum = torch.zeros(size=(1,1))

    for x_i in x_mid:
        rsum += func(x_i, i, j) * dx
    
    return rsum

def monte_carlo_integral(i, j, t0, tn, n_samples, func):
    sample_times = np.random.uniform(t0, tn, n_samples)
    int_lambda = 0.

    for t_i in sample_times:
        int_lambda += func(t_i, i, j)

    interval_length = tn-t0
    int_lambda = interval_length * (1 / n_samples) * int_lambda

    return int_lambda

def evaluate_integral(i, j, t0, tn, z, v, beta):
    a = z[i,0] - z[j,0]
    b = z[i,1] - z[j,1]
    m = v[i,0] - v[j,0]
    n = v[i,1] - v[j,1]
    return -torch.sqrt(torch.pi)*torch.exp(((-b**2 + beta)*m**2 + 2*a*b*m*n - n**2*(a**2 - beta))/(m**2 + n**2))*(torch.erf(((m**2 + n**2)*t0 + a*m + b*n)/torch.sqrt(m**2 + n**2)) - torch.erf(((m**2 + n**2)*tn + a*m + b*n)/torch.sqrt(m**2 + n**2)))/(2*torch.sqrt(m**2 + n**2))
