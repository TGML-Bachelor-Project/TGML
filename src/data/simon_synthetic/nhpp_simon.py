# from nodespace import NodeSpace
import numpy as np
import torch
import scipy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import numpy as np
import scipy.spatial.distance as sd


class NodeSpace():
    def __init__(self):
        self.beta = 5.0
        self.z0 = None
        self.v0 = None
        self.a0 = None
        self.z = None
        self.v = None
        self.a = None
    
    def step(self, t):
            self.z = self.z0[:,:] + self.v0[:,:]*t + 0.5*self.a0[:,:]*t**2
            return self.z
    
    def lambda_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_dist(t, u, v)
        l = np.exp(self.beta - d)
        return np.around(l, decimals=10)

    def lambda_ddt(self, t, u, v):
        z = self.step(t)
        dist = self.get_dist(t, u, v)
        z_uv = z[u,:] - z[v,:]
        z_uv_ddt = (self.v0[u,:]-self.v0[v,:]) + (self.a0[u,:]- self.a0[v,:])*t
        return -np.exp(self.beta - dist)*np.dot(z_uv, z_uv_ddt) / dist

    def lambda_int_rapprox(self, t, u, v):
        dt = np.mean(t[1:len(t)]-t[0:len(t)-1])
        rsum = 0
        for t_i in t:
            l = self.lambda_fun(t_i, u, v)
            rsum += dt * l
        return rsum
    
    def lambda_int_sq_rapprox(self, t, u, v):
        dt = np.mean(t[1:len(t)]-t[0:len(t)-1])
        rsum = 0
        for t_i in t:
            l = self.lambda_sq_fun(t_i, u, v)
            rsum += dt * l
        return rsum

    def lambda_sq_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_sq_dist(t, u, v)
        l = np.exp(self.beta - d)
        return np.around(l, decimals=10)

    def lambda_sq_ddt(self, t, u, v):
        z = self.step(t)
        dist = self.get_sq_dist(t, u, v)
        z_uv = z[u,:] - z[v,:]
        z_uv_ddt = (self.v0[u,:]-self.v0[v,:]) + (self.a0[u,:]- self.a0[v,:])*t
        return -np.exp(self.beta - dist)*np.dot(z_uv, z_uv_ddt)

    def init_conditions(self, z0, v0, a0):
        self.z0 = z0
        self.v0 = v0
        self.a0 = a0
        self.z = z0
        self.v = v0
        self.a = a0

    def init_clusters(self, n_clusts, n_points, centers, rads, seed=0):
        np.random.seed(seed)
        clusts = []
        for c in range(n_clusts):
            clust = self.init_points(n_points[c], centers[c], rads[c])
            clusts.append(clust)
        clusts = np.array(clusts)
        clusts = np.reshape(clusts, (sum(n_points),2))
        return np.array(clusts)

    def init_dynamics(self, n_clusts, n_points, v, a):
        v0 = []
        a0 = []
        for i in range(n_clusts):
            v_i = self.init_velocity(n_points[i], v[i][0], v[i][1])
            a_i = self.init_acceleration(n_points[i], a[i][0], a[i][1])
            v0.append(v_i)
            a0.append(a_i)
        v0 = np.reshape(np.array(v0), (sum(n_points),2))
        a0 = np.reshape(np.array(a0), (sum(n_points),2))
        return v0, a0

    def rand_init_dynamics(self, n_points):
        v0 = np.random.uniform(-1, 1, size=(n_points, 2))
        a0 = np.random.uniform(-1, 1, size=(n_points, 2))
        return v0, a0

    def custom_init_dynamics(self, n_points, labels, vdir, adir):
        v0 = np.zeros(shape=(n_points, 2))
        a0 = np.zeros(shape=(n_points, 2))

        for idx, label in enumerate(labels):
            #noise = np.random.normal(loc=0.0, scale=1e-3, size=(2,2))
            noise = np.zeros(shape=(2,2))
            v0[idx,:] = vdir[label] + noise[0]
            a0[idx,:] = adir[label] + noise[1]

        return v0, a0

    def init_points(self, n, center, rad):
        points = []
        for node in range(n):
            point = []
            for coordinate in center:
                lb = coordinate - rad
                ub = coordinate + rad
                p = np.random.uniform(low=lb, high=ub, size=1)
                point.append(p)
            points.append(point)
        points = np.reshape(np.array(points), newshape=(n,2))
        return points
    
    def init_velocity(self, n, vx, vy):
        v0 = np.repeat(np.array([[vx, vy]]), repeats=n, axis=0)
        return v0

    def init_acceleration(self, n, ax, ay):
        a0 = np.repeat(np.array([[ax, ay]]), repeats=n, axis=0)
        return a0
    
    def get_dist(self, t, u, v):
        m = len(self.z0)
        if u==v: 
            return 0.0
        if u > v:
            u, v = v, u
        z = self.step(t)
        d = sd.pdist(z, metric="euclidean")
        idx = m * u + v - ((u + 2) * (u + 1)) // 2
        return d[idx]
    
    def get_sq_dist(self, t, u, v):
        dist = self.get_dist(t,u,v)
        sqdist = dist**2
        debug=True
        return self.get_dist(t,u,v)**2



# USES SQUARE EUCLIDEAN DISTANCE
def root_matrix(ns):
    n_points = len(ns.z0)
    ind = np.triu_indices(n_points, k=1)
    rmat = np.zeros(shape=(n_points, n_points), dtype=object)

    for u, v in zip(ind[0], ind[1]):
        z_ux, z_uy = ns.z0[u, 0], ns.z0[u, 1]
        z_vx, z_vy = ns.z0[v, 0], ns.z0[v, 1]

        # phi-substitutions
        p1 = ns.v0[u,0] - ns.v0[v, 0] 
        p2 = ns.a0[u,0] - ns.a0[v, 0]
        p3 = ns.v0[u,1] - ns.v0[v, 1]
        p4 = ns.a0[u,1] - ns.a0[v, 1]

        # coefficients
        a = (p4**2 + p2**2)/2
        b = (3*p1*p2 + 3*p3*p4)/2
        c = p2*(z_ux - z_vx) + p4*(z_uy -z_vy) + p1**2 +p3**2
        d = p1*(z_ux - z_vx) + p3*(z_uy - z_vy)

        r = np.roots([a,b,c,d])
        r = r[np.isreal(r)] # real solutions
        r = np.real(r)
        r = r[r >= 0]       # positive time axis
        r = np.sort(r)
        
        rmat[u, v] = r
    return rmat

def monotonicity_mat(ns, root_matrix):
    beta = ns.beta
    n_points = len(ns.z0)
    ind = np.triu_indices(n_points, k=1)
    mmat = np.zeros(shape=(n_points, n_points), dtype=object)

    for u, v in zip(ind[0], ind[1]):
        i = 0
        roots = root_matrix[u,v]

        # If no real, non-negative roots, then the intensity rate is
        # either constant or monotonously increasing on positive time axis
        if len(roots) == 0:
            mmat[u,v] = np.array([])
            continue

        time_points = [roots[0] - 0.5] # t to get sign of dLambda(t)/dt before first root
        while i < len(roots)-1:
            t = roots[i] + (roots[i+1] - roots[i]) / 2 # t's to get sign of dLambda(t)/dt between roots
            time_points.append(t)
            i += 1 
        time_points.append(roots[-1] + 0.5) # t to get sign of dLambda(t)/dt after last root

        monotonicity = []
        for t in time_points:
            val = ns.lambda_sq_ddt(t, u, v) # value of derivative of lambda(t,u,v) at t

            if val < 0:
                monotonicity.append("dec")
            elif val > 0:
                monotonicity.append("inc")

        mmat[u,v] = np.array(monotonicity)

    return mmat

def time_to_monotonicity(time, roots, monotonicity):
    rprev = -np.inf
    cur_r, cur_m = 0, 0
    last_m = False if len(roots) > 1 else True
    ttm = []

    rdist = np.abs(np.diff(roots))
    dt = np.mean(np.diff(time))
    if len(rdist) > 0:
        assert all(rdist > dt), "time_to_monotonicity: Some time interval contains two sign changes."

    if len(monotonicity)==0:
        return ttm
        
    for t in time:
        if (rprev < t) & (t < roots[cur_r]):
            ttm.append(monotonicity[cur_m])
        elif last_m:
            ttm.append(monotonicity[cur_m+1])
        elif (roots[cur_r] < t) & (t < roots[cur_r+1]):
            rprev = roots[cur_r]
            cur_r += 1
            cur_m += 1
            if cur_r == len(roots) - 1:
                last_m = True
            ttm.append(monotonicity[cur_m])
    return ttm

def new_upperbounds(ns, u, v, time, roots):
    idx = 0
    if len(roots)>0:
        lambda_arr = []
        while idx < len(time)-1:
            start_t, end_t = time[idx], time[idx+1]
            roots_in_interval = roots[np.logical_and(start_t<roots, roots<end_t)]
            if len(roots_in_interval) > 0:
                check_points = [start_t, *roots_in_interval, end_t]
            else:
                check_points = [start_t, end_t]
            
            lambda_values = [ns.lambda_sq_fun(c, u, v) for c in check_points]
            arg_check = np.argmax(lambda_values)
            lambda_arr.append(np.max(lambda_values))
            idx += 1
    else:
        lambda_arr = []
        while idx < len(time)-1:
            cur_lambda = ns.lambda_sq_fun(time[idx], u, v)
            next_lambda = ns.lambda_sq_fun(time[idx+1], u, v)
            lambda_arr.append(np.max([cur_lambda, next_lambda]))
            idx += 1

    for idx, val in enumerate(lambda_arr):
        if val == 0: 
            lambda_arr[idx] = 1e-100

    return np.array(lambda_arr)

def upperbounds(ns, u, v, time, roots, time_to_monotonicity):
    beta = ns.beta
    lambda_arr = []
    mon = time_to_monotonicity
    rcount=0
    idx = 0

    if len(mon)==0:
        while idx < len(time)-1:
            cur_lambda = ns.lambda_sq_fun(time[idx], u, v)
            next_lambda = ns.lambda_sq_fun(time[idx+1], u, v)
            lambda_arr.append(np.maximum(cur_lambda, next_lambda))
            idx += 1
    else:
        while idx < len(time)-1:

            if mon[idx] != mon[idx + 1]:
                lambda_root = ns.lambda_sq_fun(roots[rcount], u, v)
                max_lambda = lambda_root

                if mon[idx] == "dec":
                    lambda_cur_t = ns.lambda_sq_fun(time[idx], u, v)
                    lambda_next_t = ns.lambda_sq_fun(time[idx+1], u, v)
                    max_lambda = np.maximum(lambda_cur_t, lambda_next_t)

                lambda_arr.append(max_lambda)
                rcount += 1
            
            elif mon[idx] == "inc":
                lambda_next_t = ns.lambda_sq_fun(time[idx+1], u, v)
                lambda_arr.append(lambda_next_t)

            elif mon[idx] == "dec":
                lambda_cur_t = ns.lambda_sq_fun(time[idx], u, v)
                lambda_arr.append(lambda_cur_t)

            idx += 1
    
    for idx, val in enumerate(lambda_arr):
        if val == 0: 
            lambda_arr[idx] = 1e-100

    lambda_arr = np.array(lambda_arr)
    return lambda_arr

def nhpp(ns, u, v, time, upperbounds):
    beta = ns.beta
    interval = 0
    t = 0
    event_times = [] 
    exceed_interval = False
    stop = False 

    while not stop:
        if not exceed_interval:
            u1 = np.random.uniform(0,1)
            x = -1 / upperbounds[interval] * np.log(u1)

        if (t + x) < time[interval + 1]:
            exceed_interval = False

            t += x
            u2 = np.random.uniform(0,1)

            lambda_t = ns.lambda_sq_fun(t, u, v)
            prob = lambda_t / upperbounds[interval]

            assert lambda_t <= upperbounds[interval], "Lambda value exceeds upperbound"
            assert prob <= 1, "Probability out of range >1"
            assert prob >= 0, "Probability out of range <0"

            if u2 < prob: # accept event time t if u2 < lambda(t) / lambda upperbound
                event_times.append(t)

        elif (t + x) >=  time[interval + 1]:
            exceed_interval = True
            if interval == len(upperbounds) - 1:
                stop = True
            else:
                x = (x - time[interval + 1] + t) * upperbounds[interval] / upperbounds[interval+1]
                t = time[interval + 1]
                interval += 1

    return np.array(event_times)

def nhpp_mat(ns, time, root_matrix, monotonicity_matrix):
    beta = ns.beta
    n_points = len(ns.z0)
    ind = np.triu_indices(n_points, k=1)
    nhpp_mat = np.zeros(shape=(n_points, n_points), dtype=object)

    for u, v in zip(ind[0], ind[1]):
        r = root_matrix[u,v]
        m = monotonicity_matrix[u,v]

        # map time to monotonicity
        #t2m = time_to_monotonicity(time, roots=r, monotonicity=m) 
        # find upperbounds for all time intervals
        #ubl = upperbounds(ns, u, v, time, roots=r, time_to_monotonicity=t2m)

        ubl = new_upperbounds(ns, u, v, time, roots=r)
        # simulate nhpp
        nhpp_sim = nhpp(ns, u, v, time=time, upperbounds=ubl) 

        nhpp_mat[u,v] = nhpp_sim
    
    return nhpp_mat

def get_entry(mat, u, v):
    assert u != v, "Upper triangular matrix: No diagonal elements."
    if u > v:
        u, v = v, u
    return mat[u, v]

if __name__ == "__main__":
    # Objective: 
    # Simulate N = n(n-1)/2 non-homogeneous Poisson processess (nhpp) at a time

    #np.random.seed(150)
    np.seterr(all='raise')

    ns = NodeSpace()
    #n_clusts = 3
    n_clusts = 2
    #n_points = [7, 7, 7]
    n_points = [1, 1]
    #centers = [[-6,0], [0,6], [8,-6]]
    centers = [[-6,0], [1,6]]
    #radius = [1.5,1.5,1.5]
    radius = [0,0]
    #v = [[1,0], [0,-1], [-1,1]]
    v = [[1,0], [0,-1]]
    #a =  [[0,-0.1], [0.1,0], [0,-0.1]]
    a =  [[0,0], [0,0]]

    # set beta
    ns.beta = 5
    ns.alpha = 0.1

    z0 = ns.init_clusters(n_clusts, n_points, centers, radius)
    v0, a0 = ns.init_dynamics(n_clusts, n_points, v, a)
    ns.init_conditions(z0, v0, a0)

    # find roots
    rmat = root_matrix(ns) 

    #find monotonicity
    mmat = monotonicity_mat(ns, rmat) 
    t = np.linspace(0, 12)

    # simulate
    nhppmat = nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

    test_u = 0
    test_v = 1

    e = get_entry(nhppmat, test_u,test_v)
    plt.hist(e)
    plt.show()


    print("Root at:", get_entry(rmat, test_u,test_v))

    lambda_int_0_T = ns.lambda_int_sq_rapprox(t, test_u, test_v)
    print("Expected value (no. events):", lambda_int_0_T)
    print("Actual value (no. events):", len(e))
