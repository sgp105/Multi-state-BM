# distutils: extra_compile_args=-Wno-unreachable-code-fallthrough
# cython: language_level=3

cimport numpy as np
import numpy as np
import scipy as sp
import os
from numpy.random import uniform, normal, beta, exponential

def createDirectory(str path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")

cdef public double D_min  = 0.001
cdef public double D_max  = 2.0
cdef public double unit_T = 0.1    # s
cdef public double unit_L = 0.167  # um

cpdef dict get_params():
    return {
        "D_min": D_min,
        "D_max": D_max,
        "unit_T": unit_T,
        "unit_L": unit_L,
    }

def set_params(dict params=None, **kwargs):
    """
    set_params({"D_min":0.002, "unit_T":0.05})
    or set_params(D_min=0.002, unit_T=0.05)
    """
    cdef object v
    global D_min, D_max, unit_T, unit_L
    if params is None:
        params = {}
    params = {**params, **kwargs}

    if "D_min" in params:
        v = params["D_min"];  D_min  = <double> v
    if "D_max" in params:
        v = params["D_max"];  D_max  = <double> v
    if "unit_T" in params:
        v = params["unit_T"]; unit_T = <double> v
    if "unit_L" in params:
        v = params["unit_L"]; unit_L = <double> v

    if D_min <= 0 or D_max <= 0 or D_min >= D_max:
        raise ValueError("Invalid D_min/D_max")
    if unit_T <= 0 or unit_L <= 0:
        raise ValueError("unit_T/unit_L must be > 0")

    return get_params()

def example_uses_globals():
    return D_min, D_max, unit_T, unit_L


def prior_nonuniform(int nstates, int order):
    cdef np.ndarray[np.float64_t, ndim = 2] alpha
    cdef int row, col
    
    if order < nstates:
        return uniform(D_min, D_max)
    else:
        alpha = np.ones((nstates, nstates)) * (1 + 0.5 / (nstates - 1))
        alpha[range(nstates), range(nstates)] = 10.5
        row = (int)((order - nstates) / nstates)
        col = order % nstates
        return beta(alpha[row, col], alpha[row + 1:, col].sum())

def convert_theta(np.ndarray[np.float64_t, ndim = 1] theta, int nstates):
    cdef np.ndarray[np.float64_t, ndim = 2] sigma, P
    cdef int i, j
    
    # diffusivity to sigma
    sigma = np.sqrt(2 * theta[:nstates] * unit_T)[:, None]
    P = np.ones((nstates, nstates), dtype = float)
    # to transition probability

    if nstates == 1 :
        return sigma, P
    else:
        for i in range(nstates - 1):  # row
            for j in range(nstates):  # column
                if i == 0: #first row
                    P[i, j] = theta[nstates + j]
                elif i == 1:
                    P[i, j] = (1 - P[i-1, j]) * theta[nstates*2 + j]
                elif i == 2:
                    P[i, j] = (1 - P[i-1, j] - P[i-2, j]) * theta[nstates*3 + j]
        P[nstates - 1, :] = 1 - P[:nstates - 1, :].sum(axis = 0)
        return sigma, P
    
def hmm_p_ini(np.ndarray[np.float64_t, ndim = 2] P, int nstates): ##P : transition probability, p_ij : prob. from j to i
    cdef np.ndarray[np.float64_t, ndim = 2] p_ini
    cdef np.ndarray[np.float64_t, ndim = 1] eig_val
    cdef np.ndarray[np.float64_t, ndim = 2] eig_vec
    
    eig_val = np.real(np.linalg.eig(P)[0])
    eig_vec = np.real(np.linalg.eig(P)[1])
    p_ini = eig_vec[:, abs(eig_val - 1) < 0.0000001] / (eig_vec[:, abs(eig_val - 1) < 0.0000001]).sum()
    return p_ini #column vector

def hmm_emission(np.ndarray[np.float64_t, ndim = 1] obs,
                 np.ndarray[np.float64_t, ndim = 2] sigma):
    return -np.log(2*np.pi*(sigma**2))/2 - (obs**2)/(2*(sigma**2))

def logsumexpcol(np.ndarray[np.float64_t, ndim = 2] x):
    cdef np.ndarray[np.float64_t] y, z
    y = np.max(x, axis = 0)
    x = x - y
    z = y + np.log((np.exp(x)).sum(0))
    z[y[:]==-np.inf] = -np.inf
    return z

def forward(np.ndarray[np.float64_t, ndim = 2] obs,
            np.ndarray[np.float64_t, ndim = 2] P,
            np.ndarray[np.float64_t, ndim = 2] sigma, #sigma : column vector
            int nstates):
    cdef np.ndarray[np.float64_t, ndim = 2] pstate_ini, logPT, aprev, a  #pstate_ini : column vector
    cdef np.ndarray[np.float64_t, ndim = 1] obs_cur # value
    cdef int i
    
    pstate_ini = hmm_p_ini(P, nstates).copy() #initial distribution
#    pstate_ini = np.ones(nstates, dtype = float)[:, None] / nstates #uniform initial distribution
    
    logPT = np.log(P.T).copy()
    aprev = np.log(pstate_ini).copy()
    a = np.zeros((nstates, obs.shape[0]))
    
    for i in range(obs.shape[0]):
        obs_cur = obs[i].copy()
        a[:, i][:, None] = hmm_emission(obs_cur, sigma) + aprev  #log alpha(time), column vector
        aprev = logsumexpcol(logPT + a[:, i][:, None])[:, None]

    return a, logsumexpcol(a[:, obs.shape[0]-1][:, None])  #log alpha and log likelihood

def backward(np.ndarray[np.float64_t, ndim = 2] obs,
             np.ndarray[np.float64_t, ndim = 2] P,
             np.ndarray[np.float64_t, ndim = 2] sigma, #sigma : column vector
             int nstates):
    
    cdef np.ndarray[np.float64_t, ndim = 2] logP, b
    cdef np.ndarray[np.float64_t, ndim = 1] obs_cur
    cdef int i, trjlen
    
    logP = np.log(P).copy()
    b = np.zeros((nstates, obs.shape[0]))
    
    trjlen = obs.shape[0]
    for i in range(1, trjlen):
        obs_cur = obs[trjlen - i].copy()
        b[:, trjlen - i - 1][:, None] = logsumexpcol(logP + b[:, trjlen - i][:, None] + hmm_emission(obs_cur, sigma))[:, None]
    
    return b

#update gamma and xi
def gamma_probs(np.ndarray[np.float64_t, ndim = 2] obs,
                np.ndarray[np.float64_t, ndim = 2] P,
                np.ndarray[np.float64_t, ndim = 2] sigma, #sigma : column vector
                
                np.ndarray[np.float64_t, ndim = 2] a,
                np.ndarray[np.float64_t, ndim = 2] b,
                double logL,
                int nstates):
    
    cdef np.ndarray[np.float64_t, ndim = 2] gamma
    cdef int i, j
    
    gamma = np.zeros((nstates, obs.shape[0]))
    for i in range(obs.shape[0]):
        gamma[:, i] = a[:, i] + b[:, i] - logL
        
    return np.exp(gamma)

def gamma_est(np.ndarray[np.float64_t, ndim = 2] obs,
              np.ndarray[np.float64_t, ndim = 2] P,
              np.ndarray[np.float64_t, ndim = 2] sigma, #sigma : column vector
              int nstates):
    
    cdef np.ndarray[np.float64_t, ndim = 2] gamma, a, b
    cdef double logL
    cdef int i, j
    
    a, logL = forward(obs, P, sigma, nstates)  #log alpha, log L
    b = backward(obs, P, sigma, nstates)  #log beta
    
    gamma = np.zeros((nstates, obs.shape[0]))
    for i in range(obs.shape[0]):
        gamma[:, i] = a[:, i] + b[:, i] - logL
        
    return np.exp(gamma)

def xi_probs(np.ndarray[np.float64_t, ndim = 2] obs,
             np.ndarray[np.float64_t, ndim = 2] P,
             np.ndarray[np.float64_t, ndim = 2] sigma, #sigma : column vector
             
             np.ndarray[np.float64_t, ndim = 2] a,
             np.ndarray[np.float64_t, ndim = 2] b,
             double logL,
             int nstates):
    
    cdef np.ndarray[np.float64_t, ndim = 3] xi
    cdef np.ndarray[np.float64_t, ndim = 2] logPT
    cdef np.ndarray[np.float64_t, ndim = 1] obs_cur
    cdef int t
    
    xi = np.zeros((obs.shape[0] - 1, nstates, nstates))   #log xi
    logPT = np.log(P.T).copy()
    
    for t in range(obs.shape[0] - 1):
        obs_cur = obs[t + 1].copy()
        xi[t] = (a[:, t][:, None] + logPT +  (b[:, t + 1][:, None] + hmm_emission(obs_cur, sigma))[:, 0] - logL).T
    
    return np.exp(xi)

def baum_welch(np.ndarray[np.float64_t, ndim = 2] obs, #obs[t] : v(t), time * dimension
               np.ndarray[np.float64_t, ndim = 1] D,
               int nstates,
               int iteration,
               double stop_logratio = 0.00000001,
               int max_cnt = 1000,
               int seed = 1):
    cdef np.ndarray[np.float64_t, ndim = 3] xi
    cdef np.ndarray[np.float64_t, ndim = 2] gamma
    cdef np.ndarray[np.float64_t, ndim = 2] a, b, tempa
    cdef np.ndarray[np.float64_t, ndim = 1] params
    cdef np.ndarray[np.float64_t, ndim = 2] sigma, P, opt_P, opt_sigma
    cdef double logL, opt_logL, new_logL
    cdef double logratio
    cdef int i, j, nparams, cnt
    
    nparams = nstates ** 2
    params = np.zeros(nparams)
    opt_params = np.zeros(nparams)
    opt_logL = -np.inf

    np.random.seed(seed)
    
    for i in range(iteration):  #run baum-welch with different initial paramters
        print('iter:', i)
        #setting initial parameters from prior distribution
        for j in range(nparams):
#            params[j] = prior_jeffrey(nstates, j)
            params[j] = prior_nonuniform(nstates, j)
        print('D:', D)
        params[:nstates] = D.copy() ## fixed diffusivity
        sigma, P = convert_theta(params[:nparams], nstates)
        
        print('sigma:', sigma)
        print('P:', P)
        
        a, logL = forward(obs, P, sigma, nstates)  #log alpha, log L
        b = backward(obs, P, sigma, nstates)  #log beta
        
        cnt = 0
        while(1):
            xi = xi_probs(obs, P, sigma, a, b, logL, nstates)
            gamma = gamma_probs(obs, P, sigma, a, b, logL, nstates)
            
            #P = np.sum(xi, axis = 0) / np.sum(gamma[:, :obs.shape[0] - 1], axis = 1)
            P = np.sum(xi, axis = 0) / np.sum(np.sum(xi, axis = 0), axis = 0) #P update
            #sigma = np.sqrt(np.sum(gamma * (obs ** 2)[:, 0], axis = 1)/ np.sum(gamma, axis = 1))[:, None]
            
            a, new_logL = forward(obs, P, sigma, nstates)
            b = backward(obs, P, sigma, nstates)
            
            logratio = new_logL - logL
            logL = new_logL
            
            if logratio < stop_logratio or cnt > max_cnt:
                if opt_logL < logL:
                    opt_logL = logL
                    opt_P = P
                    opt_sigma = sigma
                break
            cnt += 1
        print(opt_P)
    return opt_P


def baum_skel_for_P(np.ndarray[np.float64_t, ndim = 2] obs,
                    np.ndarray[np.float64_t, ndim = 1] D,
                    int nstates,
                    int iteration,
                    double stop_logratio = 0.00001,
                    int max_cnt = 100,
                    int seed = 1):

    cdef int i, j
    cdef np.ndarray[np.float64_t, ndim = 2] opt_D, opt_P
    cdef double opt_logL

    print('nstate:', nstates)
    opt_P = baum_welch(obs, D, nstates, iteration, max_cnt = max_cnt)
    
    return opt_P

def viterbi(np.ndarray[np.float64_t, ndim = 2] obs, # column vector
            np.ndarray[np.float64_t, ndim = 2] P, # (optimal) transition matrix
            np.ndarray[np.float64_t, ndim = 2] sigma, # (optimal) sigmas, column vector
            int nstates
           ):
    
    cdef np.ndarray[np.float64_t, ndim = 2] T1
    cdef np.ndarray[np.int64_t, ndim = 2] T2
    cdef np.ndarray[np.float64_t, ndim = 2] pi # initial distribution, column vector
    cdef np.ndarray[np.float64_t, ndim = 2] pstate_ini #
    cdef np.ndarray[np.int64_t, ndim = 1] X # optimal sequence
    cdef int i, j, T
    
    
    T = np.shape(obs)[0]
    pi = np.log(hmm_p_ini(P, nstates))  # log initial probability
    
    T1 = np.zeros((nstates, T))  # nstates * trjlen
    T2 = np.zeros((nstates, T), dtype = np.int64)  # log zero
    
    T1[:, 0] = pi[:, 0] + hmm_emission(obs[0], sigma)[:, 0]  # log
    for i in range(1, T):
        T1[:, i] = hmm_emission(obs[i], sigma)[:, 0] + np.max(np.log(P) + np.array([T1[:, i-1],] * nstates), axis = 1)
        T2[:, i] = np.argmax(np.log(P) + np.array([T1[:, i-1],] * nstates), axis = 1)
    
    X = np.zeros(T, dtype = int)
    X[T-1] = np.argmax(T1[:, T-1])

    for i in range(1, T):
        X[T-(i+1)] = T2[X[T-i], T-i]

    return X

def dp_to_theta(np.ndarray[np.float64_t, ndim = 1] D, np.ndarray[np.float64_t, ndim = 2] P, int nstates):
    cdef int i, j
    cdef np.ndarray[np.float64_t, ndim = 1] theta
    
    theta = np.zeros(nstates ** 2)
    theta[:nstates] = D.copy()
    
    if nstates != 1:
        for i in range(nstates - 1):
            for j in range(nstates):
                if i == 0:
                    theta[nstates * (i + 1) + j] = P[i, j]
                elif i == 1:
                    theta[nstates * (i + 1) + j] = P[i, j] / (1 - P[i-1, j])
                elif i == 2:
                    theta[nstates * (i + 1) + j] = P[i, j] / (1 - P[i-1, j] - P[i-2, j])
    return theta