# distutils: extra_compile_args=-Wno-unreachable-code-fallthrough
# cython: language_level=3

cimport numpy as np
import numpy as np

import scipy as sp
import time
import os
import pickle
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

########## functions for HMM

def hmm_p_ini(np.ndarray[np.float64_t, ndim = 2] P, int nstates): ##P : transition probability, p_ij : prob. from j to i
    cdef np.ndarray[np.float64_t, ndim = 2] p_ini
    cdef np.ndarray[np.float64_t, ndim = 1] eig_val
    cdef np.ndarray[np.float64_t, ndim = 2] eig_vec
    
    eig_val = np.real(np.linalg.eig(P)[0])
    eig_vec = np.real(np.linalg.eig(P)[1])
    p_ini = eig_vec[:, abs(eig_val - 1) < 0.0000001] / (eig_vec[:, abs(eig_val - 1) < 0.0000001]).sum()
    return p_ini #column vector

def logsumexpcol(np.ndarray[np.float64_t, ndim = 2] x):
    cdef np.ndarray[np.float64_t] y, z
    y = np.max(x, axis = 0)
    x = x - y
    z = y + np.log((np.exp(x)).sum(0))
    z[y[:]==-np.inf] = -np.inf
    return z

def log_space_product(np.ndarray[np.float64_t, ndim = 2] A, np.ndarray[np.float64_t, ndim = 2] B):
    Astack = np.stack([A]*A.shape[0]).transpose(2,1,0)
    Bstack = np.stack([B]*B.shape[1]).transpose(1,0,2)
    return sp.special.logsumexp(Astack+Bstack, axis=0)
    
def hmm_emission(np.ndarray[np.float64_t, ndim = 1] obs,
                   np.ndarray[np.float64_t, ndim = 2] sigma):
    return -np.log(2*np.pi*(sigma**2))/2 - (obs**2)/(2*(sigma**2))

def hmm_main(np.ndarray[np.float64_t, ndim = 2] obs,   #obs : column vector, displacements
             np.ndarray[np.float64_t, ndim = 2] P,
             np.ndarray[np.float64_t, ndim = 2] sigma, #sigma : column vector
             int nstates):
    cdef np.ndarray[np.float64_t, ndim = 2] pstate_ini, logPT, logpstatexprev, logpstatex  #pstate_ini : column vector
    cdef np.ndarray[np.float64_t, ndim = 1] obs_cur # value
    cdef int i
    
    pstate_ini = hmm_p_ini(P, nstates).copy()
    logPT = np.log(P.T).copy()
    logpstatexprev = np.log(pstate_ini).copy()
    
    for i in range(obs.shape[0]):
        obs_cur = obs[i].copy()
        logpstatex = hmm_emission(obs_cur, sigma) + logpstatexprev
        logpstatexprev = logsumexpcol(logPT + logpstatex)[:, None]

    return logsumexpcol(logpstatex)  #log likelihood ?



########## functions for nested sampling
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

def initial_stepsize(int nstates):
    cdef np.ndarray[np.float64_t, ndim = 1] stepsize
    
    stepsize = np.zeros(nstates**2, dtype = float)
    if nstates == 1:
        stepsize[0] = np.sqrt(1/12) * (D_max - D_min) # step for diffusivity
    elif nstates == 2:
        stepsize[:2] = np.sqrt(1/12) * (D_max - D_min) # step for diffusivity
        stepsize[2:4] = np.sqrt(1/12)
    elif nstates == 3:
        stepsize[:3] = np.sqrt(1/12) * (D_max - D_min) # step for diffusivity
        stepsize[3:6] = np.sqrt(1/18)
        stepsize[6:9] = np.sqrt(1/12)
    elif nstates == 4:
        stepsize[:4] = np.sqrt(1/12) * (D_max - D_min) # step for diffusivity
        stepsize[4:8] = np.sqrt(3/5)/4
        stepsize[8:12] = np.sqrt(1/18)
        stepsize[12:16] = np.sqrt(1/12)
    stepsize *= 0.8
    return stepsize
        
def adjust_stepsize(np.ndarray[np.float64_t, ndim = 1] stepsize,
                    np.ndarray[np.int64_t, ndim = 1] reject,
                    int nsweep,
                    double target_rate):
    cdef int i
    for i in range(len(stepsize)):
        stepsize[i] = np.min([stepsize[i] * np.exp(target_rate - (reject[i]/nsweep)), 1])
    return stepsize

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

def propagator(double theta_prev,  # previous value of theta
               double stepsize,  # step size of random walker for each direction
               double rn):  # standard normal random number
    return theta_prev + stepsize * rn

def verification(theta, int nstates, int index):
    if index < nstates:
        if D_min <= theta <= D_max:
            return 1
        else:
            return -np.inf
    else:
        if 0 <= theta <= 1:
            return 1
        else:
            return -np.inf

def acceptance_nonuniform(double theta, double theta_prev, int nstates, int index):
    cdef np.ndarray[np.float64_t, ndim = 2] alpha
    cdef int row, col
    
    if index < nstates: # diffusivity
        return 1
    else:
        alpha = np.ones((nstates, nstates)) * (1 + 0.5 / (nstates - 1))
        alpha[range(nstates), range(nstates)] = 10.5
        row = (int)((index - nstates) / nstates)
        col = index % nstates
        return sp.stats.beta.pdf(theta, alpha[row, col], alpha[row + 1:, col].sum()) / sp.stats.beta.pdf(theta_prev, alpha[row, col], alpha[row + 1:, col].sum())
        
def nested_main(np.ndarray[np.float64_t, ndim = 2] obs,  # input displacements, column vector (length * dimension)
                int nwalkers,  # number of random walkers
                int imax,  # maximum sampling iteration
                double stop_ratio,  # allowed ratio of Z_remain / Z, ~ 0.0001
                double reject_rate,  # targec rejection rate, ~ 0.25
                int nsweep,  # number of random walk step per random walk (in MH algorithm)
                int nstates,
                object log_q = None):  # number of states of given model
    cdef int i, j, k, min_index, rcnt_n, rcnt_u
    cdef int nparams
    cdef double start, runningtime
    cdef np.ndarray[np.float64_t, ndim = 2] walker # active walkers [walker index, parameter index]  #### last parameter: likelihood value
    cdef np.ndarray[np.float64_t, ndim = 2] series # series of selected parameters and their likelihood values
    cdef np.ndarray[np.float64_t, ndim = 2] new_walker # new walker [sampling time, parameter index]  #### last parameter: likelihood
    cdef np.ndarray[np.float64_t, ndim = 1] stepsize, walker_keep, stnormal, unif
    cdef np.ndarray[np.int64_t, ndim = 1] reject
    cdef double evidence, evidence_remain, logweight, minL
    

    start = time.time()

    nparams = nstates ** 2
    walker = np.zeros((nwalkers, nparams + 1), dtype = float)  # walker[walker index, parameter index]
    stepsize = np.zeros(nparams, dtype = float)
    series = np.zeros((imax, nparams + 1), dtype = float)
    
    
    # determine initial step sizes for each parameter space, step size : stdev of Gaussian propagator
    stepsize = initial_stepsize(nstates)
    
    # distribute walkers to follow prior distribution and obtain their likelihood
    for i in range(nwalkers):
        for j in range(nparams):
            #walker[i, j] = prior_sampling(nstates, j)
            walker[i, j] = prior_nonuniform(nstates, j)
            #walker[i, j] = prior_jeffrey(nstates, j)
        sigma, P = convert_theta(walker[i, :nparams], nstates)
        walker[i, nparams] = hmm_main(obs, P, sigma, nstates)  #log likelihood of first parameters
    
    # nested sampling iteration
    evidence = -np.inf # initial log evidence
    logweight = np.log(1 / (nwalkers + 1)) # initial log weight

    
    for i in range(imax):
        min_index = np.argmin(walker[:, nparams])
        series[i] = walker[min_index, :].copy()  # keep parameters and log likelihood of minimum likelihood walker
        minL = series[i, nparams]
        evidence = sp.special.logsumexp([evidence, minL + logweight])
        
        new_walker = np.zeros((nsweep + 1, nparams + 1))
        walker_keep = np.zeros(nparams + 1)
        
        while(1):  # choose a new random walker from the remaining K-1 walkers
            new_walker[0, :] = walker[np.random.randint(0, nwalkers), :].copy()
            if new_walker[0, nparams] != minL:
                break
        walker_keep = new_walker[0, :].copy()
        
        ######## initialization
        rcnt_n = 0
        stnormal = normal(0, 1, nsweep * nparams)
        rcnt_u = 0
        unif = uniform(0, 1, nsweep * nparams)
        reject = np.zeros(nparams, dtype = int)
        ########
        for j in range(1, nsweep + 1): # metropolis within gibbs
            new_walker[j, :] = walker_keep.copy()
            for k in range(nparams):
                new_walker[j, k] = propagator(walker_keep[k], stepsize[k], stnormal[rcnt_n])  #previous position, parameter index, step size of random walk
                rcnt_n += 1
                
                # reject forbidden parameters
                new_walker[j, nparams] = verification(new_walker[j, k], nstates, k)
                if new_walker[j, nparams] == -np.inf:
                    new_walker[j, :] = walker_keep.copy()
                    reject[k] += 1
                else:
                    sigma, P = convert_theta(new_walker[j, :nparams], nstates)
                    new_walker[j, nparams] = hmm_main(obs, P, sigma, nstates)  #log likelihood of the sampled parameter set
                    
                    if new_walker[j, nparams] <= minL:  # reject -- likelihood boundary
                        new_walker[j, :] = walker_keep.copy()
                        reject[k] += 1
                    #elif acceptance(new_walker[j, k], walker_keep[k], nstates, k) < unif[rcnt_u]:  # reject -- metropolis
                    elif acceptance_nonuniform(new_walker[j, k], walker_keep[k], nstates, k) < unif[rcnt_u]:  # reject -- metropolis
                    #elif acceptance_jeffrey(new_walker[j, k], walker_keep[k], nstates, k) < unif[rcnt_u]:  # reject -- metropolis
                        new_walker[j, :] = walker_keep.copy()
                        reject[k] += 1
                    rcnt_u += 1
                walker_keep = new_walker[j, :].copy()
        
        
        # replace walker having minumum likelihood with a new better walker
        walker[min_index, :] = new_walker[nsweep, :].copy()
        logweight += np.log(nwalkers / (nwalkers + 1))
        
        #calculate Z_remain
        evidence_remain = -np.inf
        for j in range(nwalkers):
            evidence_remain = sp.special.logsumexp([evidence_remain, walker[j, nparams]])
        evidence_remain += logweight
        
        # check stop ratio
        if evidence_remain - evidence < np.log(stop_ratio):
            evidence = sp.special.logsumexp([evidence, evidence_remain])  # final evidence
            runningtime = time.time() - start
            # print('nstate = {} done, {} seconds'.format(nstates, runningtime))
            if log_q is not None:
                log_q.put(f'nstate = {nstates} done, {runningtime} seconds')
            break
        
        if i%100 == 0:
            if log_q is not None:
                log_q.put(f'log(Z_remain / Z): {evidence_remain - evidence}')
            # print('i:', i)
            # print('log(Z_remain / Z):', evidence_remain - evidence)
        
        stepsize = adjust_stepsize(stepsize, reject, nsweep, reject_rate)
        
    return evidence, series[:i+1, :], walker

def mp_nested_sampling(np.ndarray[np.int64_t, ndim = 1] indices,
                       int nwalkers,
                       int imax,
                       double stop_ratio,
                       double reject_rate,
                       int nsweep,
                       str path,
                       object log_q = None):

    cdef np.ndarray[np.float64_t, ndim = 1] evidence
    cdef np.ndarray[np.float64_t, ndim = 2] prob, mle
    cdef np.ndarray[np.float64_t, ndim = 2] series, temp_walker, walker, totalseries
    cdef int i, j, index, max_index, model
    
    prob = np.zeros((len(indices), 3))
    mle = np.zeros((len(indices), 9))
    fnames = np.genfromtxt(f'{path}/filenames.txt', dtype=str)
    
    for i, index in enumerate(indices):
        evidence = np.ones(3) * (-np.inf)
        trj = np.loadtxt('{}/{}'.format(path, fnames[index-1]))
        # trj = np.loadtxt('{}/{}trj{}.txt'.format(path, protein, index))
        obs_v = np.diff(trj)[:, None]
        
        for j in range(3):
            evidence[j], series, temp_walker = nested_main(obs_v, nwalkers, imax, stop_ratio, reject_rate, nsweep, j+1, log_q)
            totalseries = np.zeros((len(series) + len(temp_walker), (j+1) ** 2 + 1))  # j+1 : nstates
            totalseries[:len(series), :] = series.copy()
            totalseries[len(series):, :] = temp_walker.copy()
            
            createDirectory('{}/results'.format(path))
            
            #save the sampled parameters of each state
            with open('{}/results/{}totalseries{}state.pkl'.format(path, index, j+1), 'wb') as f:
                pickle.dump(totalseries, f)
            
            if j == np.argmax(evidence):
                walker = temp_walker.copy()
            #if j == 0:
            #    walker = temp_walker.copy()
            #elif j > 0 and evidence[j] > evidence[j-1]:
            #    walker = temp_walker.copy()
        
        #save the model probabilities
        prob[i, :] = evidence - np.max(evidence)  # unnormalized log probabilities
        prob[i, :] = np.exp(prob[i, :]) / np.exp(sp.special.logsumexp(prob[i, :]))  # normalized probabilities
        
        with open('{}/results/evidence{}.pkl'.format(path, index), 'wb') as f:
            pickle.dump(evidence, f)
        with open('{}/results/prob{}.pkl'.format(path, index), 'wb') as f:
            pickle.dump(prob[i, :], f)
        """
        model = np.argmax(prob[i, :]) + 1 #optimal nstates
        
        #save the maximum likelihood estimator of the best-fit model
        max_index = np.argmax(walker[:, model ** 2])
        mle[i, :model ** 2] = walker[max_index, :model ** 2].copy()
        with open('{}/results/MLE{}.pkl'.format(path, index), 'wb') as f:
            pickle.dump(walker[max_index, :], f)
        """     
    return 0

def NS_Params(str prior_type,   # uniform / jeffrey
              str selection_method,   # Bayesian / AIC / BIC
              str path,
              int fnum
             ):
    cdef int i, j, k, nstates, nparams
    cdef np.ndarray[np.float64_t, ndim = 2] series, P, alpha, IC
    cdef np.ndarray[np.float64_t, ndim = 1] prior, D, posterior, MAP, MLE, IC_bayes
    cdef double prior_D, prior_P, norm
    cdef np.ndarray[np.int64_t, ndim = 1] model
    
    
    #### model selection
        
    model = np.zeros(fnum, dtype = int)
    if selection_method == 'AIC':
        with open('{}/results/AIC.pkl'.format(path), 'rb') as f:
            IC = pickle.load(f)
        model = np.argmin(IC, axis = 1).astype(int) + 1
            
    elif selection_method == 'Bayesian':
        for i in range(fnum):
            with open('{}/results/prob{}.pkl'.format(path, i + 1), 'rb') as f:
                IC_bayes = pickle.load(f)
            model[i] = np.argmax(IC_bayes) + 1
    
    #calculate prior values of nested samples
    
    if prior_type == 'uniform':
        for i in range(fnum):
            nstates = model[i]
            nparams = nstates ** 2
            
            if nstates != 1:
                alpha = np.ones((nstates, nstates)) * (1 + 0.5 / (nstates - 1))
                alpha[range(nstates), range(nstates)] = 10.5
            
                with open('{}/results/{}totalseries{}state.pkl'.format(path, i+1, nstates), 'rb') as f:
                    series = pickle.load(f)
                prior = np.zeros(series.shape[0])
                posterior = np.zeros(series.shape[0])
            
                for j in range(series.shape[0]):
                    D = series[j, :nstates].copy()
                    prior_D = np.log(1/(D_max - D_min)) * nstates
                
                    _, P = convert_theta(series[j], nstates)
                    prior_P = 0
                    for k in range(nstates):
                        prior_P += np.log(sp.stats.dirichlet.pdf(P[:, k], alpha[:, k]))
            
                    prior[j] = prior_D + prior_P
                    posterior[j] = prior[j] + series[j, nparams]
            
            elif nstates == 1:
                with open('{}/results/{}totalseries{}state.pkl'.format(path, i+1, nstates), 'rb') as f:
                    series = pickle.load(f)
                prior = np.zeros(series.shape[0])
                posterior = np.zeros(series.shape[0])
                
                for j in range(series.shape[0]):
                    D = series[j, :nstates].copy()
                    prior_D = np.log(1/(D_max - D_min)) * nstates
            
                    prior[j] = prior_D
                    posterior[j] = prior[j] + series[j, nparams]
            
            MAP = series[np.argmax(posterior), :].copy()
            MLE = series[np.argmax(series[:, nparams]), :].copy()
            with open('{}/results/MAP{}_{}.pkl'.format(path, i+1, selection_method), 'wb') as f:
                pickle.dump(MAP, f)
            with open('{}/results/MLE{}_{}.pkl'.format(path, i+1, selection_method), 'wb') as f:
                pickle.dump(MLE, f)
                
    return 0

def sample_sorting(np.ndarray[np.float64_t, ndim = 1] theta, int nstates):
    cdef int i, j
    cdef np.ndarray[np.float64_t, ndim = 1] sigma
    cdef np.ndarray[np.float64_t, ndim = 2] P
    cdef double temp_sigma
    cdef np.ndarray[np.float64_t, ndim = 1] temp_P
    
    sigma = convert_theta(theta, nstates)[0].flatten()
    P = convert_theta(theta, nstates)[1]
    if nstates > 1:
        for i in range(1, nstates):
            for j in range(0, nstates - i):
                if sigma[j] > sigma[j+1]:
                    #switch sigma
                    temp_sigma = sigma[j]
                    sigma[j] = sigma[j+1]
                    sigma[j+1] = temp_sigma
                    #switch P
                    temp_P = P[j, :].copy()
                    P[j, :] = P[j+1, :].copy()
                    P[j+1, :] = temp_P.copy()
                    temp_P = P[:, j].copy()
                    P[:, j] = P[:, j+1].copy()
                    P[:, j+1] = temp_P.copy()
    return sigma ** 2 / 2 / unit_T, P

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