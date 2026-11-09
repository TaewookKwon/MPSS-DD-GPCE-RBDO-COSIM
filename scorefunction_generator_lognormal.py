import numpy as np

def generate_score_function(zz, nSampleS, cova, stdZ, GPCE_CONFIG_OBJECTIVE):
    
    """
    zz : (nSampleS, nd) samples
    cova : (nd, nd) covariance matrix
    mu : (nd,) physical mean of the lognormal variables (MATLAB mu)
    par_mu : (nd,) mean of the lognormal variables in the log (Z) space (MATLAB par_mu)
    """
    
    nd = GPCE_CONFIG_OBJECTIVE['nd']
    
    mu = np.array([1] * nd)

    # Output array
    rssc = np.zeros((nSampleS, nd))

    # ----- Build the DM matrix -----
    DM = np.zeros((nd, nd))
    for i in range(nd):
        DM[i, i] = (mu[i]**2 + 2*(stdZ[i]**2)) / (mu[i]**3 + mu[i]*(stdZ[i]**2))

    par_mu = np.log(1**2/np.sqrt(stdZ[i]**2 + 1**2))  # mean in the log (Z) space
    
    # ----- Score function -----
    # Evaluated sample by sample, as in the original MATLAB loop
    cova_inv = np.linalg.inv(cova)   # cova\dist' → inv(cova) @ dist
    
    for L in range(nSampleS):
        X = zz[L, :nd]                          # X → zz[L]
        dist = np.log(X) - par_mu               # log(X) - par_mu
        score = DM @ (cova_inv @ dist)          # DM * (cova \ dist')
        rssc[L, :] = score                      # store

    return rssc