import numpy as np
import scipy.stats as st

# returns MAP or ML estimates 
def em_scpf(N, X, M, W, H, FIX_W=None, A=None, B=None, MAX_ITER=500, PRINT_PERIOD=100):
    eps = 1e-24                             # for division by zero
    CONVERGENCE_DIFF = 1e-6
    K, I, J = X.shape                    
    R = W.shape[2]                          # latent dimension
    
    if A is None: 
        # ML estimate
        A = 0.    
        B = 1.
    else:         
        # for MAP estimate
        if B is None:
            B = 1./np.sqrt(R)
    print('A = ', A, '\nB = ', B)
    ones_W = np.ones(shape=(K, I, J))
    ones_H = np.ones(shape=(K, I, J))   
    m_indices = np.where(M == 1.)           # nonmissing entries
    logP = ["-inf"]                         # loglikelihoods
    n = np.array([N for k in range(K)]) 
    N_tilde = N - (M*X).sum(0)               # residual matrix
    MX = M*X                                # masked matrix  
    one_minus_MN_tilde = np.array([(1-M[k]) * N_tilde for k in range(K)])
    
    for epoch in range(MAX_ITER):
        
        X_hat = np.array([ np.dot(W[k], H[k]) for k in range(K)]) 
        Qx = MX/X_hat     
        QxH =   np.array([np.dot(Qx[k], H[k].T) for k in range(K)])  
        
        N_tilde_hat = sum([ (1-M[k]) * np.dot(W[k], H[k]) for k in range(K) ]) + eps                               
        Qr = one_minus_MN_tilde/N_tilde_hat      
        QrH = np.array([np.dot(Qr[k], H[k].T) for k in range(K)])      
        W = (A + W * (QxH + QrH)) / ( (A/B) + np.array([np.dot(ones_W[k], H[k].T) for k in range(K)]))
        if FIX_W is not None:
            W[0] = FIX_W
        
        X_hat = np.array([ np.dot(W[k], H[k]) for k in range(K)])     
        Qx = MX/X_hat   
        QxW = np.array([np.dot(W[k].T, Qx[k]) for k in range(K)])
        N_tilde_hat = sum([ (1-M[k]) * np.dot(W[k], H[k]) for k in range(K) ]) + eps
        Qr = one_minus_MN_tilde/N_tilde_hat      
        QrW = np.array([np.dot(W[k].T, Qr[k]) for k in range(K)])
        H = (A + H * (QxW + QrW)) / ( (A/B) + np.array([np.dot(W[k].T, ones_H[k]) for k in range(K)]))

        p = np.array([X_hat[k]/X_hat.sum(0) for k in range(K)])
        
        logP.append(st.binom.logpmf(X[m_indices], n[m_indices], p[m_indices]).sum())                
        if epoch % PRINT_PERIOD == 0:   
            print(epoch, logP[-1]) 

        # checking convergence
        if len(logP) >= 3:
            if logP[-1] - logP[-2] < CONVERGENCE_DIFF:
                break
       
    return W, H, logP