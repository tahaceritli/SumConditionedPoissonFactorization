import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

def sample_multinomial(I, J, Ns, phi):
    assert phi.shape[0] == Ns.size
    samples = np.array([np.random.multinomial(n, phi_i) for phi_i, n in zip(phi, Ns)])
    # Indices for easy COO sparse matrix construction for each latent factor k, and an array of data arrays
    return I, J, [samples[:, k] for k in range(phi.shape[1])]

# random-related routines
def rand_gen_gamma(a,b,N=1):
    return np.random.gamma(a,b,size=N)

def rand_gen_mult(v, x=1, max_value=1):    
    return np.random.multinomial(x, v, 1)[0]

def rand_gen_poisson(v, max_value=1):
    sample = np.random.poisson(v)
    while sample > max_value:
        sample = np.random.poisson(v)
    return sample

def poisson_log_likelihood(k, l):    
    return st.poisson.logpmf(k,l)

def be_log_likelihood(p ,x):   
    return st.bernoulli(p).logpmf(x).sum()
                
def bin_log_likelihood(x, n, p):   
    return st.binom.logpmf(x, n, p).sum()

def get_mle(lambda1, lambda2, max_X):
    mle = np.zeros(max_X+1, dtype=float)
    for i in np.arange(max_X+1):
        mle[i] = (np.power(lambda1, i)/(np.math.factorial(i))) * (np.power(lambda2, max_X-i)/(np.math.factorial(max_X-i)))
    
    return mle/mle.sum()

def get_matrix_mle(matrix_lambda1, matrix_lambda2, max_X):    
    I, J = matrix_lambda1.shape
    mle = np.zeros((I,J,max_X+1), dtype=float)
    for i in np.arange(I):
        for j in np.arange(J):
            mle[i,j,:] = get_mle(matrix_lambda1[i,j], matrix_lambda2[i,j], max_X)
    return mle

# loading pml arrays
def load_array(filename):
    X = np.loadtxt(filename)
    dim = int(X[0]);
    size = []
    for i in range(dim):
        size.append(int(X[i+1]));    
    X = np.reshape(X[dim+1:], size, order='F')
    return X;
        
# saving in pml format
def save_array(filename, X, format = '%.6f'):
    with open(filename, 'w') as f:
        dim = len(X.shape)
        f.write('%d\n' % dim)
        for i in range(dim):
            f.write('%d\n' % X.shape[i])
        temp = X.reshape(np.product(X.shape), order='F')
        for num in temp:
            f.write(str(num)+"\n")
            
# plots
def plot_vector(data, title='Title', xlabel='xlabel', ylabel='ylabel'):
    plt.figure()
    plt.plot(range(data.shape[0]), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_matrix(X, title='Title', xlabel='xlabel', ylabel='ylabel', figsize=None, vmax_=None, yticklabels=None, cmap=plt.cm.gray_r):
    if figsize is None:
        plt.figure(figsize=(18,6))
    else:
        plt.figure(figsize=figsize)
    if vmax_ is None:
        VMAX = np.max(X)
    else:
        VMAX = vmax_
    if yticklabels is None:
        pass
    else:
        plt.yticks(range(len(yticklabels)), yticklabels)
    plt.imshow(X, interpolation='none', vmax=VMAX, vmin=0, aspect='auto', cmap=cmap)
    plt.colorbar()
    #    plt.set_cmap('gray_r')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    
def plot_matrices(data1, data2, title1='title1', title2='title2', xlabel='xlabel', ylabel='ylabel', vmax_=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
    data = [data1, data2]
    title = [title1, title2]
    for i in range(2):
        current_ax = axes[i]
        if vmax_ == None:
            VMAX = np.max(data[i])
        else:
            VMAX = vmax_
        im = current_ax.imshow(data[i], interpolation='none', cmap='gray_r', aspect='auto', vmax=VMAX, vmin=0)
        cbar = fig.colorbar(im, ax=current_ax)   
        current_ax.set_xlabel(xlabel, fontsize=10)
        current_ax.set_ylabel(ylabel, fontsize=10)
        current_ax.set_title(title[i], fontsize=10)
        
def normalize(data, axis):
    D = data.copy()
    I,J = D.shape
    if axis==0:
        for i in np.arange(I):
            D[i,:] = D[i,:]/D[i,:].sum()
    elif axis==1:
        for j in np.arange(J):
            D[:,j] = D[:,j]/D[:,j].sum()
    return D
    
def binary_random_mask_generator(M=30, N=150, p_miss=0.2):
    Mask = np.array(np.random.rand(M,N)>p_miss,dtype=float)

    Mask_nan = Mask.copy()
    Mask_nan[Mask==0] = np.nan
    
    return Mask, Mask_nan
    
def binary_random_matrix_generator1(M=30, N=150, p_on=0.3, p_switch=0.25):
    Y = np.zeros((M,N))
    y = np.array(np.random.rand(M,1)<p_on, dtype=float)
    for i in range(N):
        if np.random.rand()<p_switch:
            y = np.array(np.random.rand(M,1)<p_on, dtype=float)

        Y[:,i] = y.reshape(1,M)
    
    return Y

# Generate a catalog and reuse these
def binary_random_matrix_generator2(R=10, M=30, N=150, p_on=0.3, p_switch=0.25):
    Y = np.zeros((M,N))
    Catalog = np.array(np.random.rand(M,R)<p_on, dtype=float)
    idx = np.random.choice(range(R))
    for i in range(N):
        if np.random.rand()<p_switch:
            idx = np.random.choice(range(R))

        Y[:,i] = Catalog[:,idx].reshape(1,M)
    
    return Y

# Generate a catalog and reuse pairwise
def binary_random_matrix_generator3(R=10, M=30, N=150, p_on=0.3, p_switch=0.25):
    Y = np.zeros((M,N))
    Catalog = np.random.rand(M,R)<p_on
    
    sz = 2
    
    idx = np.random.choice(range(R), size=sz, replace=True)
    y = np.ones((1,M))<0
    for i in range(sz): 
        y = np.logical_or(y, Catalog[:,idx[i]])
    
    for i in range(N):
        if np.random.rand()<p_switch:
            idx = np.random.choice(range(R), size=sz, replace=True)
            y = np.ones((1,M))<0
            for i in range(sz): 
                y = np.logical_or(y, Catalog[:,idx[i]])
                
        Y[:,i] = y.reshape(1,M)
    
    return Y