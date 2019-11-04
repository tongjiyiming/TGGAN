from collections import Counter
from itertools import product
import numpy as np
from numpy import sqrt
from sklearn.metrics.pairwise import rbf_kernel
import scipy
import scipy.stats

# evaluation metrics
n_nodes = 91

def My_Node_Counter(sequences):
    """
    -1 should not be included
    """
    li = []
    for s in sequences:
        li.extend(s)
    c = Counter(li)
    if c.get(-1) is not None: c.pop(-1)
    return c


def My_Edge_Counter(sequences):
    """
    -1 can be second node of a pair to indicate a sequence end
    """
    li = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] > -1: li.append((seq[i], seq[i + 1]))
    c = Counter(li)
    return c


# Direct evaluation
def Node_Variety_Metro(sequences_1, sequences_2):
    """
    Variety: number of unique in real V.S. number of generated

    Parameters
    ------------
    sequences_1: list of list. could be the real daily sequences or the sampled walks
    sequences_2: list of list. same

    Returns
    ------------
    variety: Jaccard Similarity of unique nodes of two sequences lists
    """
    set_1 = set(My_Node_Counter(sequences_1).keys())
    set_2 = set(My_Node_Counter(sequences_2).keys())
    sim = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
    return sim


def Edge_Variety_Metro(sequences_1, sequences_2):
    """
    Variety: number of unique in real V.S. number of generated

    Parameters
    ------------
    sequences_1: list of list of sequence
    sequences_2: list of list of sequence

    Returns
    ------------
    variety: Jaccard Similarity of unique edges of two sequences lists
    """
    set_1 = set(My_Edge_Counter(sequences_1).keys())
    set_2 = set(My_Edge_Counter(sequences_2).keys())
    sim = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
    return sim


def Node_Novelty_Metro(sequences_train, sequences_generated):
    """
    Parameters
    ------------
    sequences_1: list of list of sequence
    sequences_2: list of list of sequence

    Returns
    ------------
    novelty: percentage of new nodes not in training set among all generates samples
    """
    set_train = set(My_Node_Counter(sequences_train).keys())
    set_generated = set(My_Node_Counter(sequences_generated).keys())
    set_new = set_generated.difference(set_train)
    return len(set_new) / len(set_generated)


def Edge_Novelty_Metro(sequences_train, sequences_generated):
    """
    Parameters
    ------------
    sequences_train: list of list. could be the real daily sequences or the sampled walks
    sequences_generated: list of list. same

    Returns
    ------------
    novelty: percentage of new nodes not in training set among all generates samples
    """
    set_train = set(My_Edge_Counter(sequences_train).keys())
    set_generated = set(My_Edge_Counter(sequences_generated).keys())
    set_new = set_generated.difference(set_train)
    return len(set_new) / len(set_generated)


def Node_JS_Diverg_Metro(sequences_1, sequences_2, n_nodes=91):
    """
    Divergence: divergence of node distribution using Jensen-Shannon-divergence

    Parameters
    ------------
    sequences_1: list of list of sequence
    sequences_2: list of list of sequence

    Returns
    ------------
    variety: Jaccard Similarity of unique nodes of two sequences lists
    """
    counter_1 = My_Node_Counter(sequences_1)
    counts_1 = []
    for i in range(n_nodes):
        if counter_1.get(i) is None:
            counts_1.append(0)
        else:
            counts_1.append(counter_1.get(i))
    counts_1 = np.array(counts_1)

    counter_2 = My_Node_Counter(sequences_2)
    counts_2 = []
    for i in range(n_nodes):
        if counter_2.get(i) is None:
            counts_2.append(0)
        else:
            counts_2.append(counter_2.get(i))
    counts_2 = np.array(counts_2)

    M = (counts_1 + counts_2) / 2
    js = 0.5 * scipy.stats.entropy(counts_1, M) + 0.5 * scipy.stats.entropy(counts_2, M)
    return js


def Edge_JS_Diverg_Metro(sequences_1, sequences_2, n_nodes=91):
    """
    Divergence: divergence of node distribution using Jensen-Shannon-divergence

    Parameters
    ------------
    sequences_1: list of list of sequence
    sequences_2: list of list of sequence

    Returns
    ------------
    variety: Jaccard Similarity of unique nodes of two sequences lists
    """
    counter_1 = My_Edge_Counter(sequences_1)
    counts_1 = []
    for pair in product(range(n_nodes), range(-1, n_nodes)):
        if counter_1[pair] is None:
            counts_1.append(0)
        else:
            counts_1.append(counter_1[pair])
    counts_1 = np.array(counts_1)

    counter_2 = My_Edge_Counter(sequences_2)
    counts_2 = []
    for pair in product(range(n_nodes), range(-1, n_nodes)):
        if counter_2[pair] is None:
            counts_2.append(0)
        else:
            counts_2.append(counter_2[pair])
    counts_2 = np.array(counts_2)

    M = (counts_1 + counts_2) / 2
    js = 0.5 * scipy.stats.entropy(counts_1, M) + 0.5 * scipy.stats.entropy(counts_2, M)
    return js

def MMD_3_Sample_Test(X, Y, Z, sigma=-1, SelectSigma=2, computeMMDs=False):
    '''Performs the relative MMD test which returns a test statistic for whether Y is closer to X or than Z.
    See http://arxiv.org/pdf/1511.04581.pdf
    The bandwith heuristic is based on the median heuristic (see Smola,Gretton).
    '''
    if (sigma < 0):
        # Similar heuristics
        if (SelectSigma > 1):
            siz = np.min((1000, X.shape[0]))
            sigma1 = kernelwidthPair(X[0:siz], Y[0:siz])
            sigma2 = kernelwidthPair(X[0:siz], Z[0:siz])
            sigma = (sigma1 + sigma2) / 2.
        else:
            siz = np.min((1000, X.shape[0] * 3))
            Zem = np.r_[X[0:siz / 3], Y[0:siz / 3], Z[0:siz / 3]]
            sigma = kernelwidth(Zem)

    Kyy = grbf(Y, Y, sigma)
    Kzz = grbf(Z, Z, sigma)
    Kxy = grbf(X, Y, sigma)
    Kxz = grbf(X, Z, sigma)
    Kyynd = Kyy - np.diag(np.diagonal(Kyy))
    Kzznd = Kzz - np.diag(np.diagonal(Kzz))
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    u_yy = np.sum(Kyynd) * (1. / (n * (n - 1)))
    u_zz = np.sum(Kzznd) * (1. / (r * (r - 1)))
    u_xy = np.sum(Kxy) / (m * n)
    u_xz = np.sum(Kxz) / (m * r)
    # Compute the test statistic
    t = u_yy - 2. * u_xy - (u_zz - 2. * u_xz)
    Diff_Var, Diff_Var_z2, data = MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz)

    pvalue = scipy.stats.norm.cdf(-t / np.sqrt((Diff_Var)))
    #  pvalue_z2=sp.stats.norm.cdf(-t/np.sqrt((Diff_Var_z2)))
    tstat = t / sqrt(Diff_Var)

    if (computeMMDs):
        Kxx = grbf(X, X, sigma)
        Kxxnd = Kxx - np.diag(np.diagonal(Kxx))
        u_xx = np.sum(Kxxnd) * (1. / (m * (m - 1)))
        MMDXY = u_xx + u_yy - 2. * u_xy
        MMDXZ = u_xx + u_zz - 2. * u_xz
    else:
        MMDXY = None
        MMDXZ = None
    return pvalue, tstat, sigma, MMDXY, MMDXZ


def MMD_Diff_Var(Kyy, Kzz, Kxy, Kxz):
    '''
    Compute the variance of the difference statistic MMDXY-MMDXZ
    See http://arxiv.org/pdf/1511.04581.pdf Appendix for derivations
    '''
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    Kyynd = Kyy - np.diag(np.diagonal(Kyy))
    Kzznd = Kzz - np.diag(np.diagonal(Kzz))

    u_yy = np.sum(Kyynd) * (1. / (n * (n - 1)))
    u_zz = np.sum(Kzznd) * (1. / (r * (r - 1)))
    u_xy = np.sum(Kxy) / (m * n)
    u_xz = np.sum(Kxz) / (m * r)

    # compute zeta1
    t1 = (1. / n ** 3) * np.sum(Kyynd.T.dot(Kyynd)) - u_yy ** 2
    t2 = (1. / (n ** 2 * m)) * np.sum(Kxy.T.dot(Kxy)) - u_xy ** 2
    t3 = (1. / (n * m ** 2)) * np.sum(Kxy.dot(Kxy.T)) - u_xy ** 2
    t4 = (1. / r ** 3) * np.sum(Kzznd.T.dot(Kzznd)) - u_zz ** 2
    t5 = (1. / (r * m ** 2)) * np.sum(Kxz.dot(Kxz.T)) - u_xz ** 2
    t6 = (1. / (r ** 2 * m)) * np.sum(Kxz.T.dot(Kxz)) - u_xz ** 2
    t7 = (1. / (n ** 2 * m)) * np.sum(Kyynd.dot(Kxy.T)) - u_yy * u_xy
    t8 = (1. / (n * m * r)) * np.sum(Kxy.T.dot(Kxz)) - u_xz * u_xy
    t9 = (1. / (r ** 2 * m)) * np.sum(Kzznd.dot(Kxz.T)) - u_zz * u_xz

    zeta1 = (t1 + t2 + t3 + t4 + t5 + t6 - 2. * (t7 + t8 + t9))

    zeta2 = (1 / m / (m - 1)) * np.sum((Kyynd - Kzznd - Kxy.T - Kxy + Kxz + Kxz.T) ** 2) - (
                u_yy - 2. * u_xy - (u_zz - 2. * u_xz)) ** 2

    data = dict({'t1': t1,
                 't2': t2,
                 't3': t3,
                 't4': t4,
                 't5': t5,
                 't6': t6,
                 't7': t7,
                 't8': t8,
                 't9': t9,
                 'zeta1': zeta1,
                 'zeta2': zeta2,
                 })
    # TODO more precise version for zeta2
    #    xx=(1/m^2)*sum(sum(Kxxnd.*Kxxnd))-u_xx^2
    # yy=(1/n^2)*sum(sum(Kyynd.*Kyynd))-u_yy^2
    # xy=(1/(n*m))*sum(sum(Kxy.*Kxy))-u_xy^2
    # xxy=(1/(n*m^2))*sum(sum(Kxxnd*Kxy))-u_xx*u_xy
    # yyx=(1/(n^2*m))*sum(sum(Kyynd*Kxy'))-u_yy*u_xy
    # zeta2=(xx+yy+xy+xy-2*(xxy+xxy +yyx+yyx))

    Var = (4. * (m - 2) / (m * (m - 1))) * zeta1
    Var_z2 = Var + (2. / (m * (m - 1))) * zeta2

    return Var, Var_z2, data


def grbf(x1, x2, sigma):
    '''Calculates the Gaussian radial base function kernel'''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2.T, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    return np.exp(-1. * h / (2. * pow(sigma, 2)))


def kernelwidthPair(x1, x2):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    mdist = np.median([i for i in h.flat if i])

    sigma = sqrt(mdist / 2.0)
    if not sigma: sigma = 1

    return sigma


def kernelwidth(Zmed):
    '''Alternative median heuristic when we cant partition the points
    '''
    m = Zmed.shape[0]
    k1 = np.expand_dims(np.sum((Zmed * Zmed), axis=1), 1)
    q = np.kron(np.ones((1, m)), k1)
    r = np.kron(np.ones((m, 1)), k1.T)
    del k1

    h = q + r
    del q, r

    # The norm
    h = h - 2. * Zmed.dot(Zmed.T)
    h = np.array(h, dtype=float)

    mdist = np.median([i for i in h.flat if i])

    sigma = sqrt(mdist / 2.0)
    if not sigma: sigma = 1

    return sigma


def MMD_unbiased(Kxx, Kyy, Kxy):
    # The estimate when distribution of x is not equal to y
    m = Kxx.shape[0]
    n = Kyy.shape[0]

    t1 = (1. / (m * (m - 1))) * np.sum(Kxx - np.diag(np.diagonal(Kxx)))
    t2 = (2. / (m * n)) * np.sum(Kxy)
    t3 = (1. / (n * (n - 1))) * np.sum(Kyy - np.diag(np.diagonal(Kyy)))

    MMDsquared = (t1 - t2 + t3)

    return MMDsquared


def Edge_MMD_Metro(real_daily_sequences, sampled_walks, generated_walks):
    counter_1 = My_Edge_Counter(real_daily_sequences)
    counts_1 = []
    for pair in product(range(n_nodes), range(-1, n_nodes)):
        if counter_1[pair] is None:
            counts_1.append(0)
        else:
            counts_1.append(counter_1[pair])
    counts_1 = np.array(counts_1)

    counter_2 = My_Edge_Counter(sampled_walks)
    counts_2 = []
    for pair in product(range(n_nodes), range(-1, n_nodes)):
        if counter_2[pair] is None:
            counts_2.append(0)
        else:
            counts_2.append(counter_2[pair])
    counts_2 = np.array(counts_2)

    counter_3 = My_Edge_Counter(generated_walks)
    counts_3 = []
    for pair in product(range(n_nodes), range(-1, n_nodes)):
        if counter_3[pair] is None:
            counts_3.append(0)
        else:
            counts_3.append(counter_3[pair])
    counts_3 = np.array(counts_3)

    pvalue, tstat, sigma, MMDXY, MMDXZ = MMD_3_Sample_Test(
        counts_1.reshape(-1, 1), counts_3.reshape(-1, 1), counts_2.reshape(-1, 1), computeMMDs=True)
    return MMDXY

if __name__ == "__main__":
    fake_graphs = np.load("./outputs-metro-user-4/")