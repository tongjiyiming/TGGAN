from collections import Counter
from itertools import product
import numpy as np
from numpy import sqrt
from sklearn.metrics.pairwise import rbf_kernel
import scipy
import scipy.stats
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import os


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

def Time_Plot(fake_graphs, real_walks, train_edges, test_edges, N, t_end, output_directory, _it):

    fake_walks = fake_graphs.reshape(-1, 3)
    fake_mask = fake_walks[:, 0] > -1
    fake_walks = fake_walks[fake_mask]

    real_walks = np.array(real_walks).reshape(-1, 3)
    real_mask = real_walks[:, 0] > -1
    real_walks = real_walks[real_mask]

    # truth_train_walks = train_edges[:, 1:3]
    truth_train_time = train_edges[:, 3:]
    truth_train_res_time = t_end - truth_train_time
    truth_train_walks = np.concatenate([train_edges[:, 1:3], truth_train_res_time], axis=1)
    truth_train_x_t0 = np.c_[np.zeros((len(train_edges), 1)), truth_train_res_time]
    truth_train_x_t0 = np.r_[truth_train_x_t0, np.ones((len(train_edges), 2))]

    truth_test_time = test_edges[:, 3:]
    truth_test_res_time = t_end - truth_test_time
    truth_test_walks = np.c_[test_edges[:, 1:3], truth_test_res_time]
    truth_test_x_t0 = np.c_[np.zeros((len(test_edges), 1)), truth_test_res_time]
    truth_test_x_t0 = np.r_[truth_test_x_t0, np.ones((len(test_edges), 2))]

    # plot edges time series for qualitative evaluation
    fake_e_list, fake_e_counts = np.unique(fake_walks[:, 0:2], return_counts=True, axis=0)
    real_e_list, real_e_counts = np.unique(real_walks[:, 0:2], return_counts=True, axis=0)
    truth_train_e_list, truth_train_e_counts = np.unique(truth_train_walks[:, 0:2], return_counts=True,
                                                         axis=0)
    truth_test_e_list, truth_test_e_counts = np.unique(truth_test_walks[:, 0:2], return_counts=True,
                                                       axis=0)
    truth_e_list, truth_e_counts = np.unique(
        np.r_[truth_test_walks[:, 0:2], truth_test_walks[:, 0:2]], return_counts=True, axis=0)
    n_e = len(truth_e_list)

    fig = plt.figure(figsize=(2 * 9, 2 * 9))
    fig.suptitle('Truth, Real, and Fake temporal edges comparisons')
    dx = 0.3
    dy = dx
    zpos = 0

    fake_ax = fig.add_subplot(221, projection='3d')
    fake_ax.bar3d(fake_e_list[:, 0], fake_e_list[:, 1], zpos, dx, dy, fake_e_counts)
    fake_ax.set_xlim([0, N])
    fake_ax.set_ylim([0, N])
    fake_ax.set_xticks(range(N))
    fake_ax.set_yticks(range(N))
    fake_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    fake_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    fake_ax.set_title('Total fake edges number: {}'.format(len(fake_e_list)))
    fake_ax.set_xlabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    fake_ax.set_ylabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    fake_ax.set_zlabel('Total counts', fontdict={'fontsize': 14})

    real_ax = fig.add_subplot(222, projection='3d')
    real_ax.bar3d(real_e_list[:, 0], real_e_list[:, 1], zpos, dx, dy, real_e_counts)
    real_ax.set_xlim([0, N])
    real_ax.set_ylim([0, N])
    real_ax.set_xticks(range(N))
    real_ax.set_yticks(range(N))
    real_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    real_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    real_ax.set_title('Total sampled edges number: {}'.format(len(real_e_list)))
    real_ax.set_xlabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    real_ax.set_ylabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    real_ax.set_zlabel('Total counts', fontdict={'fontsize': 14})

    truth_ax = fig.add_subplot(223, projection='3d')
    truth_ax.bar3d(truth_train_e_list[:, 0], truth_train_e_list[:, 1], zpos, dx, dy,
                   truth_train_e_counts)
    truth_ax.set_xlim([0, N])
    truth_ax.set_ylim([0, N])
    truth_ax.set_xticks(range(N))
    truth_ax.set_yticks(range(N))
    truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    truth_ax.set_title('Ground truth train edges number: {}'.format(len(truth_train_e_list)))
    truth_ax.set_xlabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    truth_ax.set_ylabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    truth_ax.set_zlabel('Total counts', fontdict={'fontsize': 14})

    truth_ax = fig.add_subplot(224, projection='3d')
    truth_ax.bar3d(truth_test_e_list[:, 0], truth_test_e_list[:, 1], zpos, dx, dy, truth_test_e_counts)
    truth_ax.set_xlim([0, N])
    truth_ax.set_ylim([0, N])
    truth_ax.set_xticks(range(N))
    truth_ax.set_yticks(range(N))
    truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(N)])
    truth_ax.set_title('Ground truth test edges number: {}'.format(len(truth_test_e_list)))
    truth_ax.set_xlabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    truth_ax.set_ylabel('{} Nodes'.format(N), fontdict={'fontsize': 14})
    truth_ax.set_zlabel('Total counts', fontdict={'fontsize': 14})

    plt.tight_layout()
    plt.savefig('{}/iter_{}_edges_counts_validation.eps'.format(output_directory, _it + 1), dpi=200)
    plt.close()

    fig, ax = plt.subplots(n_e, 4, figsize=(4 * 6, n_e * 4))
    i = 0
    for j, e in enumerate(truth_e_list):
        real_ax = ax[i + j, 0]
        real_mask = np.logical_and(real_walks[:, 0] == e[0], real_walks[:, 1] == e[1])
        real_times = real_walks[real_mask][:, 2]
        real_ax.hist(real_times, range=[0, 1], bins=100)
        real_ax.set_title('Real start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
            [int(v) for v in e], real_times.mean(), real_times.std()))

        fake_ax = ax[i + j, 1]
        fake_mask = np.logical_and(fake_walks[:, 0] == e[0], fake_walks[:, 1] == e[1])
        fake_times = fake_walks[fake_mask][:, 2]
        fake_ax.hist(fake_times, range=[0, 1], bins=100)
        fake_ax.set_title('Fake start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
            [int(v) for v in e], fake_times.mean(), fake_times.std()))

        truth_train_ax = ax[i + j, 2]
        truth_train_mask = np.logical_and(truth_train_walks[:, 0] == e[0],
                                          truth_train_walks[:, 1] == e[1])
        truth_train_times = truth_train_walks[truth_train_mask][:, 2]
        truth_train_ax.hist(truth_train_times, range=[0, 1], bins=100)
        truth_train_ax.set_title(
            'Ground truth train start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], truth_train_times.mean(), truth_train_times.std()))

        truth_test_ax = ax[i + j, 3]
        truth_test_mask = np.logical_and(truth_test_walks[:, 0] == e[0], truth_test_walks[:, 1] == e[1])
        truth_test_times = truth_test_walks[truth_test_mask][:, 2]
        truth_test_ax.hist(truth_test_times, range=[0, 1], bins=100)
        truth_test_ax.set_title(
            'Ground truth test start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], truth_test_times.mean(), truth_test_times.std()))

    plt.tight_layout()
    plt.savefig('{}/iter_{}_validation.eps'.format(output_directory, _it + 1), dpi=200)
    plt.close()


if __name__ == "__main__":

    dataset = 'auth'

    if dataset == 'auth': 
        N = 28
        user_id = 0
        _it = 23000
        fake_file = './outputs-auth-user-0/20191028-222439_assembled_graph_iter_{}.npz'.format(_it)
    else: 
        N = 91
        user_id = 4
        _it = 44000
        fake_file = './outputs-metro-user-4/20191103-235308_assembled_graph_iter_{}.npz'.format(_it)

    file = "data/{}_user_{}.txt".format(dataset, user_id)
    edges = np.loadtxt(file)
    t_end = 1.
    train_ratio = 0.9
    output_directory = 'outputs-{}'.format(dataset)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    train_edges, test_edges = Split_Train_Test(edges, train_ratio)

    res = np.load(fake_file)
    fake_graphs = res['fake_graphs']
    real_walks = res['real_walks']
    print('fake_graphs', fake_graphs.shape)
    print('real_walks', real_walks.shape)
    # save to matlab .mat data file
    savemat(mdict={'fake_graphs': fake_graphs, 'real_walks': real_walks, 'train_edges':train_edges, 'test_edges':test_edges},
    file_name='{}/{}_fake_graphs_iter_{}'.format(output_directory, dataset, _it))

    Time_Plot(fake_graphs, real_walks, train_edges, test_edges, N, t_end, output_directory, _it)

