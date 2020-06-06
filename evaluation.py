from collections import Counter
from itertools import product
import numpy as np
from numpy import sqrt
import scipy
import scipy.stats
from scipy.io import savemat

import platform
import matplotlib
if platform.system() == 'Linux': matplotlib.use('Agg')
if platform.system() == 'Darwin': matplotlib.use('TkAgg')
matplotlib.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import os
from itertools import product

from sklearn.metrics.pairwise import rbf_kernel
import tacoma as tc
import teneto
from tacoma import degree_distribution
from tacoma.drawing import edge_activity_plot
from tacoma.analysis import plot_group_size_histogram, plot_group_durations, plot_social_trajectory
import networkx as nx

from func_timeout import func_timeout, func_set_timeout, FunctionTimedOut
import time


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


def MMD(X, Y, sigma=-1, SelectSigma=2):
    '''Performs the relative MMD test which returns a test statistic for whether Y is closer to X or than Z.
    See http://arxiv.org/pdf/1511.04581.pdf
    The bandwith heuristic is based on the median heuristic (see Smola,Gretton).
    '''
    Kyy = grbf(Y, Y, sigma)
    Kxy = grbf(X, Y, sigma)
    Kyynd = Kyy - np.diag(np.diagonal(Kyy))
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    u_yy = np.sum(Kyynd) * (1. / (n * (n - 1)))
    u_xy = np.sum(Kxy) / (m * n)

    Kxx = grbf(X, X, sigma)
    Kxxnd = Kxx - np.diag(np.diagonal(Kxx))
    u_xx = np.sum(Kxxnd) * (1. / (m * (m - 1)))
    MMDXY = u_xx + u_yy - 2. * u_xy
    return MMDXY


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


def Edge_MMD_Metro(real_daily_sequences, sampled_walks, generated_walks, n_nodes):
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


def Create_Temporal_Graph(edges, N, tmax, edge_contact_time=None):
    '''

    :param edges: a sampled temporal graph, with shape(None, 3), its time from t0 to t_end
    :param N:
    :param tmax:
    :param edge_contact_time:
    :return:
    '''

    el = tc.edge_lists()
    el.N = N
    el.tmax = tmax

    edges = edges[edges[:, 0] != edges[:, 1]]
    unique_times = np.sort(np.unique(edges[:, 2]))
    edge_times = [0.0]
    edge_list = [[]]
    for t in unique_times:
        edge_times.append(t - edge_contact_time)
        e = edges[edges[:, 2] == t][:, :2].astype(int)
        edge_list.append([(x, y) for x, y in e])
        if edge_contact_time is not None:
            edge_list.append([])
            edge_times.append(t)
    el.t = edge_times
    el.edges = edge_list
    return el


def Plot_Graph(tn, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig, ax = edge_activity_plot(tn,
                                 ax=ax,
                                 alpha=1.0,  # opacity
                                 linewidth=3,
                                 )
    ax.set_xlim([0, tn.tmax])
    plt.tight_layout()
    plt.savefig(file_name)


class Graphs:
    def __init__(self, data, N, tmax, edge_contact_time):
        self.N = N
        self.tmax = tmax
        self.edge_contact_time = edge_contact_time

        graph_list = []
        all_d = np.unique(data[:, 0])
        for d in all_d:
            one_graph = data[data[:, 0] == d][:, 1:]
            tg = Create_Temporal_Graph(edges=one_graph, N=N, tmax=tmax,
                                       edge_contact_time=edge_contact_time)
            graph_list.append(tg)
        self.graph_list = graph_list

        all_edges = np.unique(data[:, 1:3], axis=0)
        edge_time_set = dict()
        for e in all_edges:
            times = data[(data[:, 1] == e[0]) & (data[:, 2] == e[1])][:, 3]
            edge_time_set[tuple(e)] = times
        self.edge_time_set = edge_time_set

        group_metric_results = []
        for one_graph in self.graph_list:
            try:
                result = tc.measure_group_sizes_and_durations(one_graph)
                group_metric_results.append(result)
            except ValueError as e:
                # print('error encounter: \n{} \n ignored!'.format(e))
                continue
        self.group_metric_results = group_metric_results

    def Sample_Average_Degree_Distribution(self):
        sample_avg_degree = []
        for one_graph in self.graph_list:
            avg_degree = tc.degree_distribution(one_graph)
            sample_avg_degree.append(avg_degree)
        return np.array(sample_avg_degree)

    def Mean_Average_Degree_Distribution(self):
        return self.Sample_Average_Degree_Distribution().mean(axis=0)

    def Edge_Counts(self):
        return tc.edge_counts(self.graph_list[0])

    def Sample_Mean_Degree(self):
        sample_mean_degree = []
        for one_graph in self.graph_list:
            t, mean_k = tc.mean_degree(one_graph)
            mean_degree = tc.time_average(t, mean_k, one_graph.tmax)
            sample_mean_degree.append(mean_degree)
        return np.array(sample_mean_degree)

    def Mean_Mean_Degree(self):
        return self.Sample_Mean_Degree().mean()

    def Plot_Contact_Coverage(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for d in range(len(self.graph_list)):
            try:
                t, C = tc.contact_coverage(self.graph_list[d])
                ax.scatter(t, C, c='blue', alpha=0.4)
            except:
                continue
        plt.show()

    def Sample_Group_Size_Distribution(self):
        m = len(self.group_metric_results)
        group_size_distribution = np.zeros((m, self.N))
        for i in range(m):
            result = self.group_metric_results[i]
            g, N_g = tc.group_size_histogram(result)
            # N_g = N_g / N_g.sum()
            g = g.astype(int)
            group_size_distribution[i, g] = N_g
        return group_size_distribution

    def Sample_Group_Duration(self, bins):
        m = len(self.group_metric_results)
        group_duration = np.zeros((m, bins * self.N))
        for i in range(m):
            result = self.group_metric_results[i]
            # g, N_g = tc.group_size_histogram(result)
            # N_g = N_g / N_g.sum()
            # g = g.astype(int)
            # print(result.group_durations[3])
            all_counts = []
            for j in range(self.N):
                counts, _ = np.histogram(result.group_durations[j], bins=bins, range=[0., 1.])
                all_counts.append(counts)
            group_duration[i, :] = np.array(all_counts).reshape(1, -1)[0]
        return group_duration

    def Sample_Average_Group_Size(self):
        sample_avg_size = []
        for result in self.group_metric_results:
            try:
                mean_g = tc.mean_group_size(result)
                # print('mean_g', mean_g)
                sample_avg_size.append(mean_g)
            except ValueError as e:
                # print('error encounter: \n{} \n ignored!'.format(e))
                continue
        return np.array(sample_avg_size)

    def Mean_Average_Group_Size_Distribution(self):
        return self.Sample_Average_Group_Size().mean()

    def Sample_Mean_Group_Number(self):
        sample_avg_number = []
        for result in self.group_metric_results:
            try:
                mean_c = tc.mean_number_of_groups(result)
                # print('mean_g', mean_g)
                sample_avg_number.append(mean_c)
            except ValueError as e:
                # print('error encounter: \n{} \n ignored!'.format(e))
                continue
        return np.array(sample_avg_number)

    def Mean_Mean_Group_Number(self):
        return self.Sample_Mean_Group_Number().mean()

    def Sample_Mean_Coordination_Number(self):
        sample_avg_number = []
        for result in self.group_metric_results:
            try:
                mean_c = tc.mean_coordination_number(result)
                # print('mean_g', mean_g)
                sample_avg_number.append(mean_c)
            except ValueError as e:
                # print('error encounter: \n{} \n ignored!'.format(e))
                continue
        return np.array(sample_avg_number)

    def Mean_Mean_Coordination_Number(self):
        return self.Sample_Mean_Coordination_Number().mean()

    def Plot_Group_Size(self):
        fig, ax = plt.subplots(1, 1)
        for result in self.group_metric_results:
            try:
                plot_group_size_histogram(result, ax)
            except:
                continue
        plt.show()

    def Plot_Group_Duration(self):
        fig, ax = plt.subplots(1, 1)
        for result in self.group_metric_results:
            try:
                plot_group_durations(result, ax)
            except:
                continue
        plt.show()

    def Plot_one_node_social_trajectory(self, d, node):
        one_graph = self.graph_list[d]
        soc_traj = tc.social_trajectory(one_graph, node=node)
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        plot_social_trajectory(soc_traj, ax, time_unit='s')
        plt.show()


def MMD_Group_Size_Distribution(Gs, FGs):
    return MMD(
        Gs.Sample_Group_Size_Distribution(),
        FGs.Sample_Group_Size_Distribution()
    )


def MMD_Average_Degree_Distribution(Gs, FGs):
    return MMD(
        Gs.Sample_Average_Degree_Distribution(),
        FGs.Sample_Average_Degree_Distribution()
    )


def MMD_Mean_Degree(Gs, FGs):
    return MMD(
        Gs.Sample_Mean_Degree().reshape(-1, 1),
        FGs.Sample_Mean_Degree().reshape(-1, 1)
    )


def MMD_Average_Group_Size(Gs, FGs):
    return MMD(
        Gs.Sample_Average_Group_Size().reshape(-1, 1),
        FGs.Sample_Average_Group_Size().reshape(-1, 1)
    )


def MMD_Mean_Coordination_Number(Gs, FGs):
    return MMD(
        Gs.Sample_Mean_Coordination_Number().reshape(-1, 1),
        FGs.Sample_Mean_Coordination_Number().reshape(-1, 1)
    )


def MMD_Mean_Group_Number(Gs, FGs):
    return MMD(
        Gs.Sample_Mean_Group_Number().reshape(-1, 1),
        FGs.Sample_Mean_Group_Number().reshape(-1, 1)
    )


def MMD_Mean_Group_Duration(Gs, FGs):
    return MMD(
        Gs.Sample_Group_Duration(bins=10),
        FGs.Sample_Group_Duration(bins=10)
    )


# def MMD_Mean_Coordination_Number(Gs, FGs):
#     return MMD(
#         Gs.Sample_Mean_Coordination_Number().reshape(-1, 1),
#         FGs.Sample_Mean_Coordination_Number().reshape(-1, 1)
#     )

def convert_to_weighted_graph(data, N, time_interval, thres):
    data[:, 3] = data[:, 3] / time_interval
    # create one weighted graph
    max_d = np.unique(data[:, 0]).shape[0]
    data = data.astype(int)
    edge_list, edge_counts = np.unique(data[:, 1:4], return_counts=True, axis=0)
    edge_list = np.c_[edge_list, (edge_counts / max_d).reshape(-1, 1)]
    # print('edge_list', edge_list)

    edge_list = [list(e) for e in edge_list]
    tg = teneto.TemporalNetwork(from_edgelist=edge_list, N=N, nettype='wd')
    # print('check_input(tg)', teneto.utils.utils.check_input(tg))

    # contact_list = edge_list
    # contact_list = contact_list[contact_list[:, 3] >= thres]
    # contact_list = contact_list[:, 0:3].astype(int)
    # contact_list = [list(e) for e in contact_list]
    #
    # contact = {
    #     'contacts': contact_list,
    #     'nettype': 'bd',
    #     'netshape': (N, N, int(1. / time_interval)),
    #     't0': 0,
    #     'nodelabels': [str(i + 1) for i in range(N)],
    #     'timeunit': 'unit',
    #     'timetype': 'discrete',
    #     'dimord': 'node,node,time',
    #     'diagonal': 0
    # }
    # contact_tg = teneto.TemporalNetwork(from_dict=contact)
    # print('check_input(contact_tg)', teneto.utils.utils.check_input(contact_tg))
    # # print(np.array(list(map(list, contact['contacts']))))
    #
    # graplet_tg = teneto.utils.contact2graphlet(contact)
    # # print('graplet_tg.network', self.graplet_tg)
    #
    # # teneto.utils.binarize(contact, threshold_type='percent', threshold_level=0.1)
    # # binary_tg = teneto.utils.binarize(contact, threshold_type='percent', threshold_level=0.1)
    # binary_tg = teneto.TemporalNetwork(from_edgelist=contact_list, N=N, nettype='bd')
    return tg

def Plot_Discrete_Graph(data_name, file_name, ax=None, nodesize=0.1, show=False):
    if 'protein' in data_name:
        methods = ['netgan', 'dsbm', 'wenbin']
        if ax is None: fig, ax = plt.subplots(1, len(methods)+1, figsize=(20, 5))
    else:
        methods = ['tggan', 'graphrnn', 'netgan', 'graphvae', 'dsbm', 'wenbin']
        if ax is None: fig, ax = plt.subplots(1, len(methods)+1, figsize=(20, 5))

    # plot
    edgeweightscalar = 3
    plotedgeweights = True
    cmap = "Dark2"

    for i in range(len(methods)):
        method_name = methods[i]
        dataset = "{}_{}".format(method_name, data_name)
        print('dataset', dataset)
        N, n_times, tmax, time_interval, thres, edge_contact_time, fake_graphs = get_fake_graph_data(dataset)
        print('fake_graphs', fake_graphs)
        fake_tg = convert_to_weighted_graph(fake_graphs, N, time_interval, thres)
        fake_nodes_list = np.unique(fake_tg.network.values[:, 1:3].reshape(1, -1)[0])
        max_node = N
        min_node = -1
        min_time = -1
        max_time = 5

        # fake graph
        ax_fake = ax[i+1]
        ax_fake = fake_tg.plot('slice_plot', ax=ax_fake, nodesize=nodesize,
                               plotedgeweights=plotedgeweights, edgeweightscalar=edgeweightscalar,
                               cmap=cmap)

        yticklabels = [str(i) if i in fake_nodes_list else '' for i in range(min_node, max_node + 1)]
        # ax_fake.grid(axis='y')
        ax_fake.set_ylim([min_node, max_node])
        if 'protein' in data_name: ax_fake.set_xlim([min_time, max_time])
        ax_fake.set_yticklabels(yticklabels)
        ax_fake.set_title('{}) {} - {}'.format(i+2, method_name, data_name), fontsize=14)

    # plot real
    ax_real = ax[0]
    real_graphs = get_real_graph_data(data_name, time_interval, edge_contact_time)
    real_tg = convert_to_weighted_graph(real_graphs, N, time_interval, thres)
    real_nodes_list = np.unique(real_tg.network.values[:, 1:3].reshape(1, -1)[0])
    ax_real = real_tg.plot('slice_plot', ax=ax_real, nodesize=nodesize,
                           plotedgeweights=True, edgeweightscalar=5, cmap="Dark2")
    yticklabels = [str(i) if i in real_nodes_list else '' for i in range(min_node, max_node + 1)]
    # ax_real.grid(axis='y')
    ax_real.set_ylim([min_node, max_node])
    if 'protein' in data_name: ax_real.set_xlim([min_time, max_time])
    ax_real.set_yticklabels(yticklabels)
    ax_real.set_title('1) real graphs of {}'.format(data_name))

    for ax_i in ax:
        ax_i.set_ylim([min_node, max_node])
        ax_i.grid(False)

    if show: plt.show()
    plt.tight_layout()
    plt.savefig(file_name, dpi=120)
    plt.close()
    return ax

def Create_Discrete_Temporal_Graph(edges, time_interval, N):
    edges[:, 2] = edges[:, 2] / time_interval
    edges = edges.astype(int)
    edges = [list(e) for e in edges]
    tn = teneto.TemporalNetwork(from_edgelist=edges, N=N, nettype='bd')
    return tn


def convert_discrete_to_continuous(fake_file, n_samples, time_interval, edge_contact_time):
    res = np.loadtxt(fake_file)
    # print('read fake graph file', res.shape)
    unique_d_list = []
    n_times = len(np.unique(res[:, 3]))
    for t in range(n_times):
        edges = res[res[:, 3] == t]
        unique_d_list.append(np.unique(edges[:, 0]))

    fake_graphs = None
    for d in range(n_samples):
        for t in range(n_times):
            if len(unique_d_list[t]) > 0:
                k = np.random.choice(unique_d_list[t])
                one_graph = res[(res[:, 0] == k) & (res[:, 3] == t)]
                one_graph[:, 0] = d
                if fake_graphs is None:
                    fake_graphs = one_graph
                else:
                    fake_graphs = np.r_[fake_graphs, one_graph]

    fake_graphs[:, 3] = (fake_graphs[:, 3] + 1) * time_interval \
                        - time_interval / 2 + edge_contact_time / 2
    return fake_graphs


def get_real_graph_data(data_name, time_interval, edge_contact_time, n_samples=1000):
    print('real data', data_name)
    if 'protein' in data_name:
        real_graphs = np.loadtxt('./data/protein_100_real.txt')
        d_col = 0
        t_col = 3
        real_graphs = real_graphs[real_graphs[:, d_col] < n_samples]
        real_graphs = real_graphs.astype(float)
        real_graphs[:, t_col] = (real_graphs[:, t_col] + 0.5) * time_interval + edge_contact_time / 2
    if 'auth' in data_name:
        user_id = 0
        name = '{}_user_{}'.format(data_name, user_id)
        file = "data/{}_user_{}.txt".format(data_name, user_id)
        real_graphs = np.loadtxt('./data/{}_user_{}.txt'.format('auth', user_id))
        # real_graphs[:, 1] = real_graphs[:, 1] - 1
        # real_graphs[:, 1] = real_graphs[:, 1] - 1
    elif 'metro' in data_name:
        user_id = 4
        name = '{}_user_{}'.format(data_name, user_id)
        file = "data/{}_user_{}.txt".format(data_name, user_id)
        real_graphs = np.loadtxt('./data/{}_user_{}.txt'.format('metro', user_id))
    elif 'scale-free-nodes-100' in data_name:
        name = 'data-scale-free-nodes-100-samples-200'
        real_graphs = np.loadtxt('./{}/{}.txt'.format(name, name))
    elif 'scale-free-nodes-500' in data_name:
        name = 'data-scale-free-nodes-500-samples-100'
        real_graphs = np.loadtxt('./{}/{}.txt'.format(name, name))
    elif 'scale-free-nodes-2500' in data_name:
        name = 'data-scale-free-nodes-2500-samples-100'
        real_graphs = np.loadtxt('./{}/{}.txt'.format(name, name))
    return real_graphs


def get_fake_graph_data(dataset, n_samples=100):
    if 'tggan' in dataset:
        if 'auth' in dataset:
            N = 27
            l = 103
            n_times = 4
            tmax = 1.0
            edge_contact_time = 1e-4
            time_interval = 1. / n_times
            thres = 0.001
            _it = '20191231-204105_assembled_graph_iter_30000'
            matlab_file = 'fake_graph_TGGAN_auth_iter_30000'
            dir = 'outputs-auth-user-0-best'
            fake_file = './{}/{}.npz'.format(dir, _it)
        elif 'metro' in dataset:
            N = 91
            l = 4
            n_times = 6
            user_id = 4
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = 1. / n_times
            thres = 0.01

            _it = '20191103-235308_assembled_graph_iter_71000'
            matlab_file = 'fake_graph_TGGAN_metro_iter_71000'
            dir = 'outputs-metro-user-4-best'
            fake_file = './{}/{}.npz'.format(dir, _it)
        elif 'scale-free-nodes-100' in dataset:
            N = 100
            l = 4
            n_times = 5
            tmax = 1.0
            edge_contact_time = 0.1
            time_interval = 1. / n_times
            thres = 0.01

            _it = '20200210-110001_assembled_graph_iter_39000'
            matlab_file = 'fake_graph_TGGAN_scale-free-nodes-100_iter_39000'
            dir = 'outputs-scale-free-nodes-100-best'
            fake_file = './{}/{}.npz'.format(dir, _it)
        elif 'scale-free-nodes-500' in dataset:
            N = 500
            l = 4
            n_times = 5
            tmax = 1.0
            edge_contact_time = 0.1
            time_interval = 1. / n_times
            thres = 0.01

            _it = '20200207-082904_assembled_graph_iter_23000'
            matlab_file = 'fake_graph_TGGAN_scale-free-nodes-500_iter_23000'
            dir = 'outputs-scale-free-nodes-500-best'
            fake_file = './{}/{}.npz'.format(dir, _it)
        elif 'scale-free-nodes-2500' in dataset:
            N = 2500
            l = 4
            n_times = 5
            tmax = 1.0
            edge_contact_time = 0.05
            time_interval = 1. / n_times
            thres = 0.01

            _it = '20200207-140714_assembled_graph_iter_13000'
            matlab_file = 'fake_graph_TGGAN_scale-free-nodes-2500_iter_13000'
            dir = 'outputs-scale-free-nodes-2500-samples-100-best'
            fake_file = './{}/{}.npz'.format(dir, _it)

        res = np.load(fake_file)
        # print(list(res.keys()))
        if dataset == 'tggan_auth':
            fake_graphs = scipy.io.loadmat('./{}/{}.mat'.format(dir, matlab_file))
            fake_graphs = fake_graphs['fake_graphs'][:10]
            fake_graphs = convert_graphs(fake_graphs)
            fake_graphs[:, 3] = 1. - fake_graphs[:, 3]
        else:
            fake_graphs = res['fake_graphs']
            scipy.io.savemat('./{}/{}.mat'.format(dir, matlab_file),
                             dict(fake_graphs=fake_graphs))

    if 'wenbin' in dataset:
        if 'auth' in dataset:
            N = 27
            l = 4
            n_times = 4
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 2000
            fake_file = 'baselines/outputs/fake_graph_wenbin_auth_user_generated.txt'
        if 'metro' in dataset:
            N = 91
            l = 4
            n_times = 6
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = 1. / n_times
            thres = 0.01
            n_samples = 1
            fake_file = 'baselines/outputs/fake_graph_wenbin_metro_user_generated.txt'
        if 'protein' in dataset:
            N = 8
            l = 4
            n_times = 100
            tmax = 1.0
            edge_contact_time = 0.002
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 20
            fake_file = 'baselines/outputs/fake_graph_wenbin_protein100_user_generated.txt'

        fake_graphs = np.loadtxt(fake_file)
        d_col = 0
        t_col = 3
        fake_graphs = fake_graphs[fake_graphs[:, d_col] < n_samples]
        fake_graphs = fake_graphs.astype(float)
        fake_graphs[:, t_col] = (fake_graphs[:, t_col] + 0.5) * time_interval + edge_contact_time / 2

    if 'graphrnn' in dataset or 'graphvae' in dataset or 'netgan' in dataset:
        if 'protein' in dataset:
            N = 8
            l = 4
            n_times = 100
            tmax = 1.0
            edge_contact_time = 0.002
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 2000
            if 'graphrnn' in dataset: fake_file = 'baselines/outputs/GraphRNN_RNN_'
            if 'graphvae' in dataset: fake_file = 'baselines/outputs/GraphRNN_VAE_conditional_'
            if 'netgan' in dataset: fake_file = 'baselines/outputs/fake_graph_netgan_protein_node_8_sample_1000_times_100.txt'
        if 'auth' in dataset:
            N = 27
            l = 4
            n_times = 4
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 2000
            if 'graphrnn' in dataset: fake_file = 'baselines/outputs/GraphRNN_RNN_auth_epoch_3000.txt'
            if 'graphvae' in dataset: fake_file = 'baselines/outputs/GraphRNN_VAE_conditional_auth_epoch_3000.txt'
            if 'netgan' in dataset: fake_file = 'baselines/outputs/fake_graph_netgan_auth_node_27_sample_1000_times_4.txt'
            # if 'wenbin' in dataset: fake_file = 'baselines/outputs/fake_graph_wenbin_auth_user_generated.txt'
        elif 'metro' in dataset:
            N = 91
            l = 4
            n_times = 6
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 2000
            if 'graphrnn' in dataset: fake_file = 'baselines/outputs/GraphRNN_RNN_metro_epoch_3000.txt'
            if 'graphvae' in dataset: fake_file = 'baselines/outputs/GraphRNN_VAE_conditional_metro_epoch_3000.txt'
            if 'netgan' in dataset: fake_file = 'baselines/outputs/fake_graph_netgan_metro_node_91_sample_1000_times_6.txt'
            # if 'wenbin' in dataset: fake_file = 'baselines/outputs/fake_graph_wenbin_metro_user_generated.txt'
        elif 'scale-free-nodes-100' in dataset:
            N = 100
            l = 4
            n_times = 5
            tmax = 1.0
            edge_contact_time = 0.01
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 2000
            if 'graphrnn' in dataset: fake_file = './baselines/outputs/GraphRNN_RNN_simulation_nodes_100_samples_200_epoch_3000.txt'
            if 'graphvae' in dataset: fake_file = './baselines/outputs/GraphRNN_VAE_simulation_nodes_100_samples_200_epoch_3000.txt'
            if 'netgan' in dataset: fake_file = 'baselines/outputs/fake_graph_netgan_simulation_node_100_sample_200_times_5.txt'
        elif 'scale-free-nodes-500' in dataset:
            N = 500
            l = 4
            n_times = 5
            tmax = 1.0
            edge_contact_time = 0.01
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 2000
            if 'graphrnn' in dataset: fake_file = './baselines/outputs/GraphRNN_RNN_simulation_nodes_500_samples_100_epoch_3000.txt'
            if 'graphvae' in dataset: fake_file = './baselines/outputs/GraphRNN_VAE_simulation_nodes_500_samples_100_epoch_3000.txt'
            if 'netgan' in dataset: fake_file = 'baselines/outputs/fake_graph_netgan_simulation_node_500_sample_100_times_5.txt'
        elif 'scale-free-nodes-2500' in dataset:
            N = 2500
            l = 4
            n_times = 5
            tmax = 1.0
            edge_contact_time = 0.01
            time_interval = 1. / n_times
            thres = 0.01
            # n_samples = 2000
            if 'graphrnn' in dataset: fake_file = './baselines/outputs/GraphRNN_RNN_simulation_nodes_2500_samples_100_epoch_3000.txt'
            if 'graphvae' in dataset: fake_file = './baselines/outputs/GraphRNN_VAE_simulation_nodes_2500_samples_100_epoch_3000.txt'
            if 'netgan' in dataset: fake_file = 'baselines/outputs/fake_graph_netgan_simulation_node_2500_sample_100_times_5.txt'

        fake_graphs = convert_discrete_to_continuous(fake_file, n_samples, time_interval, edge_contact_time)

    if 'dsbm' in dataset:
        if 'protein' in dataset:
            N = 8
            l = 4
            n_times = 100
            tmax = 1.0
            edge_contact_time = 0.002
            time_interval = 1. / n_times
            thres = 0.01
            n_samples = 10
            fake_file = './baselines/outputs/DSBM_protein100_May-27-2020_22-46-03.mat'
        if 'auth' in dataset:
            N = 27
            l = 4
            n_times = 4
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = tmax / n_times
            thres = 0.01
            # n_samples = 10
            fake_file = 'baselines/outputs/DSBM_auth_Jan-10-2020_17-33-49.mat'
        elif 'metro' in dataset:
            N = 91
            l = 4
            n_times = 6
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = tmax / n_times
            thres = 0.01
            n_samples = 10
            fake_file = 'baselines/outputs/DSBM_metro_Jan-13-2020_15-31-35.mat'
        elif 'scale-free-nodes-100' in dataset:
            N = 100
            l = 25
            n_times = 5
            tmax = 1.0
            edge_contact_time = 0.02
            time_interval = tmax / n_times
            thres = 0.01
            # n_samples = 10
            fake_file = 'baselines/outputs/DSBM_scale-free-nodes-100_Feb-05-2020_12-15-32.mat'

        t_col = 3
        res = scipy.io.loadmat(fake_file)
        res = res['adjDsbm_fake_graphs']
        # res = res['adjSbtm_fake_graphs']
        res = res.reshape(N, N, n_times, -1)
        res = res[:, :, :, :n_samples]
        # print(dataset, res.shape)
        fake_graphs = []
        for d in range(res.shape[-1]):
            i_inx, j_inx, t_inx = np.nonzero(res[:, :, :, d])
            for i, j, t in zip(i_inx, j_inx, t_inx):
                fake_graphs.append([d, i, j, t])
        fake_graphs = np.array(fake_graphs).astype(float)
        fake_graphs[:, t_col] = (fake_graphs[:, t_col] + 0.5) * time_interval + edge_contact_time / 2

    return N, n_times, tmax, time_interval, thres, edge_contact_time, fake_graphs


if __name__ == "__main__":

    # data_name = 'auth'
    # data_name = 'metro'
    # data_name = 'scale-free-nodes-100'
    # data_name = 'protein'

    save_directory = './evaluation_outputs'
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # Plot_Discrete_Graph(data_name=data_name,
    #                     file_name='{}/discrete_{}'.format(save_directory, data_name))

    # ### mmd purposes
    # # dataset = 'tggan_auth'
    # # dataset = 'tggan_metro'
    # # dataset = 'graphrnn_auth'
    # # dataset = 'graphrnn_metro'
    # # dataset = 'graphvae_auth'
    # # dataset = 'graphvae_metro'
    # # dataset = 'dsbm_auth'
    # # dataset = 'dsbm_metro'
    # # dataset = 'tggan_scale-free-nodes-100'
    # # dataset = 'graphrnn_scale-free-nodes-100'
    # dataset = 'graphrnn_scale-free-nodes-2500'
    # # dataset = 'graphvae_scale-free-nodes-100'
    # # dataset = 'dsbm_scale-free-nodes-100'
    #
    # # dataset = 'netgan_auth'
    # # dataset = 'netgan_metro'
    # # dataset = 'netgan_scale-free-nodes-100'
    # # dataset = 'netgan_scale-free-nodes-500'
    # # dataset = 'netgan_scale-free-nodes-2500'
    #
    # # dataset = 'tggan_scale-free-nodes-500'
    # # dataset = 'tggan_scale-free-nodes-2500'
    #
    # train_ratio = 0.8
    #
    # N, n_times, tmax, time_interval, thres, edge_contact_time, fake_graphs = get_fake_graph_data(dataset)
    # real_graphs = get_real_graph_data(dataset, time_interval, edge_contact_time)
    #
    #
    # Gs = Graphs(real_graphs, N, tmax, edge_contact_time)
    # # Plot_Graph(Gs.graph_list[n_times], '{}/{}_real'.format(save_directory, dataset.split('_')[1]))
    #
    # print('Real Mean_Average_Degree_Distribution:\n', Gs.Mean_Average_Degree_Distribution())
    # print('Real Mean_Degree:\n', Gs.Mean_Mean_Degree())
    # print('Real Mean_Average_Group_Size_Distribution:\n', Gs.Mean_Average_Group_Size_Distribution())
    # print('Real Mean_Average_Group_Number:\n', Gs.Mean_Mean_Group_Number())
    # print('Real Mean_Mean_Coordination_Number:\n', Gs.Mean_Mean_Coordination_Number())
    #
    # print('fake_graphs', fake_graphs)
    # FGs = Graphs(fake_graphs, N, tmax, edge_contact_time)
    # # Plot_Graph(FGs.graph_list[n_times], '{}/{}_fake'.format(save_directory, dataset))
    #
    # print('Fake Mean_Average_Degree_Distribution:\n', FGs.Mean_Average_Degree_Distribution())
    # print('Fake Mean_Degree:\n', FGs.Mean_Mean_Degree())
    # print('Fake Mean_Average_Group_Size_Distribution:\n', FGs.Mean_Average_Group_Size_Distribution())
    # print('Fake Mean_Average_Group_Number:\n', FGs.Mean_Mean_Group_Number())
    # print('Fake Mean_Mean_Coordination_Number:\n', FGs.Mean_Mean_Coordination_Number())
    #
    # print('MMD_Average_Degree', MMD_Average_Degree_Distribution(Gs, FGs))
    # print('MMD_Mean_Degree', MMD_Mean_Degree(Gs, FGs))
    # print('MMD_Group_Size_Distribution', MMD_Group_Size_Distribution(Gs, FGs))
    # print('MMD_Average_Group_Size', MMD_Average_Group_Size(Gs, FGs))
    # print('MMD_Mean_Coordination_Number', MMD_Mean_Coordination_Number(Gs, FGs))
    # print('MMD_Mean_Group_Number', MMD_Mean_Group_Number(Gs, FGs))
    # print('MMD_Mean_Group_Duration', MMD_Mean_Group_Duration(Gs, FGs))

    # DGs = Discrete_Graphs(real_graphs, N, time_interval, thres, is_real_graph=True)
    # print('DGs.graplet_tg', DGs.graplet_tg.shape)
    #
    # FDGs = Discrete_Graphs(fake_graphs, N, time_interval, thres, is_real_graph=False)
    # print('FDGs.graplet_tg', FDGs.graplet_tg.shape)

    # print('3d network matrix of graplet representation', Gs.graplet_tg.shape)
    #
    # try:
    #     print('Temporal_Degree_Centrality\n', Gs.Temporal_Degree_Centrality())
    # except FunctionTimedOut:
    #     print("Gs.Temporal_Degree_Centrality() could not complete within 60 seconds and was terminated.")
    # except Exception as e:
    #     print('other error happened:\n{}'.format(e))
    # print('\n')

    # start = time.clock()
    # try:
    #     Gs.Get_Temporal_Shortest_Path()
    #     Gs.paths.to_csv('{}/{}_discrete_user_{}_shortest_path.csv'.format(save_directory, dataset, user_id))
    #     print('paths', Gs.paths.shape)
    # except FunctionTimedOut:
    #     print("Gs.Get_Temporal_Shortest_Path() could not complete within 60 seconds and was terminated.")
    # except Exception as e:
    #     print('other error happened:\n{}'.format(e))
    # end = time.clock()
    # print('Get_Temporal_Shortest_Path() used time', end-start)
    # print('\n')

    # try:
    #     print('Temporal_Betweenness_Centrality', Gs.Temporal_Betweenness_Centrality())
    # except FunctionTimedOut:
    #     print("Gs.Temporal_Betweenness_Centrality() could not complete within 60 seconds and was terminated.")
    # except Exception as e:
    #     print('other error happened:\n{}'.format(e))
    # print('\n')
    #
    # try:
    #     print('Temporal_Closeness_Centrality', Gs.Temporal_Closeness_Centrality())
    # except FunctionTimedOut:
    #     print("Gs.Temporal_Closeness_Centrality() could not complete within 60 seconds and was terminated.")
    # except Exception as e:
    #     print('other error happened:\n{}'.format(e))
    # print('\n')

    # try:
    #     print('Topological_Overlap', Gs.Topological_Overlap())
    # except FunctionTimedOut:
    #     print("Gs.Topological_Overlap() could not complete within 60 seconds and was terminated.")
    # except Exception as e:
    #     print('other error happened:\n{}'.format(e))
    # print('\n')
    #
    # try:
    #     print('Bursty_Coeff', Gs.Bursty_Coeff())
    # except FunctionTimedOut:
    #     print("Gs.Bursty_Coeff() could not complete within 60 seconds and was terminated.")
    # except Exception as e:
    #     print('other error happened:\n{}'.format(e))
    # print('\n')

    # try:
    #     print('Temporal_Efficiency', Gs.Temporal_Efficiency())
    # except FunctionTimedOut:
    #     print("Gs.Temporal_Efficiency() could not complete within 60 seconds and was terminated.")
    # except Exception as e:
    #     print('other error happened:\n{}'.format(e))
    # print('\n')

    ### running time
    file_name = 'running_time_nodes'
    timing_100 = np.loadtxt('./timing_results/simulation-scale-free-nodes-100_iterations_100000.txt')
    timing_500 = np.loadtxt('./timing_results/simulation-scale-free-nodes-500_iterations_1000.txt')
    timing_2500 = np.loadtxt('./timing_results/simulation-scale-free-nodes-2500_iterations_100000.txt')
    m = 200
    n = m + 100
    node_list = [100, 500, 2500]
    tggan_time= np.log10([timing_100[m:n].mean(), timing_500[m:n].mean(), timing_2500[m:n].mean()])

    graphrnn_100 = np.load(
        './timing_results/GraphRNN_RNN_simulation_nodes_100_samples_200_times_5_4_128_pred__nodes_100_days_200_time_0_GraphRNN_RNN.npy')
    graphrnn_500 = np.load(
        './timing_results/GraphRNN_VAE_conditional_simulation_nodes_500_samples_100_times_5_4_128_pred__nodes_500_days_100_time_0_GraphRNN_VAE_conditional.npy')
    graphrnn_2500 = np.load(
        './timing_results/GraphRNN_RNN_simulation_nodes_2500_samples_100_times_5_4_128_pred__nodes_2500_days_100_time_0_GraphRNN_RNN.npy')
    graphrnn_time = np.log10([graphrnn_100[m:n].mean(), graphrnn_500[m:n].mean(), graphrnn_2500[m:n].mean()])

    graphvae_100 = np.load(
        './timing_results/GraphRNN_VAE_conditional_simulation_nodes_100_samples_200_times_5_4_128_pred__nodes_100_days_200_time_0_GraphRNN_VAE_conditional.npy'
    )
    graphvae_500 = np.load(
        './timing_results/GraphRNN_VAE_conditional_simulation_nodes_500_samples_100_times_5_4_128_pred__nodes_500_days_100_time_0_GraphRNN_VAE_conditional.npy'
    )
    graphvae_time = np.log10([graphvae_100[m:n].mean(), graphvae_500[m:n].mean()*5])

    dsbm_100 = 1.9352e+03/20
    dsbm_time = [np.log10(dsbm_100)]

    netgan_time = np.array([272, 486, 655])
    netgan_time = np.log10(netgan_time/200)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(node_list, tggan_time, c='black', marker='s', label='TG-GAN')
    ax.plot(node_list, graphrnn_time, c='b', marker='s', label='GraphRNN')
    ax.plot(node_list, netgan_time, c='purple', marker='s', label='NetGAN')
    ax.plot(node_list[0:2], graphvae_time, c='r', marker='s', label='GraphVAE')
    ax.plot(node_list[0:1], dsbm_time, c='orange', marker='s', label='DSBM')
    ax.set_xlabel('Number of nodes', fontsize=18)
    ax.set_xticks(node_list)
    ax.set_ylabel('Running Time \n[per epoch]', fontsize=18)
    y_ticks = [-1, 0, 1, 2]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([r'$10^{%d}$' % i for i in y_ticks])
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig('{}/{}.png'.format(save_directory, file_name), dpi=120)

    ### running time with time snapshots
    file_name = 'running_time_snapshots'

    snapshot_list = [5, 10, 20]
    print([netgan_time[0]*s/5 for s in snapshot_list])
    print([graphrnn_time[0]*s/5 for s in snapshot_list])

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(snapshot_list, [tggan_time[0] for s in snapshot_list],
            c='black', marker='s', label='TG-GAN')
    ax.plot(snapshot_list, [graphrnn_time[0]*s/5 for s in snapshot_list],
            c='b', marker='s', label='GraphRNN')
    ax.plot(snapshot_list, [netgan_time[0]*s/5 for s in snapshot_list],
            c='purple', marker='s', label='NetGAN')
    ax.plot(snapshot_list[0:2], [graphvae_time[0]*s/5 for s in snapshot_list[0:2]],
            c='r', marker='s', label='GraphVAE')
    ax.plot(snapshot_list[0:1], [dsbm_time[0]*s/5 for s in snapshot_list[0:1]],
            c='orange', marker='s', label='DSBM')
    ax.set_xlabel('Number of snapshots', fontsize=18)
    ax.set_xticks(snapshot_list)
    ax.set_ylabel('Running Time \n[per epoch]', fontsize=18)
    y_ticks = [-1, 0, 1, 2]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([r'$10^{%d}$' % i for i in y_ticks])
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig('{}/{}.png'.format(save_directory, file_name), dpi=120)

    ### loss function plot
    # file_name = 'tggan_loss'
    # # loss = np.load('outputs-auth-user-0/20200209-004116_training_loss_iter_45000.npz')
    # loss = np.load('outputs-scale-free-nodes-100-samples-200/20200210-110001_training_loss_iter_55000.npz')
    #
    # gen_losses = loss['gen_losses']
    # disc_losses = loss['disc_losses']
    # epoches = np.arange(len(gen_losses))
    # # mask = disc_losses < 20
    # mask = (disc_losses < 20) & (epoches < 2800)
    # gen_losses = gen_losses[mask]
    # disc_losses = disc_losses[mask]
    # epoches = epoches[mask]
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax.plot(epoches, disc_losses, label='Discriminator loss')
    # ax.plot(epoches, gen_losses, label='Generator loss')
    # ax.set_xlabel('Epoches', fontsize=20)
    # ax.set_ylabel('Loss', fontsize=20)
    # ax.tick_params(labelsize=18)
    # ax.legend(fontsize=18)
    # plt.subplots_adjust(left=0.16, bottom=0.16)
    # plt.savefig('{}/{}.png'.format(save_directory, file_name), dpi=120)

    print('finish execution!')
