import matplotlib
from networkx.utils import py_random_state

matplotlib.use('Agg')
from teneto import TemporalNetwork
import tacoma as tc
import teneto
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import random
from evaluation import *


def discrete_time_simulation(n_nodes, n_times, prob, simProcess, ncontacts, lam, nettype):
    tnet = teneto.TemporalNetwork(timetype='discrete')
    if simProcess == 'rand_binomial':
        tnet.generatenetwork(simProcess, size=(n_nodes, n_nodes, n_times), nettype=nettype, prob=prob, netrep='graphlet')
    elif simProcess == 'rand_poisson':
        tnet.generatenetwork(simProcess, nnodes=n_nodes, ncontacts=ncontacts, nettype=nettype, lam=lam, netrep='graphlet')
    else:
        raise Exception('simulation method do not supported!')
    return tnet

def multi_discrete_time_simulate(n_days, n_nodes, n_times, prob, simProcess, ncontacts, lam, nettype):
    res_sim = []
    for d in range(n_days):
        tnet = discrete_time_simulation(n_nodes, n_times, prob, simProcess, ncontacts, lam, nettype)
        arr = tnet.network.values
        res_sim.extend(list(
            np.c_[[[d]]*arr.shape[0], arr]
        ))
    return np.array(res_sim).reshape(-1, 4)

def scale_free_graph(n_nodes, t0, t_max, shape, scale, alpha=0.51, beta=0.44, gamma=0.05, delta_in=0.4,
                     delta_out=0.1, create_using=None, seed=None, edge_contact_time=0.02):
    """Returns a scale-free directed graph.

    Parameters
    ----------
    n : integer
        Number of nodes in graph
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution.
    beta : float
        Probability for adding an edge between two existing nodes.
        One existing node is chosen randomly according the in-degree
        distribution and the other chosen randomly according to the out-degree
        distribution.
    gamma : float
        Probability for adding a new node conecgted to an existing node
        chosen randomly according to the out-degree distribution.
    delta_in : float
        Bias for choosing ndoes from in-degree distribution.
    delta_out : float
        Bias for choosing ndoes from out-degree distribution.
    create_using : graph, optional (default MultiDiGraph)
        Use this graph instance to start the process (default=3-cycle).
    seed : integer, optional
        Seed for random number generator

    Examples
    --------
    Create a scale-free graph on one hundred nodes::

    Notes
    -----
    The sum of ``alpha``, ``beta``, and ``gamma`` must be 1.

    References
    ----------
    .. [1] B. Bollobás, C. Borgs, J. Chayes, and O. Riordan,
           Directed scale-free graphs,
           Proceedings of the fourteenth annual ACM-SIAM Symposium on
           Discrete Algorithms, 132--139, 2003.
    """

    def _choose_node(G, distribution, delta, psum):
        cumsum = 0.0
        # normalization
        r = random.random()
        for n, d in distribution:
            cumsum += (d + delta) / psum
            if r < cumsum:
                break
        return n

    edge_lists = []
    t = []
    t_total = t0
    if create_using is None:
        # start with 3-cycle
        G = nx.MultiDiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        edge_lists.append(list(G.edges()))
        # t.append(t_total)

        t.append(t_total - edge_contact_time)
        edge_lists.append([])
        t.append(t_total)
    else:
        # keep existing graph structure?
        G = create_using
        if not (G.is_directed() and G.is_multigraph()):
            raise nx.NetworkXError( \
                "MultiDiGraph required in create_using")

    if alpha <= 0:
        raise ValueError('alpha must be >= 0.')
    if beta <= 0:
        raise ValueError('beta must be >= 0.')
    if gamma <= 0:
        raise ValueError('beta must be >= 0.')

    if alpha + beta + gamma != 1.0:
        raise ValueError('alpha+beta+gamma must equal 1.')

    G.name = "directed_scale_free_graph(alpha=%s,beta=%s,gamma=%s,delta_in=%s,delta_out=%s)" % (
    alpha, beta, gamma, delta_in, delta_out)

    # seed random number generated (uses None as default)
    random.seed(seed)

    number_of_edges = G.number_of_edges()
    while t_total < t_max and len(G) < n_nodes:
        psum_in = number_of_edges + delta_in * len(G)
        psum_out = number_of_edges + delta_out * len(G)
        # r = np.random.gamma(shape=shape, scale=scale)
        r = random.random()
        t_total += r

        if t_total >= t_max:
            break
        # random choice in alpha,beta,gamma ranges
        if r < alpha:
            # alpha
            # add new node v
            v = len(G)
            # v = np.random.choice(range(len(G), n_nodes-1))

            # choose w according to in-degree and delta_in
            w = _choose_node(G, G.in_degree(), delta_in, psum_in)
        elif r < alpha + beta:
            # beta
            # choose v according to out-degree and delta_out
            v = _choose_node(G, G.out_degree(), delta_out, psum_out)
            # choose w according to in-degree and delta_in
            w = _choose_node(G, G.in_degree(), delta_in, psum_in)
        else:
            # gamma
            # choose v according to out-degree and delta_out
            v = _choose_node(G, G.out_degree(), delta_out, psum_out)
            # add new node w
            w = len(G)
            # w = np.random.choice(range(len(G), n_nodes-1))
        number_of_edges += 1

        G.add_edge(v, w)
        edge_lists.append([(v, w)])
        # t.append(t_max)
        # t.append(t_total)

        t.append(t_total - edge_contact_time)
        edge_lists.append([])
        t.append(t_total)
    # print(t_max, t)
    t = [i / t_max for i in t]
    return edge_lists, t

def continuous_time_simulation(n_nodes, t0, t_max, shape, scale, edge_contact_time,
                               alpha, beta, gamma, delta_in, delta_out, mean_degree=1.5):
    # # static structure parameters
    # p = mean_degree / (n_nodes - 1.0)

    # temporal parameters
    edge_lists = []
    t = []
    this_time = t0

    edge_lists, t = scale_free_graph(n_nodes, t0=t0, t_max=t_max, shape=shape, scale=scale,
                                     edge_contact_time=edge_contact_time,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     delta_in=delta_in, delta_out=delta_out)
    # while this_time < tmax:
    #     G = nx.fast_gnp_random_graph(n_nodes, p)  # Generate a new random network
    #     # G = nx.gnm_random_graph(N, M)  # Generate a Erdos Renyi network
    #     # G = nx.barabasi_albert_graph(N, M) # Generate a Erdos Renyi network
    #     these_edges = list(G.edges())
    #     t.append(this_time)
    #     edge_lists.append(these_edges)
    #     this_time += np.random.exponential(scale=scale)

    # save to _tacoma-object
    el = tc.edge_lists()
    el.N = n_nodes
    el.t = t
    el.edges = edge_lists
    el.tmax = t_max
    res_sim = []
    for k in range(len(el.t)):
        t = el.t[k]
        # print('el.t[k]', el.t[k])
        # print('el.edges[k]', el.edges[k])
        for i, j in el.edges[k]:
            res_sim.append([i, j, t])
    return res_sim, el

def multi_continuous_time_simulate(n_nodes, n_days, t0=0.1, t_max=10.0, shape=1.5, scale=0.2, edge_contact_time=0.02,
                                   alpha=0.51, beta=0.44, gamma=0.05, delta_in=0.4, delta_out=0.1, mean_degree=1.5):
    res_sim = []
    for d in range(n_days):
        d_res, el = continuous_time_simulation(n_nodes, t0, t_max, shape, scale, edge_contact_time=edge_contact_time,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     delta_in=delta_in, delta_out=delta_out, mean_degree=mean_degree)
        for e in d_res:
            res_sim.append([d] + e)
    return np.array(res_sim)

if __name__ == "__main__":
    from evaluation import *
    from tacoma.drawing import edge_activity_plot

    N = 5
    t0 = 0.2
    edges, el = continuous_time_simulation(n_nodes=N, t0=t0, t_max=N, shape=1., scale=1., edge_contact_time=t0*0.5,
                                    alpha=0.23, beta=0.54, gamma=0.23, delta_in=0.2, delta_out=0.)
    print('one graph:\n', edges)

    tn_graph = Create_Discrete_Temporal_Graph(np.array(edges), 1./4, N)

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    fig.subplots_adjust(bottom=0.3)

    edge_activity_plot(el,
                       ax=ax[0],
                       time_normalization_factor=1.,
                       time_unit='s',
                       alpha=1.0,  # opacity
                       linewidth=3.5,
                       )

    tn_graph.plot('slice_plot', ax=ax[1], cmap='Pastel2')
    # tn_graph.plot('circle_plot', ax=ax[1], cmap='Pastel2')
    # tn_graph.plot('graphlet_stack_plot', ax=ax[1], cmap='Greys')

    # ax[0].annotate('Continuou-time plot', xy=(0.4, -2), fontsize=18)
    ax[0].set_title(r'$\bf{a)}$ Continuou time graph', y=-0.5, fontsize=14)
    ax[0].set_xlim([0., 1.])
    ax[1].set_title(r'$\bf{b)}$ Time snapshots graph', y=-0.5, fontsize=14)
    ax[1].set_xlabel('snapshots [0.25s per snapshot]')
    ax[1].set_ylabel('node id')
    plt.show()
    # plt.savefig('example.png', dpi=120)
    # plt.close()

    N = 10
    t0 = 0.2
    edges = multi_continuous_time_simulate(n_days=100, n_nodes=N, t0=0.1, t_max=N*2, scale=0.7,
                                         edge_contact_time=t0*0.5, alpha=0.13, beta=0.14, gamma=0.73,
                                         delta_in=0.2, delta_out=0.)
    Gs = Graphs(edges, N, tmax=1.0, edge_contact_time=t0*0.8)
    print('Mean_Average_Degree_Distribution', Gs.Mean_Average_Degree_Distribution())

    # from pandas.compat.numpy import *
    # edges=np.array(np.loadtxt('edgeaa.txt'))
    # edges[:,2]=edges[:,2]-edges[0,2]
    # edges[:,0]=edges[:,0]-1
    # edges[:,1]=edges[:,1]-1
    # edges=edges[edges[:,0]!= edges[:,1]]
    # print(edges[1:9])
    # y=len(edges)
    # edges=list(edges)
    # # edges[i]=int(edges[i])R
    # for i in range(len(edges)):
    #     edges[i]=list(edges[i])
    # tnet = TemporalNetwork(N=5,from_edgelist=edges[0:9],nettype='bu')#from_edgelist=edges[1:9],timeunit='s',nettype='bu')
    # # tnet.generatenetwork('rand_binomial',size=(5,2), prob=0.3)
    # ij = tnet.netshape[0]
    # t = tnet.netshape[1]
    # tnet=TemporalNetwork(timetype='continuous')
    # tnet.generatenetwork('rand_binomial',size=(3,10), prob=0.5)
    # print('tnet\n', tnet.network)
    # print('tnet shape:', tnet.network.shape)

    # n_nodes = 27
    # n_times = 10
    # lam = int(n_times / 10)
    # n_days = 1000
    # prob = (0.5, 0.5)
    # ncontacts = 3
    # simProcess = 'rand_binomial'
    # # simProcess = 'rand_poisson'
    # tnet = simulation(n_nodes, n_times, prob, simProcess, ncontacts, lam, nettype='bd')
    # print('check_input(tnet)', teneto.utils.utils.check_input(tnet))
    # print('one sampled graph:\n', tnet.network)
    #
    # all_graphs = multi_simulate(n_days, n_nodes, n_times, prob, simProcess, ncontacts, lam)
    # print('all sampled graphs:\n', all_graphs.shape)

    # edges = np.c_[
    #     np.random.choice(n_nodes, (n_lines,)),
    #     np.random.choice(n_nodes, (n_lines,)),
    #     np.random.choice(n_times, (n_lines,)),
    # ]

    # edges=list(edges)
    # for i in range(len(edges)):
    #     edges[i]=list(edges[i])
    # tnet = TemporalNetwork(N=n_nodes,from_edgelist=edges,nettype='bu')

    # tnet = teneto.TemporalNetwork(timetype='continuous')
    # tnet.generatenetwork(simProcess, size=(n_nodes, n_times), prob=prob,
    #                      # netrep='contact'
    #                      )

    # print('tnet shape:', tnet.network.shape)
    # print('tnet:', tnet.network)

    # tnet.get_network_when(i=1, j=1, logic='or')
    # p20=teneto.utils.get_network_when(tnet, t=t)
    # tnet.plot('slice_plot', cmap='Pastel2')
    # plt.show()

    # print('node:', n_nodes, 'time:', n_times, 'prob', prob)
    # start = time.clock()
    # print('temporal_degree_centrality')
    # teneto.networkmeasures.temporal_degree_centrality(tnet)
    # end = time.clock()
    # print(end-start)

    # start = time.clock()
    # print('temporal_betweenness_centrality')
    # teneto.networkmeasures.temporal_betweenness_centrality(tnet)
    # end = time.clock()
    # print(end-start)

    # start = time.clock()
    # print('temporal_closeness_centrality')
    # teneto.networkmeasures.temporal_closeness_centrality(tnet)
    # end = time.clock()
    # print(end-start)
    #
    # start = time.clock()
    # print('topological_overlap')
    # teneto.networkmeasures.topological_overlap(tnet)
    # end = time.clock()
    # print(end-start)
    #
    # start = time.clock()
    # print('bursty_coeff')
    # teneto.networkmeasures.bursty_coeff(tnet)
    # end = time.clock()
    # print(end-start)

    # start = time.clock()
    # print('temporal_efficiency')
    # teneto.networkmeasures.temporal_efficiency(tnet)
    # end = time.clock()
    # print(end-start)
    #
    # start = time.clock()
    # print('reachability_latency')
    # y=teneto.networkmeasures.reachability_latency(tnet)
    # end = time.clock()
    # print(end-start)
    #
    # start = time.clock()
    # print('fluctuability')
    # u=teneto.networkmeasures.fluctuability(tnet,calc='global')
    # end = time.clock()
    # print(end-start)
    #
    # start = time.clock()
    # print('volatility')
    # p=teneto.networkmeasures.volatility(tnet,calc='global')
    # end = time.clock()
    # print(end-start)
    #
    # start = time.clock()
    # print('topological_overlap')
    # l=teneto.networkmeasures.topological_overlap(tnet,calc='global')
    # end = time.clock()
    # print(end-start)
    #
    # start = time.time()
    # print('shortest_temporal_path')
    # # k=teneto.networkmeasures.sid(tnet,tnet,calc='global')
    # g=teneto.networkmeasures.shortest_temporal_path(tnet)
    # print(tnet)
    # end = time.time()
    # print(end-start)
    #
    # start = time.clock()
    # print('intercontacttimes')
    # h=teneto.networkmeasures.intercontacttimes(tnet)
    # end = time.clock()
    # print(end-start)
    #
    # start = time.clock()
    # print('local_variation')
    # j=teneto.networkmeasures.local_variation(tnet)
    # end = time.clock()
    # print(end-start)

    # end = time.clock()
    # print(end-start)
