from teneto import TemporalNetwork
import tacoma as tc
import teneto
import numpy as np
import matplotlib.pyplot as plt
import time


def simulation(n_nodes, n_times, prob, simProcess, ncontacts, lam):
    tnet = teneto.TemporalNetwork(timetype='continuous')
    if simProcess == 'rand_binomial':
        tnet.generatenetwork(simProcess, size=(n_nodes, n_nodes, n_times), nettype='bu', prob=prob, netrep='graphlet')
    elif simProcess == 'rand_poisson':
        tnet.generatenetwork(simProcess, nnodes=n_nodes, ncontacts=ncontacts, nettype='bu', lam=lam, netrep='graphlet')
    else:
        raise Exception('simulation method do not supported!')
    return tnet

def multi_simulate(n_days, n_nodes, n_times, prob, simProcess, ncontacts, lam):
    res_sim = []
    for d in range(n_days):
        tnet = simulation(n_nodes, n_times, prob, simProcess, ncontacts, lam)
        arr = tnet.network.values
        res_sim.extend(list(
            np.c_[[[d]]*arr.shape[0], arr]
        ))
    return np.array(res_sim).reshape(-1, 4)

if __name__ == "__main__":
    import numpy as np
    import tacoma as tc
    import networkx as nx

    # static structure parameters
    N = 5
    M = 10
    mean_degree = 1.5
    p = mean_degree / (N - 1.0)

    # temporal parameters
    edge_lists = []
    mean_tau = 1.0
    t0 = 0.0
    tmax = 100.0
    t = []
    this_time = t0

    while this_time < tmax:
        #     G = nx.fast_gnp_random_graph(N, p) # Generate a new random network
        G = nx.gnm_random_graph(N, M)  # Generate a Erdos Renyi network
        #     G = nx.barabasi_albert_graph(N, M) # Generate a Erdos Renyi network
        these_edges = list(G.edges())
        t.append(this_time)
        edge_lists.append(these_edges)

        this_time += np.random.exponential(scale=1 / mean_tau)

    # save to _tacoma-object
    el = tc.edge_lists()

    el.N = N
    el.t = t
    el.edges = edge_lists
    el.tmax = tmax
    print(el.t)
    print(el.edges)
    print("Number of mistakes:", tc.verify(el))
    sis = tc.SIS(N, tmax, 0.1, 0.1)
    result = tc.gillespie_SIS(el, sis)
    print(sis)

    from tacoma.drawing import edge_activity_plot

    edge_activity_plot(el,
                       time_normalization_factor=1.,
                       time_unit='h',
                       alpha=1.0,  # opacity
                       linewidth=1.5,
                       )

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

    # n_nodes = 10
    # n_times = 100
    # lam = int(n_times / 10)
    # n_days = 1000
    # prob = (0.5, 0.5)
    # ncontacts=3
    # # simProcess = 'rand_binomial'
    # simProcess = 'rand_poisson'
    # tnet = simulation(n_nodes, n_times, prob, simProcess, ncontacts, lam)
    # print('one sampled graph:\n', tnet.network.values.shape)
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
    # start = time.clock()
    # print('shortest_temporal_path')
    # # k=teneto.networkmeasures.sid(tnet,tnet,calc='global')
    # g=teneto.networkmeasures.shortest_temporal_path(tnet)
    # end = time.clock()
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
