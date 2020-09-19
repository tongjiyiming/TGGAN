
import numpy as np
np.set_printoptions(precision=6, suppress=True)
from scipy.stats import gaussian_kde, norm
from datetime import date, timedelta
import matplotlib.pyplot as plt

class TemporalWalker:
    """
    Helper class to generate temporal random walks on the input user-trips matrix.
    The matrix gets shape: [day, hour, origin, destination]
    Parameters
    -----------
    edges: edges [[d, i, j]], shape: samples x 3
    edges_times: real time of edges [time], shape: samples x 1
    """

    def __init__(self, n_nodes, edges, t_end,
                 scale=0.1, rw_len=4, batch_size=8,
                 init_walk_method='uniform', isTeleport=False, isJump=False,
                 ):
        if edges.shape[1] != 4: raise Exception('edges must have shape: samples x 4')

        self.n_nodes = n_nodes
        self.t_end = t_end
        self.edges_days = edges[:, [0]]
        self.edges = edges[:, [1, 2]]
        self.edges_times = edges[:, [3]]
        self.rw_len = rw_len
        self.batch_size = batch_size
        # self.loc = loc
        self.scale = scale
        self.init_walk_method = init_walk_method
        self.isTeleport = isTeleport
        self.isJump = isJump

        # # kernel density estimation around current time
        # e_t_dict = {}
        # for i in range(len(self.edges)):
        #     e = list(self.edges[i, ])

    def walk(self):
        while True:
            yield temporal_random_walk(
                self.n_nodes, self.edges_days, self.edges, self.edges_times, self.t_end,
                self.scale, self.rw_len, self.batch_size,
                self.init_walk_method, self.isTeleport, self.isJump,
            )

# @jit(nopython=True)
def temporal_random_walk(n_nodes, edges_days, edges, edges_times, t_end,
                         scale, rw_len, batch_size,
                         init_walk_method, isTeleport, isJump
                         ):
    unique_days = np.unique(edges_days.reshape(1, -1)[0])
    walks = []

    for _ in range(batch_size):
        while True:
            # select a day with uniform distribution
            walk_day = np.random.choice(unique_days)
            mask = edges_days.reshape(1, -1)[0] == walk_day
            # subset for this day
            walk_day_edges = edges[mask]
            walk_day_times = edges_times[mask]

            # select a start edge. and unbiased or biased to the starting edges
            n = walk_day_edges.shape[0]
            if n >= rw_len: break

        n = n - rw_len + 1
        if init_walk_method is 'uniform': probs = Uniform_Prob(n)
        elif init_walk_method is 'linear': probs = Linear_Prob(n)
        elif init_walk_method is 'exp': probs = Exp_Prob(n)
        else: raise Exception('wrong init_walk_method!')

        if n == 1: start_walk_inx = 0
        else: start_walk_inx = np.random.choice(n, p=probs)

        # choice nodes from start to walker lengths. if isTeleport: sample by the teleport probability
        # which allow sample a nearby nodes based on a small random teleport
        if isTeleport:
            # selected_walks = np.apply_along_axis(
            #     func1d=lambda x: np.random.choice(n_nodes, p=teleport_probs[x[0]]),
            #     axis=1,
            #     arr=selected_walks,
            # )
            raise Exception('isTeleport not implemented yet')
        else:
            selected_walks = walk_day_edges[start_walk_inx:start_walk_inx + rw_len]
            selected_times = walk_day_times[start_walk_inx:start_walk_inx + rw_len]
        if isJump:
            raise Exception('isJump not implemented yet')

        # get start residual time
        if start_walk_inx == 0: t_res_0 = t_end
        else:
            # print('selected start:', selected_walks[0])
            t_res_0 = t_end - walk_day_times[start_walk_inx-1, 0]

        # convert to residual time
        selected_times = t_end - selected_times

        # # convert to edge index
        # selected_walks = [nodes_to_edge(e[0], e[1], n_nodes) for e in selected_walks]

        # add a stop sign of -1
        x = 1
        if start_walk_inx > 0: x = 0
        walks_mat = np.c_[selected_walks, selected_times]
        if rw_len > len(selected_walks):
            n_stops = rw_len - len(selected_walks)
            walks_mat = np.r_[walks_mat, [[-1, -1, -1]] * n_stops]

        # add start resdidual time
        if start_walk_inx == n-1:
            is_end = 1.
        else:
            is_end = 0.
        walks_mat = np.r_[[[x] + [is_end] + [t_res_0]], walks_mat]

        walks.append(walks_mat)
    return np.array(walks)

def nodes_to_edge(v, u, N):
    return v * N + u

def edge_to_nodes(e, N):
    return (e // N, e % N)

def Split_Train_Test(edges, train_ratio):
    days = sorted(np.unique(edges[:, 0]))
    b = days[int(train_ratio*len(days))]
    train_mask = edges[:, 0] <= b
    train_edges = edges[train_mask]
    test_edges = edges[ ~ train_mask]
    return train_edges, test_edges

def KDE(data):
    kernel = gaussian_kde(dataset=data, bw_method='silverman')
    return kernel

def Sample_Posterior_KDE(kernel, loc, scale, n):
    points = []
    n1 = 100
    for i in range(n):
        new_data = np.random.normal(loc, scale, n1)
        prior_probs = norm.pdf((new_data - loc) / scale)
        gau_probs = kernel.pdf(new_data)
        new_data_probs = prior_probs * gau_probs
        selected = np.random.choice(n1, p=new_data_probs / new_data_probs.sum())
        points.append(new_data[selected])
    return np.array(points)

def Exp_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = 1. / np.arange(1, n + 1, dtype=np.int)
    #     c = np.cbrt(1. / np.arange(1, n+1, dtype=np.int))
    exp_c = np.exp(c)
    return exp_c / exp_c.sum()

def Linear_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = np.arange(n+1, 1, dtype=np.int)
    return c / c.sum()

def Uniform_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = [1./n]
    return c * n


def Distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# def Teleport():
    # metro_stations = gpd.read_file('data/metro/metro-stations-regional-correct.geojson')
    # n = metro_stations.shape[0]
    # metro_stations['STATION'] = metro_stations['STATION'] - 1
    # distance_matrix = np.zeros((91, 91), dtype=np.float)
    # for i in range(n):
    #     i_id = metro_stations['STATION'][i]
    #     i_geom = metro_stations.geometry[i]
    #     for j in range(i + 1, n):
    #         j_id = metro_stations['STATION'][j]
    #         j_geom = metro_stations.geometry[j]
    #         # 0.02 degree is about 1 km
    #         distance_matrix[i_id, j_id] = 0.1 if Distance(i_geom, j_geom) < 0.02 else 0.001
    #
    # distance_matrix = distance_matrix + distance_matrix.T
    # np.fill_diagonal(distance_matrix, 1)
    # # distance_matrix = np.exp(distance_matrix)
    #
    # teleport_probs = distance_matrix / distance_matrix.sum(axis=1).reshape(-1, 1)
    #
    # return teleport_probs


def Get_Weekday(day):
    start_day = date(2016, 5, 1)
    current_day = start_day + timedelta(day)
    return current_day.weekday()


def Is_Weekend(day):
    w = Get_Weekday(day)
    return w in [0, 6]

def get_edge_times(data):
    edge_dict = {}
    for i, j, t in data[:, 1:]:
        edge = (int(i), int(j))
        if edge in edge_dict: edge_dict[edge] = edge_dict[edge] + [t]
        else: edge_dict[edge] = [t]
    return edge_dict

def plot_edge_time_hist(edge_dict, t0, tmax, bins, ymax, save_file=None, show=True):
    edges = list(edge_dict.keys())
    n_fig = len(edges)
    hight = n_fig * 2
    fig, ax = plt.subplots(n_fig, 1, figsize=(12, hight))
    for i in range(n_fig):
        e = edges[i]
        ax[i].hist(edge_dict[e], range=[t0, tmax], bins=bins)
        ax[i].set_ylim(0, ymax)
        ax[i].set_title('edge {}'.format(e))
        if i < n_fig-1: ax[i].set_xticklabels([])
    if save_file: plt.savefig(save_file)
    if show: plt.show()

def convert_graphs(fake_graphs):
    _, _, e, k = fake_graphs.shape
    fake_graphs = fake_graphs.reshape([-1, e, k])
    tmp_list = None
    for d in range(fake_graphs.shape[0]):
        d_graph = fake_graphs[d]
        d_graph = d_graph[d_graph[:, 2] > 0.]
        d_graph = np.c_[np.array([[d]] * d_graph.shape[0]), d_graph]
        if tmp_list is None:
            tmp_list = d_graph
        else:
            tmp_list = np.r_[tmp_list, d_graph]
    return tmp_list

def plot_comparisons(fake_graphs, fake_x_t0, real_walks, real_x_t0, train_edges, test_edges, t_end, n_nodes, output_directory, epoch, log):
    try:
        fake_walks = fake_graphs.reshape(-1, 3)
        fake_mask = fake_walks[:, 0] > -1
        fake_walks = fake_walks[fake_mask]
        fake_x_t0 = fake_x_t0.reshape(-1, 3)

        real_walks = real_walks.reshape(-1, 3)
        real_mask = real_walks[:, 0] > -1
        real_walks = real_walks[real_mask]
        real_x_t0 = real_x_t0.reshape(-1, 3)

        # truth_train_walks = train_edges[:, 1:3]
        truth_train_time = train_edges[:, 3:]
        # truth_train_res_time = self.params['t_end'] - truth_train_time
        truth_train_res_time = t_end - truth_train_time
        truth_train_walks = np.concatenate([train_edges[:, 1:3], truth_train_res_time], axis=1)
        truth_train_x_t0 = np.c_[np.zeros((len(train_edges), 1)), truth_train_res_time]
        truth_train_x_t0 = np.r_[truth_train_x_t0, np.ones((len(train_edges), 2))]

        truth_test_time = test_edges[:, 3:]
        truth_test_res_time = t_end - truth_test_time
        truth_test_walks = np.c_[test_edges[:, 1:3], truth_test_res_time]
        truth_test_x_t0 = np.c_[np.zeros((len(test_edges), 1)), truth_test_res_time]
        truth_test_x_t0 = np.r_[truth_test_x_t0, np.ones((len(test_edges), 2))]

        fake_e_list, fake_e_counts = np.unique(fake_walks[:, 0:2], return_counts=True, axis=0)
        real_e_list, real_e_counts = np.unique(real_walks[:, 0:2], return_counts=True, axis=0)
        truth_train_e_list, truth_train_e_counts = np.unique(truth_train_walks[:, 0:2], return_counts=True,
                                                             axis=0)
        truth_test_e_list, truth_test_e_counts = np.unique(truth_test_walks[:, 0:2], return_counts=True,
                                                           axis=0)
        truth_e_list, truth_e_counts = np.unique(
            np.r_[truth_test_walks[:, 0:2], truth_test_walks[:, 0:2]], return_counts=True, axis=0)
        n_e = len(truth_e_list)

        real_x_list, real_x_counts = np.unique(real_x_t0[:, 0], return_counts=True)
        fake_x_list, fake_x_counts = np.unique(fake_x_t0[:, 0], return_counts=True)
        truth_x_list, truth_x_counts = real_x_list, real_x_counts

        real_len_list, real_len_counts = np.unique(real_x_t0[:, 2], return_counts=True)
        fake_len_list, fake_len_counts = np.unique(fake_x_t0[:, 2], return_counts=True)
        truth_len_list, truth_len_counts = real_len_list, real_len_counts

        fig = plt.figure(figsize=(2 * 9, 2 * 9))
        fig.suptitle('Truth, Real, and Fake edges comparisons')
        dx = 0.3
        dy = dx
        zpos = 0

        fake_ax = fig.add_subplot(221, projection='3d')
        fake_ax.bar3d(fake_e_list[:, 0], fake_e_list[:, 1], zpos, dx, dy, fake_e_counts)
        # fake_ax.set_xlim([0, self.N])
        fake_ax.set_xlim([0, n_nodes])
        fake_ax.set_ylim([0, n_nodes])
        fake_ax.set_xticks(range(n_nodes))
        fake_ax.set_yticks(range(n_nodes))
        fake_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        fake_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        fake_ax.set_title('fake edges number: {}'.format(len(fake_e_list)))

        real_ax = fig.add_subplot(222, projection='3d')
        real_ax.bar3d(real_e_list[:, 0], real_e_list[:, 1], zpos, dx, dy, real_e_counts)
        real_ax.set_xlim([0, n_nodes])
        real_ax.set_ylim([0, n_nodes])
        real_ax.set_xticks(range(n_nodes))
        real_ax.set_yticks(range(n_nodes))
        real_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        real_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        real_ax.set_title('real edges number: {}'.format(len(real_e_list)))

        truth_ax = fig.add_subplot(223, projection='3d')
        truth_ax.bar3d(truth_train_e_list[:, 0], truth_train_e_list[:, 1], zpos, dx, dy,
                       truth_train_e_counts)
        truth_ax.set_xlim([0, n_nodes])
        truth_ax.set_ylim([0, n_nodes])
        truth_ax.set_xticks(range(n_nodes))
        truth_ax.set_yticks(range(n_nodes))
        truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        truth_ax.set_title('truth train edges number: {}'.format(len(truth_train_e_list)))

        truth_ax = fig.add_subplot(222, projection='3d')
        truth_ax.bar3d(truth_test_e_list[:, 0], truth_test_e_list[:, 1], zpos, dx, dy, truth_test_e_counts)
        truth_ax.set_xlim([0, n_nodes])
        truth_ax.set_ylim([0, n_nodes])
        truth_ax.set_xticks(range(n_nodes))
        truth_ax.set_yticks(range(n_nodes))
        truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
        truth_ax.set_title('truth test edges number: {}'.format(len(truth_test_e_list)))

        plt.tight_layout()
        plt.savefig('{}/iter_{}_edges_counts_validation.png'.format(output_directory, epoch), dpi=90)
        plt.close()

        fig, ax = plt.subplots(n_e + 4, 4, figsize=(4 * 6, (n_e + 4) * 4))
        i = 0
        real_ax = ax[i, 0]
        real_ax.bar(real_x_list, real_x_counts)
        real_ax.set_xlim([-1, 2])
        real_ax.set_title('real start x number: {}'.format(len(real_x_list)))

        fake_ax = ax[i, 1]
        fake_ax.bar(fake_x_list, fake_x_counts)
        fake_ax.set_xlim([-1, 2])
        fake_ax.set_title('fake start x number: {}'.format(len(fake_x_list)))

        truth_ax = ax[i, 2]
        truth_ax.bar(truth_x_list, truth_x_counts)
        truth_ax.set_xlim([-1, 2])
        truth_ax.set_title('truth start x number: {}'.format(len(truth_x_list)))
        truth_ax = ax[i, 3]
        truth_ax.bar(truth_x_list, truth_x_counts)
        truth_ax.set_xlim([-1, 2])
        truth_ax.set_title('truth start x number: {}'.format(len(truth_x_list)))

        i = 1
        max_xlim = max(max(real_len_list), max(fake_len_list)) + 1
        min_xlim = min(min(real_len_list), min(fake_len_list)) - 1
        real_ax = ax[i, 0]
        real_ax.bar(real_len_list, real_len_counts)
        real_ax.set_xlim([min_xlim, max_xlim])
        real_ax.set_title('real sampler ends: {}'.format(len(real_len_list)))

        fake_ax = ax[i, 1]
        fake_ax.bar(fake_len_list, fake_len_counts)
        fake_ax.set_xlim([min_xlim, max_xlim])
        fake_ax.set_title('fake sampler ends: {}'.format(len(fake_len_list)))

        truth_ax = ax[i, 2]
        truth_ax.bar(truth_len_list, truth_len_counts)
        truth_ax.set_xlim([min_xlim, max_xlim])
        truth_ax.set_title('truth sampler ends: {}'.format(len(truth_len_list)))
        truth_ax = ax[i, 3]
        truth_ax.bar(truth_len_list, truth_len_counts)
        truth_ax.set_xlim([min_xlim, max_xlim])
        truth_ax.set_title('truth sampler ends: {}'.format(len(truth_len_list)))

        i = 2
        for j, e in enumerate([0, 1]):
            real_ax = ax[i + j, 0]
            real_mask = real_x_t0[:, 0] == e
            real_times = real_x_t0[real_mask][:, 1]
            real_ax.hist(real_times, range=[0, 1], bins=100)
            real_ax.set_title('real x node: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                int(e), real_times.mean(), real_times.std()))

            fake_ax = ax[i + j, 1]
            fake_mask = fake_x_t0[:, 0] == e
            fake_times = fake_x_t0[fake_mask][:, 1]
            fake_ax.hist(fake_times, range=[0, 1], bins=100)
            fake_ax.set_title('fake x node: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                int(e), fake_times.mean(), fake_times.std()))

            truth_ax = ax[i + j, 2]
            truth_train_mask = truth_train_x_t0[:, 0] == e
            truth_train_times = truth_train_x_t0[truth_train_mask][:, 1]
            truth_ax.hist(truth_train_times, range=[0, 1], bins=100)
            truth_ax.set_title('truth train x node: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                int(e), truth_train_times.mean(), truth_train_times.std()))
            truth_ax = ax[i + j, 3]
            truth_test_mask = truth_test_x_t0[:, 0] == e
            truth_test_times = truth_test_x_t0[truth_test_mask][:, 1]
            truth_ax.hist(truth_test_times, range=[0, 1], bins=100)
            truth_ax.set_title('truth test x node: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                int(e), truth_test_times.mean(), truth_test_times.std()))

        i = 4
        for j, e in enumerate(truth_e_list):
            real_ax = ax[i + j, 0]
            real_mask = np.logical_and(real_walks[:, 0] == e[0], real_walks[:, 1] == e[1])
            real_times = real_walks[real_mask][:, 2]
            real_ax.hist(real_times, range=[0, 1], bins=100)
            real_ax.set_title('real start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], real_times.mean(), real_times.std()))

            fake_ax = ax[i + j, 1]
            fake_mask = np.logical_and(fake_walks[:, 0] == e[0], fake_walks[:, 1] == e[1])
            fake_times = fake_walks[fake_mask][:, 2]
            fake_ax.hist(fake_times, range=[0, 1], bins=100)
            fake_ax.set_title('fake start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], fake_times.mean(), fake_times.std()))

            truth_train_ax = ax[i + j, 2]
            truth_train_mask = np.logical_and(truth_train_walks[:, 0] == e[0],
                                              truth_train_walks[:, 1] == e[1])
            truth_train_times = truth_train_walks[truth_train_mask][:, 2]
            truth_train_ax.hist(truth_train_times, range=[0, 1], bins=100)
            truth_train_ax.set_title(
                'truth train start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                    [int(v) for v in e], truth_train_times.mean(), truth_train_times.std()))

            truth_test_ax = ax[i + j, 3]
            truth_test_mask = np.logical_and(truth_test_walks[:, 0] == e[0], truth_test_walks[:, 1] == e[1])
            truth_test_times = truth_test_walks[truth_test_mask][:, 2]
            truth_test_ax.hist(truth_test_times, range=[0, 1], bins=100)
            truth_test_ax.set_title(
                'truth test start edge: {} time distribution\nloc: {:.4f} scale: {:.4f}'.format(
                    [int(v) for v in e], truth_test_times.mean(), truth_test_times.std()))

        plt.tight_layout()
        plt.savefig('{}/iter_{}_validation.png'.format(output_directory, _it + 1))
        plt.close()

    except ValueError as e:
        print(e)
        log('reshape fake walks got error. Fake graphs shape: {} \n{}'.format(fake_graphs[0].shape, fake_walks[:3]))

if __name__ == '__main__':

    # scale = 0.1
    # rw_len = 2
    # batch_size = 8
    # train_ratio = 0.9
    #
    # # random data from metro
    # file = 'data/auth_user_0.txt'
    # edges = np.loadtxt(file)
    # n_nodes = int(edges[:, 1:3].max() + 1)
    # print(edges)
    # print('n_nodes', n_nodes)
    # t_end = 1.
    # train_edges, test_edges = Split_Train_Test(edges, train_ratio)
    # # print('train shape:', train_edges.shape)
    # # print('test shape:', test_edges.shape)
    # walker = TemporalWalker(n_nodes, train_edges, t_end,
    #                         scale, rw_len, batch_size,
    #                         init_walk_method='uniform',
    #                         )
    #
    # walks = walker.walk().__next__()
    # print('walk length:', rw_len)
    # print('walks shape:\n', walks.shape)
    # print('walks:\n', walks)

    # _it = '20191231-204105_assembled_graph_iter_30000'
    # fake_file = './outputs-auth-user-0-best/{}.npz'.format(_it)
    # res = np.load(fake_file)
    # fake_graphs = res['fake_graphs'][:2]
    # fake_graphs = convert_graphs(fake_graphs)
    # print('fake_graphs', fake_graphs.shape)
    # print(fake_graphs)
    # real_walks = res['fake_graphs']
    # np.savez_compressed(fake_file, fake_graphs=fake_graphs, real_walks=real_walks)

    _it = '20200129-001952_assembled_graph_iter_2'
    fake_file = './outputs-nodes-10-samples-1000/{}.npz'.format(_it)
    res = np.load(fake_file)
    fake_graphs = res['fake_graphs']
    print('fake_graphs', fake_graphs[fake_graphs[:, 0] < 3])

    print()