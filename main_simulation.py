from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse
import tensorflow as tf
from main import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TGGAN paper")
    assert tf.__version__.startswith('1.14.0')

    models = ['tggan', 'graphrnn', 'graphvae', 'netgan', 'RNN', 'dsbm', 'markovian']
    parser.add_argument("-m", "--model", default="tggan", type=str,
                        help="one of: {}".format(", ".join(sorted(models))))
    parser.add_argument("-re", "--runEvaluation", default=False, type=bool,
                        help="if this run should run all evaluations")

    datasets = ['simulation_scale_free', 'simulation_epidemic_process', 'metro', 'auth']

    parser.add_argument("-d", "--dataset", default="simulation_scale_free", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-ui", "--userid", default=0, type=int,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    # parser.add_argument("-f", "--file", default="data/auth_user_0.txt", type=str,
    #                     help="file path of data in format [[d, i, j, t], ...]")

    processes = ['rand_binomial', 'rand_poisson']

    parser.add_argument("-sp", "--simProcess", default="rand_binomial", type=str,
                        help="one of: {}".format(", ".join(sorted(processes))))
    parser.add_argument("-nn", "--numberNode", default=10, type=int,
                        help="if run simulation data, this is the number of nodes")
    parser.add_argument("-ns", "--numberSamples", default=100, type=int,
                        help="if run simulation data, this is the number of samples of temporal graphs (unique d value)")
    parser.add_argument("-p", "--probability", default=0.5, type=int,
                        help="if run simulation data, this is the number of time slices")
    parser.add_argument("-nt", "--numberTime", default=10, type=int,
                        help="this is the number of time slices for both real data and simulation data")

    parser.add_argument("-sc", "--scale", default=0.1, type=float,
                        help="scale of gaussian prior for kernel density estimation in DeepTemporalWalk")
    parser.add_argument("-iw", "--init_walk_method", default='uniform', type=str,
                        help="TemporalWalk sampler")

    # DeepTemporalWalk
    parser.add_argument("-bs", "--batch_size", default=32, type=int,
                        help="random walks batch size in DeepTemporalWalk")
    parser.add_argument("-lr", "--learningrate", default=0.003, type=float,
                        help="if this run should run all evaluations")
    parser.add_argument("-rl", "--rw_len", default=8, type=int,
                        help="random walks maximum length in DeepTemporalWalk")
    parser.add_argument("-uw", "--use_wgan", default=True, type=bool,
                        help="if use WGAN loss function")
    parser.add_argument("-ud", "--use_decoder", default='deep', type=str,
                        help="if decoder function")
    parser.add_argument("-es", "--embedding_size", default=8, type=int,
                        help="embedding size of nodes, W_down")
    parser.add_argument("-td", "--time_deconv", default=8, type=int,
                        help="deconv output channels number")
    parser.add_argument("-ts", "--time_sample_num", default=4, type=int,
                        help="time sampling number")
    parser.add_argument("-cm", "--constraint_method", default='min_max', type=str,
                        help="time constraint computing method")
    parser.add_argument("-ne", "--n_eval_loop", default=10, type=int,
                        help="number of walk loops")

    parser.add_argument("-mi", "--max_iters", default=100000, type=int,
                        help="max iterations")
    parser.add_argument("-ev", "--eval_every", default=1000, type=int,
                        help="evaluation interval of epochs")
    parser.add_argument("-pe", "--plot_every", default=1000, type=int,
                        help="plot generated graph interval of epochs")

    # parser.add_argument("-mi", "--max_iters", default=1000, type=int,
    #                     help="max iterations")
    # parser.add_argument("-ev", "--eval_every", default=500, type=int,
    #                     help="evaluation interval of epochs")
    # parser.add_argument("-pe", "--plot_every", default=500, type=int,
    #                     help="plot generated graph interval of epochs")

    parser.add_argument("-te", "--is_test", default=False, type=bool,
                        help="if this is a testing period.")
    parser.add_argument("--contact_time", default=0.01, type=float,
                        help="stop training if evaluation metrics are good enough")
    parser.add_argument("--early_stopping", default=None, type=float,
                        help="stop training if evaluation metrics are good enough")
    parser.add_argument("-ct", "--continueTraining", default=False, type=bool,
                        help="if this run is restored from a corrupted run")

    # run
    args = parser.parse_args()
    # for n_nodes in [100, 300, 900, 2700]:
    #     for n_days in [200, 400, 600, 800]:
    #         args.numberNode = n_nodes
    #         args.numberSamples = n_days
    #         tf.compat.v1.reset_default_graph()
    #         main(args)

    if args.is_test:
        node_list = [10]
        embedding_size_list = [8]
        early_stopping_list = [1e-2]
        eval_step_list = [2]
    else:
        node_list = [100, 500, 2500]
        t_max_list = [20, 50, 70]
        n_eval_loop_list = [25, 60, 80]
        embedding_size_list = [32, 64, 128]
        sample_list = [200, 100, 100]
        early_stopping_list = [1e-10, 1e-10, 1e-10]

    # for i in range(2, len(node_list)):
    for i in range(1):
        n_nodes = node_list[i]
        t_max = t_max_list[i]
        n_eval_loop = n_eval_loop_list[i]
        # if n_nodes > 1000:
        #     t_max = int(np.sqrt(n_nodes))
        #     n_eval_loop = t_max + 20
        # else:
        #     t_max = int(np.sqrt(n_nodes)) * 2
        #     n_eval_loop = t_max + 10


        args.numberNode = n_nodes
        args.t_max = t_max
        args.n_eval_loop = n_eval_loop
        args.embedding_size = embedding_size_list[i]
        args.numberSamples = sample_list[i]
        args.contact_time = 0.1 / t_max
        args.early_stopping = early_stopping_list[i]
        tf.compat.v1.reset_default_graph()
        run(args)
    log('finish execution')
