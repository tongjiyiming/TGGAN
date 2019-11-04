from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import tensorflow as tf
import numpy as np
import teneto

from simulation import *
from utils import *
from tggan import *

# main run
def main(args):
    log('running argument')
    log(args)
    model = args.model
    dataset_name = args.dataset
    log('model: {}'.format(model))

    if dataset_name == 'simulation':
        file = args.file
        n_times = args.numberTime
        n_nodes = args.numberNode
        simProcess = args.simProcess
        prob = args.probability

        scale = args.scale
        rw_len = args.rw_len
        batch_size = args.batch_size

        log('simulate data')
        data_sim = simulation(n_nodes, n_times, prob, simProcess)
        log('simulated data : {}'.format(data_sim))

    if dataset_name == 'auth':
        log('use auth data')
        tf.compat.v1.reset_default_graph()

        log('-'*40)

        scale = 0.1
        train_ratio = 0.8
        t_end = 1.
        gpu_id = 0


        lr = args.learningrate
        continue_training = args.continueTraining
        use_wgan = args.use_wgan
        use_decoder = args.use_decoder
        constraint_method = args.constraint_method
        time_deconv = args.time_deconv
        time_sample_num = args.time_sample_num
        n_eval_loop = args.n_eval_loop

        # random data from metro
        userid = args.userid
        file = args.file
        edges = np.loadtxt(file)
        n_nodes = 28
        embedding_size = args.embedding_size
        rw_len = args.rw_len
        batch_size = args.batch_size

        train_edges, test_edges = Split_Train_Test(edges, train_ratio)


        walker = TemporalWalker(n_nodes, train_edges, t_end,
                                scale, rw_len, batch_size,
                                init_walk_method='uniform',
                                )

        tggan = TGGAN(N=n_nodes, rw_len=rw_len,
                      t_end=t_end,
                      walk_generator=walker.walk, batch_size=batch_size, gpu_id=gpu_id,
                      noise_type="Gaussian",
                      disc_iters=3,
                      W_down_discriminator_size=embedding_size,
                      W_down_generator_size=embedding_size,
                      l2_penalty_generator=1e-7,
                      l2_penalty_discriminator=5e-5,
                      generator_x_up_layers=[64],
                      generator_t0_up_layers=[128],
                      generator_tau_up_layers=[128],
                      generator_layers=[100, 20],
                      discriminator_layers=[80, 20],
                      temp_start=5,
                      learning_rate=lr,
                      use_gumbel=True,
                      use_wgan=use_wgan,
                      wasserstein_penalty=10,
                      use_decoder=use_decoder,
                      constraint_method=constraint_method,
                      )

        max_iters = 100000
        eval_every = 1000
        plot_every = 1000
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 10
        model_name = 'auth-user-{}'.format(userid)
        save_directory = "snapshots-auth-user-{}".format(userid)
        output_directory='outputs-auth-user-{}'.format(userid)

        log_dict = tggan.train(
            train_edges=train_edges, test_edges=test_edges,
            n_eval_loop=n_eval_loop,
           stopping=None,
           eval_transitions=eval_transitions,
           eval_every=eval_every, plot_every=plot_every,
           max_patience=20, max_iters=max_iters,
           model_name=model_name,
           save_directory=save_directory,
           output_directory=output_directory,
           )
        log('-'*40)

    if dataset_name == 'metro':
        log('use metro data')
        tf.compat.v1.reset_default_graph()

        log('-'*40)

        scale = 0.1
        train_ratio = 0.9
        t_end = 1.
        gpu_id = 0

        lr = args.learningrate
        continue_training = args.continueTraining
        use_wgan = args.use_wgan
        use_decoder = args.use_decoder
        constraint_method = args.constraint_method
        time_deconv = args.time_deconv
        time_sample_num = args.time_sample_num
        n_eval_loop = args.n_eval_loop

        # random data from metro
        userid = args.userid
        file = args.file
        edges = np.loadtxt(file)
        n_nodes = 91
        embedding_size = args.embedding_size
        rw_len = args.rw_len
        is_teleport = args.is_teleport
        batch_size = args.batch_size

        train_edges, test_edges = Split_Train_Test(edges, train_ratio)

        walker = TemporalWalker(n_nodes, train_edges, t_end,
                                scale, rw_len, batch_size,
                                init_walk_method='uniform',
                                isTeleport=is_teleport,
                                )

        tggan = TGGAN(N=n_nodes, rw_len=rw_len,
                      t_end=t_end,
                      walk_generator=walker.walk, batch_size=batch_size, gpu_id=gpu_id,
                      noise_type="Gaussian",
                      disc_iters=3,
                      W_down_discriminator_size=embedding_size,
                      W_down_generator_size=embedding_size,
                      generator_time_deconv_output_depth=time_deconv,
                      generator_time_sample_num=time_sample_num,
                      l2_penalty_generator=1e-7,
                      l2_penalty_discriminator=5e-5,
                      generator_x_up_layers=[64],
                      generator_t0_up_layers=[128],
                      generator_tau_up_layers=[128],
                      generator_layers=[100, 20],
                      discriminator_layers=[80, 20],
                      temp_start=5,
                      learning_rate=lr,
                      use_gumbel=True,
                      use_wgan=use_wgan,
                      wasserstein_penalty=10,
                      use_decoder=use_decoder,
                      constraint_method=constraint_method,
                      )

        max_iters = 100000
        eval_every = 1000
        plot_every = 1000
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 100
        model_name = 'metro-user-{}'.format(userid)
        save_directory = "snapshots-metro-user-{}".format(userid)
        output_directory='outputs-metro-user-{}'.format(userid)

        log_dict = tggan.train(
            train_edges=train_edges, test_edges=test_edges,
            n_eval_loop=n_eval_loop,
           stopping=None,
           eval_transitions=eval_transitions,
           eval_every=eval_every, plot_every=plot_every,
           max_patience=20, max_iters=max_iters,
           model_name=model_name,
           save_directory=save_directory,
           output_directory=output_directory,
           )
        log('-'*40)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TGGAN paper")
    assert tf.__version__.startswith('1.14.0')

    models = ['tggan', 'graphrnn', 'graphvae', 'netgan', 'RNN', 'dsbm', 'markovian']
    parser.add_argument("-m", "--model", default="tggan", type=str,
                        help="one of: {}".format(", ".join(sorted(models))))
    parser.add_argument("-re", "--runEvaluation", default=False, type=bool,
                        help="if this run should run all evaluations")
    datasets = ['simulation', 'metro', 'auth']
    parser.add_argument("-d", "--dataset", default="metro", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-ui", "--userid", default=4, type=int,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-f", "--file", default="data/metro_user_4.txt", type=str,
                        help="file path of data in format [[d, i, j, t], ...]")
    processes = ['rand_binomial', 'rand_poisson']
    parser.add_argument("-sp", "--simProcess", default="rand_binomial", type=str,
                        help="one of: {}".format(", ".join(sorted(processes))))
    parser.add_argument("-nn", "--numberNode", default=10, type=int,
                        help="if run simulation data, this is the number of nodes")
    parser.add_argument("-p", "--probability", default=0.5, type=int,
                        help="if run simulation data, this is the number of time slices")
    parser.add_argument("-nt", "--numberTime", default=10, type=int,
                        help="this is the number of time slices for both real data and simulation data")

    parser.add_argument("-sc", "--scale", default=0.1, type=float,
                        help="scale of gaussian prior for kernel density estimation in DeepTemporalWalk")

    # DeepTemporalWalk
    parser.add_argument("-bs", "--batch_size", default=32, type=int,
                        help="random walks batch size in DeepTemporalWalk")
    parser.add_argument("-lr", "--learningrate", default=0.0003, type=float,
                        help="if this run should run all evaluations")
    parser.add_argument("-rl", "--rw_len", default=1, type=int,
                        help="random walks maximum length in DeepTemporalWalk")
    parser.add_argument("-it", "--is_teleport", default=True, type=bool,
                        help="if perform a prior teleport in spatial graph")
    parser.add_argument("-uw", "--use_wgan", default=True, type=bool,
                        help="if use WGAN loss function")
    parser.add_argument("-ud", "--use_decoder", default='deep', type=str,
                        help="if decoder function")
    parser.add_argument("-es", "--embedding_size", default=32, type=int,
                        help="embedding size of nodes, W_down")
    parser.add_argument("-td", "--time_deconv", default=8, type=int,
                        help="deconv output channels number")
    parser.add_argument("-ts", "--time_sample_num", default=4, type=int,
                        help="time sampling number")
    parser.add_argument("-cm", "--constraint_method", default='min_max', type=str,
                        help="time constraint computing method")
    parser.add_argument("-ne", "--n_eval_loop", default=4, type=int,
                        help="number of walk loops")
    parser.add_argument("-ct", "--continueTraining", default=False, type=bool,
                        help="if this run is restored from a corrupted run")

    # run
    args = parser.parse_args()
    main(args)
    log('finish execution')
