from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import tensorflow as tf
import numpy as np
import teneto

from simulation import *
from utils import *
from dtwgan import *

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

    if dataset_name == 'metro':
        tf.compat.v1.reset_default_graph()

        log('-'*40)

        n_nodes = 91
        scale = 0.1
        rw_len = 3
        batch_size = 128
        train_ratio = 0.8
        t_plus = 24.
        embedding_size = 32
        gpu_id = 0

        lr = args.learningrate
        continue_training = args.continueTraining

        # random data from metro
        userid = args.userid
        file = args.file
        edges = np.loadtxt(file)
        train_edges, test_edges = Split_Train_Test(edges, train_ratio)

        walker = TemporalWalker(n_nodes, train_edges, t_plus,
                                scale, rw_len, batch_size,
                                init_walk_method='exp',
                                )

        dtwgan = DTWGAN(N=n_nodes, rw_len=rw_len, t_plus=t_plus,
                        walk_generator=walker.walk, batch_size=batch_size, gpu_id=gpu_id,
                        use_gumbel=True,
                        disc_iters=3,
                        W_down_discriminator_size=embedding_size,
                        W_down_generator_size=embedding_size,
                        l2_penalty_generator=1e-7,
                        l2_penalty_discriminator=5e-5,
                        generator_start_layers=[50, 10],
                        generator_layers=[50, 10],
                        discriminator_layers=[40, 10],
                        varDecoder_layers=[30],
                        temp_start=5,
                        learning_rate=lr,
                        momentum=0.9,
        )

        max_iters = 100000
        eval_every = 5000
        plot_every = 5000
        n_eval_loop = 3
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 10
        model_name = 'metro-user-{}'.format(userid)
        save_directory = "snapshots-user-{}".format(userid)
        output_directory='outputs-user-{}'.format(userid)

        stopping_criterion = "val"

        assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."

        if stopping_criterion == "val":  # use val criterion for early stopping
            stopping = None
        elif stopping_criterion == "eo":  # use eo criterion for early stopping
            stopping = 0.5  # set the target edge overlap here

        log_dict = dtwgan.train(A_orig=None, val_ones=None, val_zeros=None, n_eval_loop=n_eval_loop, stopping=stopping,
                                transitions_per_iter=transitions_per_iter, eval_transitions=eval_transitions,
                                eval_every=eval_every, plot_every=plot_every,
                                max_patience=20, max_iters=max_iters,
                                model_name=model_name,
                                save_directory=save_directory,
                                output_directory=output_directory,
                                continue_training=continue_training,
                                )
        log('-'*40)

    if dataset_name == 'auth':
        tf.compat.v1.reset_default_graph()

        log('-'*40)

        n_nodes = 13
        scale = 0.1
        rw_len = 3
        batch_size = 128
        train_ratio = 0.8
        t_plus = 1.
        embedding_size = 32
        gpu_id = 0

        lr = args.learningrate

        # random data from metro
        userid = args.userid
        file = args.file
        edges = np.loadtxt(file)
        train_edges, test_edges = Split_Train_Test(edges, train_ratio)


        walker = TemporalWalker(n_nodes, train_edges, t_plus,
                                scale, rw_len, batch_size,
                                init_walk_method='uniform',
                                )

        dtwgan = DTWGAN(N=n_nodes, rw_len=rw_len, t_plus=t_plus,
                        walk_generator=walker.walk, batch_size=batch_size, gpu_id=gpu_id,
                        use_gumbel=True,
                        disc_iters=3,
                        W_down_discriminator_size=embedding_size,
                        W_down_generator_size=embedding_size,
                        l2_penalty_generator=1e-7,
                        l2_penalty_discriminator=5e-5,
                        generator_start_layers=[50, 10],
                        generator_layers=[50, 10],
                        discriminator_layers=[40, 10],
                        varDecoder_layers=[30],
                        temp_start=5,
                        learning_rate=lr,
                        momentum=0.9
        )

        max_iters = 100000
        eval_every = 5000
        plot_every = 5000
        n_eval_loop = 3
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 10
        model_name = 'auth-user-{}'.format(userid)
        save_directory = "snapshots-auth-user-{}".format(userid)
        output_directory='outputs-auth-user-{}'.format(userid)

        stopping_criterion = "val"

        assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."

        if stopping_criterion == "val":  # use val criterion for early stopping
            stopping = None
        elif stopping_criterion == "eo":  # use eo criterion for early stopping
            stopping = 0.5  # set the target edge overlap here

        log_dict = dtwgan.train(A_orig=None, val_ones=None, val_zeros=None, n_eval_loop=n_eval_loop, stopping=stopping,
                                transitions_per_iter=transitions_per_iter, eval_transitions=eval_transitions,
                                eval_every=eval_every, plot_every=plot_every,
                                max_patience=20, max_iters=max_iters,
                                model_name=model_name,
                                save_directory=save_directory,
                                output_directory=output_directory,
                                )
        log('-'*40)
    log('end model')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DTWGAN paper")
    assert tf.__version__.startswith('1.14.0')

    models = ['dtwgan', 'graphrnn', 'graphvae', 'netgan', 'RNN', 'dsbm', 'markovian']
    parser.add_argument("-m", "--model", default="dtwgan", type=str,
                        help="one of: {}".format(", ".join(sorted(models))))
    parser.add_argument("-re", "--runEvaluation", default=False, type=bool,
                        help="if this run should run all evaluations")
    datasets = ['simulation', 'metro', 'auth']
    parser.add_argument("-d", "--dataset", default="metro", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-ui", "--userid", default=6, type=int,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-f", "--file", default="data/metro_user_6.npz", type=str,
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

    # DeepTemporalWalk
    parser.add_argument("-sc", "--scale", default=0.1, type=float,
                        help="scale of gaussian prior for kernel density estimation in DeepTemporalWalk")
    parser.add_argument("-rl", "--rw_len", default=4, type=int,
                        help="random walks maximum length in DeepTemporalWalk")
    parser.add_argument("-bs", "--batch_size", default=32, type=int,
                        help="random walks batch size in DeepTemporalWalk")

    # hyperparameter for GAN
    parser.add_argument("-lr", "--learningrate", default=0.00003, type=float,
                        help="if this run should run all evaluations")
    parser.add_argument("-ct", "--continueTraining", default=False, type=bool,
                        help="if this run is restored from a corrupted run")

    # run
    args = parser.parse_args()
    main(args)
    log('finish execution')
