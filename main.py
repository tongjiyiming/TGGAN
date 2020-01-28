from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')

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
        n_days = args.numberSamples
        simProcess = args.simProcess
        prob = args.probability

        scale = args.scale
        rw_len = args.rw_len
        batch_size = args.batch_size

        lr = args.learningrate
        continue_training = args.continueTraining
        use_wgan = args.use_wgan
        use_decoder = args.use_decoder
        constraint_method = args.constraint_method
        time_deconv = args.time_deconv
        time_sample_num = args.time_sample_num
        n_eval_loop = args.n_eval_loop
        embedding_size = args.embedding_size
        rw_len = args.rw_len
        batch_size = args.batch_size

        train_ratio = 0.8
        t_end = 1.
        gpu_id = 0

        max_iters = args.max_iters
        eval_every =  args.eval_every
        plot_every =  args.plot_every
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 10
        model_name = 'simulation-nodes-{}-samples-{}'.format(n_nodes, n_days)
        save_directory = 'snapshots-nodes-{}-samples-{}'.format(n_nodes, n_days)
        output_directory='outputs-nodes-{}-samples-{}'.format(n_nodes, n_days)
        timing_directory='timings-nodes-{}-samples-{}'.format(n_nodes, n_days)
        data_directory = 'data-nodes-{}-samples-{}'.format(n_nodes, n_days)
        data_file = '{}/data-nodes-{}-samples-{}.txt'.format(data_directory, n_nodes, n_days)
        if not os.path.isdir(data_directory):
            os.mkdir(data_directory)

        if not os.path.isfile(data_file):
            log('simulate data')
            edges = multi_continuous_time_simulate(n_days, n_nodes)
            edges = np.array(edges)
            log('simulated data : \n{}'.format(edges))
            np.savetxt(data_file, edges)
        else:
            log('loading simulated data')
            edges = np.loadtxt(data_file)

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
            timing_directory=timing_directory,
        )
        log('-'*40)

    if dataset_name == 'auth':
        log('use auth data')
        tf.compat.v1.reset_default_graph()

        log('-'*40)

        scale = 0.1
        train_ratio = 0.8
        t_end = 1.
        gpu_id = 0
        n_nodes = 27

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

        max_iters = args.max_iters
        eval_every =  args.eval_every
        plot_every =  args.plot_every
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 10
        model_name = 'auth-user-{}'.format(userid)
        save_directory = "snapshots-auth-user-{}".format(userid)
        output_directory='outputs-auth-user-{}'.format(userid)
        timing_directory='timings-auth-user-{}'.format(userid)

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
            timing_directory=timing_directory,
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
        n_nodes = 91

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

        max_iters = args.max_iters
        eval_every =  args.eval_every
        plot_every =  args.plot_every
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 100
        model_name = 'metro-user-{}'.format(userid)
        save_directory = "snapshots-metro-user-{}".format(userid)
        output_directory='outputs-metro-user-{}'.format(userid)
        timing_directory='timings-metro-user-{}'.format(userid)

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
            timing_directory=timing_directory,
        )
        log('-'*40)
    return