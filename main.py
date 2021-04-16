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
def run(args):
    log('running argument')
    log(args)
    model = args.model
    dataset_name = args.dataset
    log('model: {}'.format(model))

    if 'simulation' in dataset_name:
        n_times = args.numberTime
        n_nodes = args.numberNode
        n_days = args.numberSamples
        simProcess = args.simProcess
        prob = args.probability
        early_stopping = args.early_stopping
        is_test = args.is_test
        if 'scale' in dataset_name:
            simulation_type = 'scale-free'
        elif 'epidemic' in dataset_name:
            simulation_type = 'epidemic'

        scale = args.scale
        lr = args.learningrate
        continue_training = args.continueTraining
        use_wgan = args.use_wgan
        use_decoder = args.use_decoder
        constraint_method = args.constraint_method
        time_deconv = args.time_deconv
        time_sample_num = args.time_sample_num
        t0 = 0.1
        edge_contact_time = t0*0.5
        t_max = args.t_max
        n_eval_loop = args.n_eval_loop
        embedding_size = args.embedding_size
        rw_len = args.rw_len
        batch_size = args.batch_size
        init_walk_method = args.init_walk_method

        train_ratio = 0.8
        t_end = 1.
        gpu_id = 0

        max_iters = args.max_iters
        eval_every =  args.eval_every
        plot_every =  args.plot_every
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 10
        prefix = '{}-nodes-{}-samples-{}'.format(simulation_type, n_nodes, n_days)
        model_name = 'simulation-'.format(n_nodes, n_days)
        save_directory = 'snapshots-{}'.format(prefix)
        output_directory='outputs-{}'.format(prefix)
        timing_directory='timings-{}'.format(prefix)
        simulate_data_directory = 'data-{}'.format(prefix)
        simulate_data_file = '{}/{}.txt'.format(simulate_data_directory, simulate_data_directory)

        if not os.path.isdir(simulate_data_directory):
            os.mkdir(simulate_data_directory)
        if not os.path.isfile(simulate_data_file):
            log('simulate synthetic data, save in {}'.format(simulate_data_file))
            edges = multi_continuous_time_simulate(
                n_days=n_days, n_nodes=n_nodes, t0=t0, t_max=t_max, mean_tau=1., edge_contact_time=edge_contact_time,
                alpha=0.33, beta=0.34, gamma=0.33, delta_in=0.2, delta_out=0.0)
            np.savetxt(simulate_data_file, edges)
            log('simulated data : \n{}'.format(edges))
        else:
            edges = np.loadtxt(simulate_data_file)
            log('load simulated data from {}: \n{}'.format(simulate_data_file, edges))

        train_edges, test_edges = Split_Train_Test(edges, train_ratio)
        np.savetxt(fname='{}/{}_train.txt'.format(output_directory, prefix), X=train_edges)
        np.savetxt(fname='{}/{}_test.txt'.format(output_directory, prefix), X=test_edges)

        walker = TemporalWalker(n_nodes, train_edges, t_end,
                                scale, rw_len, batch_size,
                                init_walk_method=init_walk_method,
                                )

        tggan = TGGAN(N=n_nodes, rw_len=rw_len,
                      t_end=t_end,
                      edge_contact_time=edge_contact_time,
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
            early_stopping=early_stopping,
            is_test=is_test,
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
        early_stopping = args.early_stopping
        edge_contact_time = args.contact_time
        is_test = args.is_test

        # random data from metro
        userid = args.userid
        file = file = 'data/auth_user_{}.txt'.format(userid)
        edges = np.loadtxt(file)
        embedding_size = args.embedding_size
        rw_len = args.rw_len
        batch_size = args.batch_size
        init_walk_method = args.init_walk_method

        max_iters = args.max_iters
        eval_every =  args.eval_every
        plot_every =  args.plot_every
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 10
        model_name = 'auth-user-{}'.format(userid)
        save_directory = "snapshots-auth-user-{}".format(userid)
        output_directory='outputs-auth-user-{}'.format(userid)
        timing_directory='timings-auth-user-{}'.format(userid)

        train_edges, test_edges = Split_Train_Test(edges, train_ratio)


        walker = TemporalWalker(n_nodes, train_edges, t_end,
                                scale, rw_len, batch_size,
                                init_walk_method=init_walk_method,
                                )

        tggan = TGGAN(N=n_nodes, rw_len=rw_len,
                      t_end=t_end,
                      edge_contact_time=edge_contact_time,
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
            early_stopping=early_stopping,
            is_test=is_test,
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
        early_stopping = args.early_stopping
        edge_contact_time = args.contact_time
        is_test = args.is_test

        # random data from metro
        userid = args.userid
        file = 'data/metro_user_{}.txt'.format(userid)
        edges = np.loadtxt(file)
        embedding_size = args.embedding_size
        rw_len = args.rw_len
        batch_size = args.batch_size
        init_walk_method = args.init_walk_method

        max_iters = args.max_iters
        eval_every =  args.eval_every
        plot_every =  args.plot_every
        transitions_per_iter = batch_size * n_eval_loop
        eval_transitions = transitions_per_iter * 100
        model_name = 'metro-user-{}'.format(userid)
        save_directory = "snapshots-metro-user-{}".format(userid)
        output_directory='outputs-metro-user-{}'.format(userid)
        timing_directory='timings-metro-user-{}'.format(userid)
        train_edges, test_edges = Split_Train_Test(edges, train_ratio)

        walker = TemporalWalker(n_nodes, train_edges, t_end,
                                scale, rw_len, batch_size,
                                init_walk_method=init_walk_method,
                                )

        tggan = TGGAN(N=n_nodes, rw_len=rw_len,
                      t_end=t_end,
                      edge_contact_time=edge_contact_time,
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

        log_dict = tggan.train(
            train_edges=train_edges, test_edges=test_edges,
            n_eval_loop=n_eval_loop,
            early_stopping=early_stopping,
            is_test=is_test,
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
