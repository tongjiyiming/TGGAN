import os
import shutil
import logging
import datetime
import time

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
if not os.path.isdir('logs'):
    os.makedirs('logs')
log_file = 'logs/log_{}.log'.format(datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# # create console handler and set level to debug
# th = logging.StreamHandler()
# th.setLevel(logging.INFO)
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str): logger.info(str)

import tensorflow as tf
log('is GPU available? {}'.format(tf.test.is_gpu_available()))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
import numpy as np
np.set_printoptions(precision=6, suppress=True)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from evaluation import *

class TGGAN:

    def __init__(self, N, rw_len, walk_generator,
                 t_end,
                 edge_contact_time,
                 generator_x_up_layers=[64],
                 generator_t0_up_layers=[128],
                 generator_tau_up_layers=[128],
                 generator_time_deconv_output_depth=8,
                 generator_time_sample_num=4,
                 constraint_method='min_max',
                 generator_layers=[40],
                 discriminator_layers=[30],
                 W_down_generator_size=128, W_down_discriminator_size=128, batch_size=128, noise_dim=16,
                 noise_type="Gaussian", learning_rate=0.0003, disc_iters=3, wasserstein_penalty=10,
                 l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, temp_start=5.0, min_temperature=0.5,
                 temperature_decay=1 - 5e-5, seed=15, gpu_id=0,
                 use_gumbel=True, use_wgan=False, use_decoder='normal',
                 legacy_generator=False):
        self.params = {
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            't_end': t_end,
            'edge_contact_time':edge_contact_time,
            'generator_x_up_layers': generator_x_up_layers,
            'generator_t0_up_layers': generator_t0_up_layers,
            'generator_tau_up_layers': generator_tau_up_layers,
            'generator_time_deconv_output_depth': generator_time_deconv_output_depth,
            'generator_time_sample_num': generator_time_sample_num,
            'constraint_method': constraint_method,
            'Generator_Layers': generator_layers,
            'Discriminator_Layers': discriminator_layers,
            'W_Down_Generator_size': W_down_generator_size,
            'W_Down_Discriminator_size': W_down_discriminator_size,
            'l2_penalty_generator': l2_penalty_generator,
            'l2_penalty_discriminator': l2_penalty_discriminator,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'Wasserstein_penalty': wasserstein_penalty,
            'temp_start': temp_start,
            'min_temperature': min_temperature,
            'temperature_decay': temperature_decay,
            'disc_iters': disc_iters,
            'use_gumbel': use_gumbel,
            'use_decoder': use_decoder,
            'use_wgan': use_wgan,
            'legacy_generator': legacy_generator
        }

        # assert rw_len > 1, "Random walk length must be > 1."

        tf.set_random_seed(seed)

        self.N = N
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.noise_dim = self.params['noise_dim']
        self.G_layers = self.params['Generator_Layers']
        self.D_layers = self.params['Discriminator_Layers']
        self.temp = tf.placeholder(1.0, shape=(), name="temperature")

        self.G_x_up_layers = self.params['generator_x_up_layers']
        self.G_t0_up_layers = self.params['generator_t0_up_layers']
        self.G_tau_up_layers = self.params['generator_tau_up_layers']
        self.G_t_deconv_output_depth = self.params['generator_time_deconv_output_depth']
        self.G_t_sample_n = self.params['generator_time_sample_num']

        # W_down and W_up for generator and discriminator

        self.W_down_x_generator = tf.get_variable(name='Generator.W_Down_x', dtype=tf.float32,
                                                  shape=[2, 1],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W_down_x_discriminator = tf.get_variable(name='Discriminator.W_Down_x', dtype=tf.float32,
                                                  shape=[2, 1],
                                                  initializer=tf.contrib.layers.xavier_initializer())

        self.W_down_end_generator = tf.get_variable(name='Generator.W_Down_end', dtype=tf.float32,
                                                  shape=[2, 1],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W_down_end_discriminator = tf.get_variable(name='Discriminator.W_Down_end', dtype=tf.float32,
                                                  shape=[2, 1],
                                                  initializer=tf.contrib.layers.xavier_initializer())

        self.t_end = tf.constant(value=self.params['t_end'], name='t_end',
                                 dtype=tf.float32, shape=[1])
        self.t0_deconv_filter = tf.get_variable('Generator.t0_deconv_filter',
                                                shape=[3, self.G_t_deconv_output_depth, 1],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
        self.empty_tau = tf.zeros([self.batch_size, 1], dtype=tf.float32, name="empty_tau")
        self.tau_deconv_filter = tf.get_variable('Generator.tau_deconv_filter',
                                                 shape=[3, self.G_t_deconv_output_depth, 1],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.W_down_t0 = tf.get_variable('Generator.W_down_t0', dtype=tf.float32,
                                         shape=[self.G_t_deconv_output_depth, 1],
                                         # constraint=lambda w: tf.clip_by_value(w, 1e-6, 1./(1+self.G_t_deconv_output_depth)),
                                         # initializer=tf.random_uniform_initializer(1e-2, 1e-1),
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         )
        self.W_down_t0_bias = tf.get_variable('Generator.W_down_t0_bias', dtype=tf.float32,
                                         shape=1,
                                         constraint=lambda w: tf.clip_by_value(w, 0, 1.),
                                         initializer=tf.contrib.layers.xavier_initializer())
        self.W_down_tau = tf.get_variable('Generator.W_down_tau', dtype=tf.float32,
                                         shape=[self.G_t_deconv_output_depth, 1],
                                         # constraint=lambda w: tf.clip_by_value(w, 1e-6, 1./self.G_t_deconv_output_depth),
                                         # initializer=tf.random_uniform_initializer(1e-2, 1e-1),
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                          )
        self.W_down_tau_bias = tf.get_variable('Generator.W_down_tau_bias', dtype=tf.float32,
                                         shape=1,
                                         constraint=lambda w: tf.clip_by_value(w, 0, 1.),
                                         initializer=tf.contrib.layers.xavier_initializer())

        self.W_down_generator = tf.get_variable('Generator.W_Down',
                                                shape=[self.N, self.params['W_Down_Generator_size']],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())

        self.W_down_discriminator = tf.get_variable('Discriminator.W_Down',
                                                    shape=[self.N, self.params['W_Down_Discriminator_size']],
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())

        self.W_up = tf.get_variable("Generator.W_up", shape=[self.G_layers[-1], self.N],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.b_W_up = tf.get_variable("Generator.W_up_bias", dtype=tf.float32, initializer=tf.zeros_initializer,
                                      shape=self.N)

        self.generator_function = self.generator_recurrent

        self.fake_x_inputs, self.fake_t0_res_inputs, \
        self.fake_node_inputs, self.fake_tau_inputs, \
        self.fake_ends \
            = self.generator_function(self.batch_size, edge_contact_time=self.params['edge_contact_time'],
                                      reuse=False, gumbel=use_gumbel, legacy=legacy_generator)

        # self.fake_inputs_discrete = self.generate_discrete(self.params['batch_size'], reuse=True,
        #                                                    n_eval_loop=3, gumbel=use_gumbel, legacy=legacy_generator)

        # Pre-fetch real random walks
        dataset = tf.data.Dataset.from_generator(generator=walk_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[self.batch_size, self.rw_len+1, 3])

        # dataset_batch = dataset.prefetch(2).batch(self.params['batch_size'])
        dataset_batch = dataset.prefetch(100)
        batch_iterator = dataset_batch.make_one_shot_iterator()
        self.real_data = batch_iterator.get_next()
        self.real_edge_inputs_discrete = tf.cast(self.real_data[:, 1:, 0:2], dtype=tf.int64)
        self.real_node_inputs_discrete = tf.reshape(self.real_edge_inputs_discrete, [self.batch_size, self.rw_len*2])
        self.real_node_inputs = tf.one_hot(self.real_node_inputs_discrete, self.N)
        self.real_tau_inputs = self.real_data[:, 1:, 2:3]

        self.real_x_input_discretes = tf.cast(self.real_data[:, 0, 0], dtype=tf.int64)
        self.real_x_inputs = tf.one_hot(self.real_x_input_discretes, 2)
        self.real_end_discretes = tf.cast(self.real_data[:, 0, 1], dtype=tf.int64)
        self.real_ends = tf.one_hot(self.real_end_discretes, 2)
        self.real_t0_res_inputs = self.real_data[:, 0:1, 2]

        self.discriminator_function = self.discriminator_recurrent
        self.disc_real = self.discriminator_function(self.real_x_inputs, self.real_t0_res_inputs,
                                                     self.real_node_inputs, self.real_tau_inputs,
                                                     self.real_ends,
                                                     )
        self.disc_fake = self.discriminator_function(self.fake_x_inputs, self.fake_t0_res_inputs,
                                                     self.fake_node_inputs, self.fake_tau_inputs,
                                                     self.fake_ends,
                                                     reuse=True)

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        if use_wgan:
            with tf.name_scope('WGAN_LOSS'):
                # WGAN lipschitz-penalty
                alpha_x = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0.,maxval=1.)
                self.differences_x = self.fake_x_inputs - self.real_x_inputs
                self.interpolates_x = self.real_x_inputs + (alpha_x * self.differences_x)

                alpha_t0 = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0.,maxval=1.)
                self.differences_t0 = self.fake_t0_res_inputs - self.real_t0_res_inputs
                self.interpolates_t0 = self.real_t0_res_inputs + (alpha_t0 * self.differences_t0)

                alpha_node = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0., maxval=1.)
                self.differences_node = self.fake_node_inputs - self.real_node_inputs
                self.interpolates_node = self.real_node_inputs + (alpha_node * self.differences_node)

                alpha_tau = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0., maxval=1.)
                self.differences_tau = self.fake_tau_inputs - self.real_tau_inputs
                self.interpolates_tau = self.fake_tau_inputs + (alpha_tau * self.differences_tau)

                alpha_end = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0.,maxval=1.)
                self.differences_end = self.fake_ends - self.real_ends
                self.interpolates_end = self.real_ends + (alpha_end * self.differences_end)

                self.gradients_x, self.gradients_t0, self.gradients_node, self.gradients_tau, self.gradients_end = tf.gradients(
                    self.discriminator_function(
                        self.interpolates_x, self.interpolates_t0, self.interpolates_node, self.interpolates_tau,
                        self.interpolates_end,
                        reuse=True), [
                        self.interpolates_x, self.interpolates_t0, self.interpolates_node, self.interpolates_tau,
                        self.interpolates_end
                    ])
                self.slopes = tf.sqrt(
                    tf.reduce_sum(tf.stack([
                        tf.reduce_sum(tf.square(self.gradients_x), reduction_indices=[1]),
                        tf.reduce_sum(tf.square(self.gradients_t0), reduction_indices=[1]),
                        tf.reduce_sum(tf.square(self.gradients_node), reduction_indices=[1, 2]),
                        tf.reduce_sum(tf.square(self.gradients_tau), reduction_indices=[1, 2]),
                        tf.reduce_sum(tf.square(self.gradients_end), reduction_indices=[1]),
                    ]), reduction_indices=[1]))
                self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
                self.disc_cost += self.params['Wasserstein_penalty'] * self.gradient_penalty

        with tf.name_scope('disc_l2_loss'):
            # weight regularization; we omit W_down from regularization
            self.disc_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                          if 'Disc' in v.name
                                          and not 'W_down' in v.name]) * self.params['l2_penalty_discriminator']
            self.disc_cost += self.disc_l2_loss

        with tf.name_scope('gen_l2_loss'):
            # weight regularization; we omit  W_down from regularization
            self.gen_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                         if 'Gen' in v.name
                                         and not 'W_down' in v.name]) * self.params['l2_penalty_generator']
            self.gen_cost += self.gen_l2_loss

        self.gen_params = [v for v in tf.trainable_variables() if 'Generator' in v.name]
        self.disc_params = [v for v in tf.trainable_variables() if 'Discriminator' in v.name]

        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                   beta2=0.9).minimize(self.gen_cost, var_list=self.gen_params)
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate'], beta1=0.5,
                                                    beta2=0.9).minimize(self.disc_cost, var_list=self.disc_params)

        with tf.name_scope('performance'):
            self.tf_disc_cost_ph = tf.placeholder(tf.float32,shape=None,name='disc_cost_summary')
            tf_disc_cost_summary = tf.summary.scalar('disc_cost', self.tf_disc_cost_ph)
            self.tf_gen_cost_ph = tf.placeholder(tf.float32,shape=None,name='gen_cost_summary')
            tf_gen_cost_summary = tf.summary.scalar('gen_cost', self.tf_gen_cost_ph)
        self.performance_summaries = tf.summary.merge([tf_disc_cost_summary, tf_gen_cost_summary])

        if gpu_id is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.InteractiveSession(config=config)
        self.init_op = tf.global_variables_initializer()

    def get_real_input_lengths(self, inputs_discrete):
        lengths = tf.math.reduce_sum(tf.cast(tf.math.less(-1, tf.cast(inputs_discrete, dtype=tf.int32)), dtype=tf.int64), axis=1)
        return lengths

    def discriminator_recurrent(self, x, t0_res, node_inputs, tau_inputs, end, reuse=None):
        with tf.variable_scope('Discriminator') as scope:
            if reuse == True:
                scope.reuse_variables()

            with tf.name_scope('DISC_X'):
                x_input_reshape = tf.reshape(x, [-1, 2])
                x_input_reshape = tf.matmul(x_input_reshape, self.W_down_x_discriminator)
                x_input_reshape = tf.layers.dense(x_input_reshape, int(self.W_down_discriminator.shape[-1]),
                                                  reuse=reuse, name="Discriminator.x_up_scale", activation=tf.nn.tanh,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope('DISC_T0'):
                t0_inputs = tf.reshape(t0_res, [-1, 1])
                # for ix, size in enumerate(self.D_layers):
                #     t0_inputs = tf.layers.dense(t0_inputs, size, name="Discriminator.t0_{}".format(ix), reuse=reuse,
                #                                activation=tf.nn.tanh,
                #                                kernel_initializer=tf.contrib.layers.xavier_initializer())
                t0_input_up = tf.layers.dense(t0_inputs, int(self.W_down_discriminator.shape[-1]),
                                                  reuse=reuse, name="Discriminator.t0_up_scale", activation=tf.nn.tanh,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope('DISC_NODES'):
                node_input_reshape = tf.reshape(node_inputs, [-1, self.N])
                node_output = tf.matmul(node_input_reshape, self.W_down_discriminator)
                node_output = tf.reshape(node_output, [-1, self.rw_len*2, int(self.W_down_discriminator.shape[-1])])
                node_output = tf.unstack(node_output, axis=1)

            with tf.name_scope('DISC_TAU'):
                tau_input_reshape = tf.reshape(tau_inputs, [-1, 1])
                tau_output = tf.layers.dense(tau_input_reshape, int(self.W_down_discriminator.shape[-1]),
                                             reuse=reuse, name='Discriminator.tau_up')
                tau_output = tf.reshape(tau_output, [-1, self.rw_len, int(self.W_down_discriminator.shape[-1])])
                tau_output = tf.unstack(tau_output, axis=1)

            with tf.name_scope('DISC_END'):
                end_input_reshape = tf.reshape(end, [-1, 2])
                end_input_reshape = tf.matmul(end_input_reshape, self.W_down_end_discriminator)
                end_input_reshape = tf.layers.dense(end_input_reshape, int(self.W_down_discriminator.shape[-1]),
                                                  reuse=reuse, name="Discriminator.end_up_scale", activation=tf.nn.tanh,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer())

            inputs = [x_input_reshape] + [t0_input_up]
            for i in range(self.rw_len):
                inputs += [node_output[i*2]] + [node_output[i*2+1]] + [tau_output[i]]
            inputs += [end_input_reshape]

            with tf.name_scope('DISC_LSTM'):
                def lstm_cell(lstm_size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

                disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

                output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell,
                                                                    inputs=inputs,
                                                                    dtype='float32',
                                                                    )
                last_output = output_disc[-1]
                final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
            return final_score

    def generator_recurrent(self, n_samples, edge_contact_time, reuse=None, z=None,
                            x_input=None, x_mode="uniform", t0_input=None,
                            edge_input=None, tau_input=None,
                            gumbel=True, legacy=False):
        with tf.variable_scope('Generator') as scope:
            if reuse is True:
                scope.reuse_variables()

            with tf.name_scope('NOISE'):
                # initial states h and c are randomly sampled for each lstm cell
                if z is None:
                    initial_states_noise = make_noise([n_samples, self.noise_dim], self.params['noise_type'])
                else:
                    initial_states_noise = z

            with tf.name_scope('INITIAL_STATES'):
                initial_states = []
                # Noise preprocessing
                for ix, size in enumerate(self.G_layers):
                    if legacy:  # old version to initialize LSTM. new version has less parameters and performs just as good.
                        h_intermediate = tf.layers.dense(initial_states_noise, size,
                                                         name="Generator.h_int_{}".format(ix + 1),
                                                         reuse=reuse, activation=tf.nn.tanh)
                        h = tf.layers.dense(h_intermediate, size, name="Generator.h_{}".format(ix + 1), reuse=reuse,
                                            activation=tf.nn.tanh)

                        c_intermediate = tf.layers.dense(initial_states_noise, size,
                                                         name="Generator.c_int_{}".format(ix + 1),
                                                         reuse=reuse, activation=tf.nn.tanh)
                        c = tf.layers.dense(c_intermediate, size, name="Generator.c_{}".format(ix + 1), reuse=reuse,
                                            activation=tf.nn.tanh)

                    else:
                        intermediate = tf.layers.dense(initial_states_noise, size, name="Generator.int_{}".format(ix + 1),
                                                       reuse=reuse, activation=tf.nn.tanh)
                        h = tf.layers.dense(intermediate, size, name="Generator.h_{}".format(ix + 1), reuse=reuse,
                                            activation=tf.nn.tanh)
                        c = tf.layers.dense(intermediate, size, name="Generator.c_{}".format(ix + 1), reuse=reuse,
                                            activation=tf.nn.tanh)
                    initial_states.append((c, h))

                state = initial_states

            def lstm_cell(lstm_size, name):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse,
                                                    name="LSTM_{}".format(name))

            with tf.variable_scope('LSTM_CELL'):
                self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size, "walks") for size in self.G_layers])
                # self.stacked_lstm_x = tf.contrib.rnn.MultiRNNCell([lstm_cell(size, "start") for size in self.G_layers])

            with tf.name_scope('GEN_START_X'):
                if x_mode == "uniform":
                    # generate start x, and its residual time
                    if x_input is not None:
                        x_output = x_input
                    else:
                        # generate start node binary if not need
                        x_output = tf.random_uniform(minval=0.6, maxval=1.9, shape=[n_samples, ])
                        x_output = tf.cast(x_output, dtype=tf.int64)
                        x_output = tf.one_hot(x_output, 2)
                elif x_mode == "generate":
                    # generate start x, and its residual time
                    if x_input is not None:
                        x_output = x_input
                    else:
                        inputs = tf.zeros([n_samples,self.params['W_Down_Discriminator_size']], dtype=tf.float32)
                        output, state = self.stacked_lstm.call(inputs, state)

                        # generate start node binary if not need
                        x_logit = output
                        for ix, size in enumerate(self.G_x_up_layers):
                            x_logit = tf.layers.dense(x_logit, size, name="Generator.x_logit_{}".format(ix),
                                                      reuse=reuse, activation=tf.nn.tanh)
                        x_logit = tf.layers.dense(x_logit, 2, name="Generator.x_logit_last",
                                                  reuse=reuse, activation=None)
                        self.x_logit = x_logit
                        # Perform Gumbel softmax to ensure gradients flow for e, and end node y
                        if gumbel: x_output = gumbel_softmax(x_logit, temperature=self.temp, hard=True)
                        else:      x_output = tf.nn.softmax(x_logit)

                x_down = tf.matmul(x_output, self.W_down_x_generator)
                # convert to input
                inputs = tf.layers.dense(x_down, self.params['W_Down_Discriminator_size'],
                                         name="Generator.x_lstm_input",
                                         reuse=reuse, activation=tf.nn.tanh)

            # LSTM steps
            node_outputs = []
            tau_outputs = []

            # generate the first three start elements: start x, residual time, and maximum possible length
            output, state = self.stacked_lstm.call(inputs, state)
            with tf.name_scope('GEN_START_TIME'):
                if t0_input is not None: # for evaluation generation
                    t0_res_output = t0_input
                else:
                    t0_res_output = self.generate_time(output, "t0")

                    if self.params['constraint_method'] != "none":
                        t0_res_output = self.time_constraint(
                            t0_res_output, method=self.params['constraint_method']) * self.t_end
                    condition = tf.math.equal(tf.argmax(x_output, axis=-1), 1)
                    t0_res_output = tf.where(condition, tf.ones_like(t0_res_output), t0_res_output)
                self.t0_res_output = t0_res_output
                res_time = t0_res_output
                # res_time = tf.stop_gradient(res_time)

                # convert to input
                inputs = tf.layers.dense(t0_res_output, self.params['W_Down_Discriminator_size'],
                                         name="Generator.t0_lstm_input",
                                         reuse=reuse, activation=tf.nn.tanh)
            # LSTM tine steps
            for i in range(self.rw_len):
                # generate temporal edge part
                for j in range(2):
                    # LSTM for first node
                    output, state = self.stacked_lstm.call(inputs, state)
                    with tf.variable_scope('GEN_NODE'):
                        if i > 0 or j > 0: tf.get_variable_scope().reuse_variables()

                        if edge_input is not None and i <= self.rw_len-2:  # for evaluation generation
                            output = edge_input[:, i * 2 + j]
                        else:
                            # Blow up to dimension N using W_up
                            logit = tf.matmul(output, self.W_up) + self.b_W_up
                            self.logit = logit

                            # Perform Gumbel softmax to ensure gradients flow
                            if gumbel: output = gumbel_softmax(logit, temperature=self.temp, hard=True)
                            else:      output = tf.nn.softmax(logit)

                        node_outputs.append(output)

                        # Back to dimension d
                        inputs = tf.matmul(output, self.W_down_generator)

                # LSTM for   tau
                output, state = self.stacked_lstm.call(inputs, state)
                with tf.variable_scope('GEN_TAU_TIME'):
                    if i > 0: tf.get_variable_scope().reuse_variables()

                    if tau_input is not None and i <= self.rw_len-2:  # for evaluation generation
                        tau = tau_input[:, i]
                    else:
                        tau = self.generate_time(output, "tau")

                        if self.params['constraint_method'] != "none":
                            tau = self.time_constraint(tau, method=self.params['constraint_method']) * res_time

                    self.tau = tau
                    res_time = tau
                    # res_time = tf.stop_gradient(res_time)

                    # save outputs
                    tau_outputs.append(tau)

                    # convert to input
                    inputs = tf.layers.dense(tau, int(self.W_down_generator.shape[-1]),
                                             name="Generator.tau_input", activation=tf.nn.tanh)

            # LSTM for end indicator
            output, state = self.stacked_lstm.call(inputs, state)
            with tf.name_scope('GEN_END'):
                # generate end binary
                end_logit = output
                for ix, size in enumerate(self.G_x_up_layers):
                    end_logit = tf.layers.dense(end_logit, size, name="Generator.end_logit_{}".format(ix),
                                              reuse=reuse, activation=tf.nn.tanh)
                end_logit = tf.layers.dense(end_logit, 2, name="Generator.end_logit_last",
                                          reuse=reuse, activation=None)
                self.end_logit = end_logit
                # Perform Gumbel softmax to ensure gradients flow for e, and end node y
                if gumbel: end_output = gumbel_softmax(end_logit, temperature=self.temp, hard=True)
                else:      end_output = tf.nn.softmax(end_logit)

                end_down = tf.matmul(end_output, self.W_down_end_generator)
                # # convert to input
                # inputs = tf.layers.dense(end_down, self.params['W_Down_Discriminator_size'],
                #                          name="Generator.end_lstm_input",
                #                          reuse=reuse, activation=tf.nn.tanh)

            node_outputs = tf.stack(node_outputs, axis=1)
            tau_outputs = tf.stack(tau_outputs, axis=1)

        return x_output, t0_res_output, node_outputs, tau_outputs, end_output

    def time_constraint(self, t, epsilon=1e-1, method='min_max'):
        with tf.name_scope('time_constraint'):
            if method == 'relu':
                t = tf.nn.relu(t) - tf.nn.relu(t - 1.)
            elif method == 'clip':
                t = tf.clip_by_value(t, 0., 1.)
            elif method == 'min_max':
                min_ = tf.math.reduce_min(t, axis=0)[0]
                # t = tf.case([
                #     (tf.math.less(min_, 0.), lambda : t - min_)
                # ], lambda : t)
                t = tf.case([
                    (tf.math.less(min_, epsilon), lambda: t - min_)
                ], lambda: t)

                max_ = tf.math.reduce_max(t, axis=0)[0]
                t = tf.case([
                    (tf.math.less(1., max_), lambda: t / max_)
                ], lambda : t)
                # t = tf.case([
                #     (tf.math.less(1.-epsilon, max_), lambda: t / max_)
                # ], lambda : t)

        return t

    def generate_time(self, output, name):
        n_samples = int(output.shape[0])
        if self.params['use_decoder'] == 'normal':
            with tf.name_scope('{}_normal_decoder'.format(name)):
                loc_t0 = output
                scale_t0 = output
                for ix, size in enumerate(self.G_t0_up_layers):
                    loc_t0 = tf.layers.dense(loc_t0, size,
                                             name="Generator.loc_{}_{}".format(name, ix),
                                             activation=tf.nn.tanh)
                    scale_t0 = tf.layers.dense(scale_t0, size,
                                               name="Generator.scale_{}_{}".format(name, ix),
                                               activation=tf.nn.tanh)
                loc_t0 = tf.layers.dense(loc_t0, 1, name="Generator.loc_{}_last".format(name),
                                         activation=None)
                scale_t0 = tf.layers.dense(scale_t0, 1, name="Generator.scale_{}_last".format(name),
                                           activation=None)
                t0_wait = [tf.truncated_normal([1], mean=loc_t0[i, 0], stddev=scale_t0[i, 0])
                            for i in range(n_samples)]
                t0_wait = tf.stack(t0_wait, axis=0)
        elif self.params['use_decoder'] == 'beta':
            with tf.name_scope('{}_beta_decoder'.format(name)):
                loc_t0 = output
                scale_t0 = output
                for ix, size in enumerate(self.G_t0_up_layers):
                    loc_t0 = tf.layers.dense(loc_t0, size,
                                             name="Generator.loc_{}_{}".format(name, ix),
                                             activation=tf.nn.tanh)
                    scale_t0 = tf.layers.dense(scale_t0, size,
                                               name="Generator.scale_{}_{}".format(name, ix),
                                               activation=tf.nn.tanh)
                loc_t0 = tf.layers.dense(loc_t0, 1, name="Generator.loc_{}_last".format(name),
                                         activation=None)
                scale_t0 = tf.layers.dense(scale_t0, 1, name="Generator.scale_{}_last".format(name),
                                           activation=None)
                t0_wait = self.beta_decoder(_alpha_param=loc_t0, _beta_param=scale_t0)
                t0_wait = tf.stack(t0_wait, axis=0)
        elif self.params['use_decoder'] == 'gamma':
            with tf.name_scope('{}_gamma_decoder'.format(name)):
                loc_t0 = output
                scale_t0 = output
                for ix, size in enumerate(self.G_t0_up_layers):
                    loc_t0 = tf.layers.dense(loc_t0, size,
                                             name="Generator.loc_{}_{}".format(name, ix),
                                             activation=tf.nn.tanh)
                    scale_t0 = tf.layers.dense(scale_t0, size,
                                               name="Generator.scale_{}_{}".format(name, ix),
                                               activation=tf.nn.tanh)
                loc_t0 = tf.layers.dense(loc_t0, 1, name="Generator.loc_{}_last".format(name),
                                         activation=None)
                scale_t0 = tf.layers.dense(scale_t0, 1, name="Generator.scale_{}_last".format(name),
                                           activation=None)
                t0_wait = [tf.random_gamma([1], alpha=loc_t0[i, 0], beta=scale_t0[i, 0])
                            for i in range(n_samples)]
                t0_wait = tf.stack(t0_wait, axis=0)
        elif self.params['use_decoder'] == 'deep':
            with tf.name_scope('{}_deep_decoder'.format(name)):
                t0_wait = output
                for ix, size in enumerate(self.G_t0_up_layers):
                    t0_wait = tf.layers.dense(t0_wait, size,
                                              name="Generator.{}_up_{}".format(name, ix),
                                              activation=tf.nn.tanh)
                t0_wait = tf.expand_dims(t0_wait, axis=2)

                # deconvolutional
                n_strides = 1
                t0_wait = tf.nn.conv1d_transpose(
                    t0_wait, filters=self.t0_deconv_filter,
                    output_shape=[n_samples, int(t0_wait.shape[1]) * n_strides,
                                  self.G_t_deconv_output_depth],
                    strides=n_strides, padding='SAME',
                    name='Generator.{}_deconv'.format(name))

                choice = tf.random_uniform([self.G_t_sample_n], maxval=t0_wait.shape[1],
                                           dtype=tf.int64)
                t0_wait = tf.gather(t0_wait, choice, axis=1)
                t0_wait = tf.reduce_mean(t0_wait, axis=1)
                t0_wait = tf.layers.dense(t0_wait, 1, name="Generator.{}_deconv_last".format(name),
                                          activation=None)
        else:
            raise Exception(
                "reparameterization trick {} not implemented yet. choose from 'normal', 'gamm', 'beta', 'deep'.".format(
                    self.params['use_decoder']
                ))
        return t0_wait

    def beta_decoder(self, _alpha_param, _beta_param, B = 5):
        # B is for shape augmentation
        alpha = tf.exp(_alpha_param)
        beta = tf.exp(_beta_param)
        size = _alpha_param.shape[0]
        # sample epsilon for each gamma
        epsilon_a = self.sample_pi(alpha + B, 1., size)[0]
        epsilon_b = self.sample_pi(beta + B, 1., size)[0]
        z_tilde_a = self.h(epsilon_a, alpha + B, 1.)
        z_tilde_b = self.h(epsilon_b, beta + B, 1.)
        z_a = self.shape_augmentation(z_tilde_a, B, alpha)
        z_b = self.shape_augmentation(z_tilde_b, B, beta)
        # get beta samples
        z = z_a / (z_a + z_b)
        return z

    def sample_pi(self, alpha, beta, size):
        gamma_samples = [tf.random_gamma([1], alpha[i, 0], beta) for i in range(size)]
        gamma_samples = tf.stack(gamma_samples, axis=0)
        return tf.stop_gradient(self.h_inverse(gamma_samples, alpha, beta))

    def h_inverse(self, z, alpha, beta):
        return tf.sqrt(9.0 * alpha - 3) * ((beta * z / (alpha - 1. / 3)) ** (1. / 3) - 1)

    # Transformation and its derivative
    # Transforms eps ~ N(0, 1) to proposal distribution
    def h(self, epsilon, alpha, beta):
        z = (alpha - 1. / 3.) * (1. + epsilon / tf.sqrt(9. * alpha - 3.)) ** 3. / beta
        return z

    def shape_augmentation(self, z_tilde, B, alpha):
        logz = self.log(z_tilde)
        for i in range(1, B + 1):
            u = tf.random_uniform(tf.shape(z_tilde))
            logz = logz + self.log(u) / (alpha + i - 1.)
        return tf.exp(logz)

    def log(self, x, eps=1e-8):
        return tf.log(x + eps)

    def generate_discrete(self, n_samples, edge_contact_time, n_eval_loop, reuse=True, z=None, gumbel=True, legacy=False):
        self.start_x_0 = tf.one_hot(tf.zeros(dtype=tf.int64, shape=[n_samples, ]), depth=2, name="start_x_0")
        self.start_x_1 = tf.one_hot(tf.ones(dtype=tf.int64, shape=[n_samples, ]), depth=2, name="start_x_1")
        self.start_t0 = tf.ones(dtype=tf.float32, shape=[n_samples, 1], name='start_t0')

        fake_x, fake_t0, fake_e, fake_tau, fake_end = [], [], [], [], []
        for i in range(n_eval_loop):
            if i == 0:
                fake_x_output, fake_t0_res_output, \
                fake_node_outputs, fake_tau_outputs, \
                fake_end_output = self.generator_function(
                    n_samples, edge_contact_time, reuse, z, x_input=self.start_x_1, t0_input=self.start_t0,
                    gumbel=gumbel, legacy=legacy)
            else:
                if self.rw_len == 1:
                    t0_input = fake_tau_outputs[:, -1, :]
                    fake_x_output, fake_t0_res_output, \
                    fake_node_outputs, fake_tau_outputs, \
                    fake_end_output = self.generator_function(
                        n_samples, edge_contact_time, reuse, z,
                        x_input=self.start_x_0, t0_input=t0_input,
                        gumbel=gumbel, legacy=legacy)
                else:
                    t0_input = fake_tau_outputs[:, 0, :]
                    edge_input = fake_node_outputs[:, 2:, :]
                    tau_input = fake_tau_outputs[:, 1:, :]
                    fake_x_output, fake_t0_res_output, \
                    fake_node_outputs, fake_tau_outputs, \
                    fake_end_output = self.generator_function(
                        n_samples, edge_contact_time, reuse, z,
                        x_input=self.start_x_0, t0_input=t0_input, edge_input=edge_input, tau_input=tau_input,
                        gumbel=gumbel, legacy=legacy)

            fake_x_outputs_discrete = tf.argmax(fake_x_output, axis=-1)
            fake_node_outputs_discrete = tf.argmax(fake_node_outputs, axis=-1)
            fake_end_discretes = tf.argmax(fake_end_output, axis=-1)

            fake_x.append(fake_x_outputs_discrete)
            fake_t0.append(fake_t0_res_output)
            fake_e.append(fake_node_outputs_discrete)
            fake_tau.append(fake_tau_outputs)
            fake_end.append(fake_end_discretes)

        return fake_x, fake_t0, fake_e, fake_tau, fake_end

    def train(self, train_edges, test_edges, max_iters=1000, early_stopping=None,
              eval_transitions=1e6, n_eval_loop=3, is_test=False,
              max_patience=5, eval_every=500, plot_every=1000,
              output_directory='outputs', save_directory="snapshots", timing_directory="timings",
              model_name=None, continue_training=False):
        starting_time = time.time()
        saver = tf.train.Saver()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        tensorboard_log_path = './graph'
        model_number = 0
        edge_contact_time = self.params['edge_contact_time']

        if early_stopping == None:  # if use VAL criterion
            log("**** Not using evaluation for early stopping ****")

        if not continue_training:
            log("**** Initializing... ****")
            self.session.run(self.init_op)
            log("**** Done.           ****")
        else:
            log("**** Continuing training without initializing weights. ****")
            # Find the file corresponding to the lowest vacant model number to store the snapshots into.
            while os.path.exists("{}/{}_iter_{}.ckpt".format(save_directory, model_name, model_number)):
                model_number += eval_every
            model_number -= eval_every
            save_file = "{}/{}_iter_{}.ckpt".format(save_directory, model_name, model_number)
            saver.restore(self.session, save_file)

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        if not os.path.isdir(timing_directory):
            os.makedirs(timing_directory)

        if not os.path.isdir(tensorboard_log_path):
            os.makedirs(tensorboard_log_path)
        else:
            shutil.rmtree(tensorboard_log_path)

        log("**** save computing graph to tensorboard in folder {}. ****".format(tensorboard_log_path))
        summ_writer = tf.summary.FileWriter(tensorboard_log_path, self.session.graph)

        # Some lists to store data into.
        gen_losses = []
        disc_losses = []
        graphs = []
        val_performances = []
        eo = []
        temperature = self.params['temp_start']

        # for evaluation fake walks
        p = 10
        n_smpls = self.batch_size * p
        n_eval_iters = int(eval_transitions / n_smpls)
        sample_many = self.generate_discrete(n_samples=n_smpls, edge_contact_time=edge_contact_time,
                                             n_eval_loop=n_eval_loop, reuse=True)

        # start main loop
        time_all = np.zeros(max_iters)
        log("**** Starting training. ****")

        for _it in range(max_iters):
            time_start = time.time()

            # Generator training iteration
            gen_loss, _ = self.session.run([self.gen_cost, self.gen_train_op],
                                           feed_dict={self.temp: temperature})

            _disc_l = []
            # Multiple discriminator training iterations.
            for _ in range(self.params['disc_iters']):
                disc_fake, disc_real, disc_loss, _ = self.session.run(
                    [self.disc_fake, self.disc_real, self.disc_cost, self.disc_train_op],
                    feed_dict={self.temp: temperature}
                )
                _disc_l.append(disc_loss)
            time_end = time.time()
            time_all[_it] = time_end - time_start

            gen_losses.append(gen_loss)
            disc_losses.append(np.mean(_disc_l))

            summ = self.session.run(self.performance_summaries,
                               feed_dict={self.tf_disc_cost_ph: gen_loss, self.tf_gen_cost_ph: np.mean(_disc_l)})
            summ_writer.add_summary(summ, _it)

            if (_it + 1) % 100 == 0:
                t = time.time() - starting_time
                log('{:<7}/{:<8} training iterations, took {} seconds so far...'.format(_it+1, max_iters, int(t)))
                log('gen_loss: {:.4f} disc_loss: {:.4f}'.format(gen_loss, np.mean(_disc_l)))
                log('disc_fake max:{:.4f} min:{:.4f} shape:{}'.format(disc_fake.max(), disc_fake.min(), disc_fake.shape))
                log('disc_real max:{:.4f} min:{:.4f} shape:{}'.format(disc_real.max(), disc_real.min(), disc_real.shape))

            # Evaluate the model's progress.
            if (_it+1) % eval_every == 0:

                # Update Gumbel temperature
                temperature = np.maximum(
                    self.params['temp_start'] * np.exp(-(1 - self.params['temperature_decay']) * _it),
                    self.params['min_temperature'])

                log('**** Starting Evaluation ****')

                fake_graphs = []
                fake_x_t0 = []
                real_walks = []
                real_x_t0 = []
                for q in range(n_eval_iters):
                    fake_outputs, node_logit = self.session.run([
                        sample_many, self.logit], {self.temp: 0.5})
                    fake_x, fake_t0, fake_edges, fake_t, fake_end = fake_outputs
                    smpls = None
                    stop = [False] * n_smpls
                    for i in range(n_eval_loop):
                        x, t0, e, tau, le = fake_x[i], fake_t0[i], fake_edges[i], fake_t[i], fake_end[i]

                        if q == 0 and i >= n_eval_loop-3:
                            log('eval_iters: {} eval_loop: {}'.format(q, i))
                            log('eval node logit min: {} max: {}'.format(node_logit.min(), node_logit.max()))
                            log('generated [x, t0, e, tau, end]\n[{}, {}, {}, {}, {}]'.format(
                                x[0], t0[0, 0], e[0, :], tau[0, :, 0], le[0]
                            ))
                            log('generated [x, t0, e, tau, end]\n[{}, {}, {}, {}, {}]'.format(
                                x[1], t0[1, 0], e[1, :], tau[1, :, 0], le[1]
                            ))

                        e = e.reshape(-1, self.rw_len, 2)
                        tau = tau.reshape(-1, self.rw_len, 1)
                        if i == 0:
                            smpls = np.concatenate([e, tau],axis=-1)
                        else:
                            new_pred = np.concatenate([e[:, -1:], tau[:, -1:]], axis=-1)
                            smpls = np.concatenate([smpls, new_pred], axis=1)

                        # judge if reach max length
                        for b in range(n_smpls):
                            b_le = le[b]

                            if i == 0 and b_le == 1:  # end
                                stop[b] = True
                            if i > 0 and stop[b]:  # end
                                smpls[b, -1, :] = -1
                            if i > 0 and not stop[b] and b_le == 1:
                                stop[b] = True

                    fake_x = np.array(fake_x).reshape(-1, 1)
                    fake_t0 = np.array(fake_t0).reshape(-1, 1)
                    fake_len = np.array(fake_end).reshape(-1, 1) # change to end
                    fake_start = np.c_[fake_x, fake_t0, fake_len]
                    fake_x_t0.append(fake_start)
                    fake_graphs.append(smpls)

                for _ in range(eval_transitions // self.batch_size):
                    real_x, real_t0, real_edge, real_tau, real_length \
                        = self.session.run([
                        self.real_x_input_discretes, self.real_t0_res_inputs,
                        self.real_edge_inputs_discrete, self.real_tau_inputs,
                        self.real_end_discretes
                    ], feed_dict={self.temp: 0.5})

                    walk = np.c_[real_edge.reshape(-1, 2), real_tau.reshape(-1, 1)]
                    real_walks.append(walk)
                    real_start = np.stack([real_x, real_t0[:, 0], real_length], axis=1)
                    real_x_t0.append(real_start)

                fake_graphs = np.array(fake_graphs)

                # plot edges time series for qualitative evaluation
                if is_test:
                    try:
                        fake_walks = fake_graphs.reshape(-1, 3)
                        fake_mask = fake_walks[:, 0] > -1
                        fake_walks = fake_walks[fake_mask]
                        fake_x_t0 = np.array(fake_x_t0).reshape(-1, 3)

                        real_walks = np.array(real_walks).reshape(-1, 3)
                        real_mask = real_walks[:, 0] > -1
                        real_walks = real_walks[real_mask]
                        real_x_t0 = np.array(real_x_t0).reshape(-1, 3)

                        # truth_train_walks = train_edges[:, 1:3]
                        truth_train_time = train_edges[:, 3:]
                        truth_train_res_time = self.params['t_end'] - truth_train_time
                        truth_train_walks = np.concatenate([train_edges[:, 1:3], truth_train_res_time], axis=1)
                        truth_train_x_t0 = np.c_[np.zeros((len(train_edges), 1)), truth_train_res_time]
                        truth_train_x_t0 = np.r_[truth_train_x_t0, np.ones((len(train_edges), 2))]

                        truth_test_time = test_edges[:, 3:]
                        truth_test_res_time = self.params['t_end'] - truth_test_time
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
                        fake_ax.set_xlim([0, self.N])
                        fake_ax.set_ylim([0, self.N])
                        fake_ax.set_xticks(range(self.N))
                        fake_ax.set_yticks(range(self.N))
                        fake_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        fake_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        fake_ax.set_title('fake edges number: {}'.format(len(fake_e_list)))

                        real_ax = fig.add_subplot(222, projection='3d')
                        real_ax.bar3d(real_e_list[:, 0], real_e_list[:, 1], zpos, dx, dy, real_e_counts)
                        real_ax.set_xlim([0, self.N])
                        real_ax.set_ylim([0, self.N])
                        real_ax.set_xticks(range(self.N))
                        real_ax.set_yticks(range(self.N))
                        real_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        real_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        real_ax.set_title('real edges number: {}'.format(len(real_e_list)))

                        truth_ax = fig.add_subplot(223, projection='3d')
                        truth_ax.bar3d(truth_train_e_list[:, 0], truth_train_e_list[:, 1], zpos, dx, dy,
                                       truth_train_e_counts)
                        truth_ax.set_xlim([0, self.N])
                        truth_ax.set_ylim([0, self.N])
                        truth_ax.set_xticks(range(self.N))
                        truth_ax.set_yticks(range(self.N))
                        truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        truth_ax.set_title('truth train edges number: {}'.format(len(truth_train_e_list)))

                        truth_ax = fig.add_subplot(222, projection='3d')
                        truth_ax.bar3d(truth_test_e_list[:, 0], truth_test_e_list[:, 1], zpos, dx, dy, truth_test_e_counts)
                        truth_ax.set_xlim([0, self.N])
                        truth_ax.set_ylim([0, self.N])
                        truth_ax.set_xticks(range(self.N))
                        truth_ax.set_yticks(range(self.N))
                        truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                        truth_ax.set_title('truth test edges number: {}'.format(len(truth_test_e_list)))

                        plt.tight_layout()
                        plt.savefig('{}/iter_{}_edges_counts_validation.png'.format(output_directory, _it + 1), dpi=90)
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
                        continue

                # reformat fake_graph and remove -1 value
                fake_graphs = convert_graphs(fake_graphs)

                fake_graph_file = "{}/{}_assembled_graph_iter_{}.npz".format(output_directory, timestr, _it + 1)
                fake_graphs[:, 3] = self.params['t_end'] - fake_graphs[:, 3]
                np.savez_compressed(fake_graph_file, fake_graphs=fake_graphs, real_walks=real_walks)
                fake_loss_file = "{}/{}_training_loss_iter_{}.npz".format(output_directory, timestr, _it + 1)
                np.savez_compressed(fake_loss_file, disc_losses=disc_losses, gen_losses=gen_losses)
                log('assembled graph to file: {} \nas array\n {}\n with shape: {}'.format(
                    fake_graph_file, fake_graphs[fake_graphs[:, 0] < 2], fake_graphs.shape
                ))
                save_file = "{}/{}_iter_{}.ckpt".format(save_directory, model_name, _it + 1)
                d = saver.save(self.session, save_file)
                log("**** Saving snapshots into {} ****".format(save_file))
                try:
                    Gs = Graphs(test_edges, N=self.N, tmax=self.params['t_end'], edge_contact_time=edge_contact_time)
                    FGs = Graphs(fake_graphs, N=self.N, tmax=self.params['t_end'], edge_contact_time=edge_contact_time)
                    mmd_avg_degree = MMD_Average_Degree_Distribution(Gs, FGs)
                    log('mmd_avg_degree: {}'.format(mmd_avg_degree))
                    log('Real Mean_Average_Degree_Distribution: \n{}'.format(Gs.Mean_Average_Degree_Distribution()))
                    log('Fake Mean_Average_Degree_Distribution: \n{}'.format(FGs.Mean_Average_Degree_Distribution()))
                    if early_stopping is not None:
                        if mmd_avg_degree < early_stopping:
                            log('**** end training because evaluation is reached ****')
                            break
                except:
                    print('evaluation got error! continue training ...')

                t = time.time() - starting_time
                log('**** end evaluation **** took {} seconds so far...'.format(int(t)))

            if plot_every > 0 and (_it + 1) % plot_every == 0:
                try:
                    if len(disc_losses) > 10:
                        plt.plot(disc_losses[100::10], label="Critic loss")
                        plt.plot(gen_losses[100::10], label="Generator loss")
                    else:
                        plt.plot(disc_losses, label="Critic loss")
                        plt.plot(gen_losses, label="Generator loss")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('{}/iter_{}_loss_res_final.png'.format(output_directory, _it + 1))
                    plt.close()
                except:
                    log('plotting function got error, continue training...')

        self.session.close()


        log("**** Training completed after {} iterations. ****".format(_it + 1))
        try:
            plt.plot(disc_losses[100::10], label="Critic loss")
            plt.plot(gen_losses[100::10], label="Generator loss")
            plt.legend()
            plt.savefig('{}/{}_loss_res_final.png'.format(output_directory, timestr))
            plt.close()
        except:
            log('plotting function got error, continue training...')

        # if early_stopping is None:
        #     saver.restore(self.session, save_file)
        #### Training completed.
        np.savetxt('{}/{}_iterations_{}.txt'.format(timing_directory, model_name, max_iters), time_all)
        log_dict = {"disc_losses": disc_losses, 'gen_losses': gen_losses, 'val_performances': val_performances,
                    'edge_overlaps': eo, 'generated_graphs': graphs}
        return log_dict

def make_noise(shape, type="Gaussian"):

    if type == "Gaussian":
        noise = tf.random_normal(shape)
    elif type == 'Uniform':
        noise = tf.random_uniform(shape, minval=-1, maxval=1)
    else:
        log("ERROR: Noise type {} not supported".format(type))
    return noise


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

# main run
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    from utils import *

    tf.compat.v1.reset_default_graph()

    scale = 0.1
    t_end = 1.
    lr = 0.0003
    gpu_id = 0

    # random data from metro
    file = 'data/auth_user_0.txt'
    edges = np.loadtxt(file)
    n_nodes = int(edges[:, 1:3].max() + 1)
    embedding_size = n_nodes // 2
    rw_len = 2
    batch_size = 8
    train_ratio = 0.8

    train_edges, test_edges = Split_Train_Test(edges, train_ratio)

    walker = TemporalWalker(n_nodes, train_edges, t_end,
                            scale, rw_len, batch_size,
                            init_walk_method='uniform',
                            )
    # log(walker.walk().__next__())

    tggan = TGGAN(N=n_nodes, rw_len=rw_len,
                    t_end=t_end,
                    walk_generator=walker.walk, batch_size=batch_size, gpu_id=gpu_id,
                    disc_iters=3,
                    W_down_discriminator_size=embedding_size,
                    W_down_generator_size=embedding_size,
                    l2_penalty_generator=1e-7,
                    l2_penalty_discriminator=5e-5,
                    # generator_start_layers=[40, 10],
                    generator_layers=[50, 10],
                    discriminator_layers=[40, 10],
                    # varDecoder_layers=[30],
                    temp_start=5,
                    learning_rate=lr,
                    use_wgan=True,
                    use_decoder="normal",
                    constraint_method='min_max',
                    # momentum=0.9
            )

    temperature = tggan.params['temp_start']

    tggan.session.run(tggan.init_op)

    fake_x_inputs, fake_t0_res_inputs, fake_node_inputs, fake_tau_inputs, fake_ends, \
    real_data, real_x_inputs, real_t0_res_inputs, real_node_inputs, real_tau_inputs, real_ends, \
    disc_real, disc_fake, \
    t0_res_output,tau \
        = tggan.session.run([
        tggan.fake_x_inputs, tggan.fake_t0_res_inputs,
        tggan.fake_node_inputs, tggan.fake_tau_inputs, tggan.fake_ends,
        tggan.real_data, tggan.real_x_inputs, tggan.real_t0_res_inputs,
        tggan.real_node_inputs, tggan.real_tau_inputs, tggan.real_ends,
        tggan.disc_real, tggan.disc_fake,
        tggan.t0_res_output, tggan.tau
    ], feed_dict={tggan.temp: temperature})

    tggan.session.close()

    print('real_data:\n', real_data)
    print('real_x_inputs:\n', real_x_inputs)
    print('real_t0_res_inputs:\n', real_t0_res_inputs)
    print('real_node_inputs:\n', np.argmax(real_node_inputs, axis=-1))
    print('real_tau_inputs:\n', real_tau_inputs)
    print('real_ends:\n', real_ends)

    print('fake_x_inputs:\n', fake_x_inputs)
    print('fake_t0_res_inputs:\n', fake_t0_res_inputs)
    print('fake_node_inputs:\n', np.argmax(fake_node_inputs, axis=-1))
    print('t0_res_output:\n', t0_res_output)
    print('tau:\n', tau)
    print('fake_tau_inputs:\n', fake_tau_inputs)
    print('fake_ends:\n', fake_ends)

    print('disc_real:\n', disc_real)
    print('disc_fake:\n', disc_fake)

    log('-'*40)
