"""
Implementation of the method proposed in the paper:
'Adversarial Attacks on Classification Models for Graphs'
by Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann
Published at ICML 2018 in Stockholm, Sweden.

Copyright (C) 2018
Daniel Zügner
Technical University of Munich
"""

import os
import shutil
import logging
import datetime

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

# create console handler and set level to debug
th = logging.StreamHandler()
th.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str): logger.info(str)

import tensorflow as tf
log('is GPU available? {}'.format(tf.test.is_gpu_available(cuda_only=True)))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *

class TGGAN:
    """
    NetGAN class, an implicit generative model for graphs using random walks.
    """

    def __init__(self, N, rw_len, walk_generator, t_end,
                 generator_x_up_layers=[32],
                 generator_t0_up_layers=[128],
                 generator_tau_up_layers=[64],
                 generator_time_deconv_output_depth=8,
                 generator_time_sample_num=4,
                 constraint_method='min_max',
                 generator_layers=[40],
                 discriminator_layers=[30],
                 W_down_generator_size=128, W_down_discriminator_size=128,
                 W_down_len_generator_size=3, W_down_len_discriminator_size=3,
                 batch_size=128, noise_dim=16,
                 noise_type="Gaussian", learning_rate=0.0003, disc_iters=3, wasserstein_penalty=10,
                 l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, temp_start=5.0, min_temperature=0.5,
                 temperature_decay=1 - 5e-5, seed=15, gpu_id=0,
                 use_gumbel=True, use_wgan=False, use_beta=False, use_decoder=False,
                 legacy_generator=False):
        """
        Initialize NetGAN.

        Parameters
        ----------
        N: int
           Number of nodes in the graph to generate.
        rw_len: int
                Length of random walks to generate.
        walk_generator: function
                        Function that generates a single random walk and takes no arguments.
        generator_layers: list of integers, default: [40], i.e. a single layer with 40 units.
                          The layer sizes of the generator LSTM layers
        discriminator_layers: list of integers, default: [30], i.e. a single layer with 30 units.
                              The sizes of the discriminator LSTM layers
        W_down_generator_size: int, default: 128
                               The size of the weight matrix W_down of the generator. See our paper for details.
        W_down_discriminator_size: int, default: 128
                                   The size of the weight matrix W_down of the discriminator. See our paper for details.
        batch_size: int, default: 128
                    The batch size.
        noise_dim: int, default: 16
                   The dimension of the random noise that is used as input to the generator.
        noise_type: str in ["Gaussian", "Uniform], default: "Gaussian"
                    The noise type to feed into the generator.
        learning_rate: float, default: 0.0003
                       The learning rate.
        disc_iters: int, default: 3
                    The number of discriminator iterations per generator training iteration.
        wasserstein_penalty: float, default: 10
                             The Wasserstein gradient penalty applied to the discriminator. See the Wasserstein GAN
                             paper for details.
        l2_penalty_generator: float, default: 1e-7
                                L2 penalty on the generator weights.
        l2_penalty_discriminator: float, default: 5e-5
                                    L2 penalty on the discriminator weights.
        temp_start: float, default: 5.0
                    The initial temperature for the Gumbel softmax.
        min_temperature: float, default: 0.5
                         The minimal temperature for the Gumbel softmax.
        temperature_decay: float, default: 1-5e-5
                           After each evaluation, the current temperature is updated as
                           current_temp := max(temperature_decay*current_temp, min_temperature)
        seed: int, default: 15
              Random seed.
        gpu_id: int or None, default: 0
                The ID of the GPU to be used for training. If None, CPU only.
        use_gumbel: bool, default: True
                Use the Gumbel softmax trick.

        legacy_generator: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks.
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.

        """

        self.params = {
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            't_end': t_end,
            'generator_x_up_layers': generator_x_up_layers,
            'generator_t0_up_layers': generator_t0_up_layers,
            'generator_tau_up_layers': generator_tau_up_layers,
            'generator_time_deconv_output_depth': generator_time_deconv_output_depth,
            'generator_time_sample_num': generator_time_sample_num,
            'constraint_method': constraint_method,
            'Generator_Layers': generator_layers,
            'Discriminator_Layers': discriminator_layers,
            'W_down_len_generator_size': W_down_len_generator_size,
            'W_down_len_discriminator_size': W_down_len_discriminator_size,
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
            'use_beta': use_beta,
            'use_wgan': use_wgan,
            'legacy_generator': legacy_generator
        }

        # assert rw_len > 1, "Random walk length must be > 1."

        tf.set_random_seed(seed)

        self.N = N
        self.rw_len = rw_len
        self.n_length = rw_len + 1 # allow zero length value
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
        self.start_x_0 = None

        self.W_down_x_generator = tf.get_variable(name='Generator.W_Down_x', dtype=tf.float32,
                                                  shape=[2, 1],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W_down_x_discriminator = tf.get_variable(name='Discriminator.W_Down_x', dtype=tf.float32,
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
                                         initializer=tf.zeros_initializer)
        self.W_down_tau = tf.get_variable('Generator.W_down_tau', dtype=tf.float32,
                                         shape=[self.G_t_deconv_output_depth, 1],
                                         # constraint=lambda w: tf.clip_by_value(w, 1e-6, 1./self.G_t_deconv_output_depth),
                                         # initializer=tf.random_uniform_initializer(1e-2, 1e-1),
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                          )
        self.W_down_tau_bias = tf.get_variable('Generator.W_down_tau_bias', dtype=tf.float32,
                                         shape=1,
                                         constraint=lambda w: tf.clip_by_value(w, 0, 1.),
                                         initializer=tf.zeros_initializer)

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

        self.W_down_len_generator = tf.get_variable('Generator.W_Down_length',
                                                    shape=[self.n_length, self.params['W_down_len_generator_size']],
                                                    dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())

        self.W_down_len_discriminator = tf.get_variable('Discriminator.W_Down_length',
                                                        shape=[self.n_length, self.params['W_down_len_discriminator_size']],
                                                        dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer())

        self.generator_function = self.generator_recurrent

        self.fake_x_inputs, self.fake_t0_res_inputs, self.fake_node_inputs, self.fake_tau_inputs, self.fake_lengths \
            = self.generator_function(self.batch_size, reuse=False, gumbel=use_gumbel, legacy=legacy_generator)
        self.fake_length_discretes = tf.argmax(self.fake_lengths, axis=-1)

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
        self.real_tau_inputs = self.real_data[:, 1:, 2:]

        self.real_x_input_discretes = tf.cast(self.real_data[:, 0, 0], dtype=tf.int64)
        self.real_x_inputs = tf.one_hot(self.real_x_input_discretes, 2)
        self.real_length_discretes = tf.cast(self.real_data[:, 0, 1], dtype=tf.int64)
        self.real_lengths = tf.one_hot(self.real_length_discretes, self.n_length)
        self.real_t0_res_inputs = self.real_data[:, 0:1, 2]

        self.discriminator_function = self.discriminator_recurrent
        self.real_lstm_lengths = self.real_length_discretes + 3
        self.disc_real = self.discriminator_function(x=self.real_x_inputs,
                                                     t0_res=self.real_t0_res_inputs,
                                                     node_inputs=self.real_node_inputs,
                                                     tau_inputs=self.real_tau_inputs,
                                                     length_inputs=self.real_lengths,
                                                     length_discrete=self.real_length_discretes)
        self.fake_lstm_lengths = self.fake_length_discretes + 3
        self.disc_fake = self.discriminator_function(x=self.fake_x_inputs,
                                                     t0_res=self.fake_t0_res_inputs,
                                                     node_inputs=self.fake_node_inputs,
                                                     tau_inputs=self.fake_tau_inputs,
                                                     length_inputs=self.fake_lengths,
                                                     length_discrete=self.fake_lstm_lengths,
                                                     reuse=True)

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        if use_wgan:
            with tf.name_scope('WGAN_LOSS'):
                # WGAN lipschitz-penalty
                alpha_x = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0.,maxval=1.)
                self.differences_x = self.fake_x_inputs - self.real_x_inputs
                self.interpolates_x = self.real_x_inputs + (alpha_x * self.differences_x)

                alpha_len = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0.,maxval=1.)
                self.differences_len = self.fake_lengths - self.real_lengths
                self.interpolates_len = self.real_lengths + (alpha_len * self.differences_len)

                alpha_t0 = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0.,maxval=1.)
                self.differences_t0 = self.fake_t0_res_inputs - self.real_t0_res_inputs
                self.interpolates_t0 = self.real_t0_res_inputs + (alpha_t0 * self.differences_t0)

                alpha_node = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0., maxval=1.)
                self.differences_node = self.fake_node_inputs - self.real_node_inputs
                self.interpolates_node = self.real_node_inputs + (alpha_node * self.differences_node)

                alpha_tau = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0., maxval=1.)
                self.differences_tau = self.fake_tau_inputs - self.real_tau_inputs
                self.interpolates_tau = self.fake_tau_inputs + (alpha_tau * self.differences_tau)

                self.gradients_x, self.gradients_t0, self.gradients_node, self.gradients_tau, \
                self.gradients_len = tf.gradients(
                    self.discriminator_function(
                        x=self.interpolates_x,
                        t0_res=self.interpolates_t0,
                        node_inputs=self.interpolates_node,
                        tau_inputs=self.interpolates_tau,
                        length_inputs=self.interpolates_len,
                        reuse=True), [
                        self.interpolates_x, self.interpolates_t0, self.interpolates_node,
                        self.interpolates_tau, self.interpolates_len
                    ])
                self.slopes = tf.sqrt(
                    tf.reduce_sum(tf.stack([
                        tf.reduce_sum(tf.square(self.gradients_x), reduction_indices=[1]),
                        tf.reduce_sum(tf.square(self.gradients_t0), reduction_indices=[1]),
                        tf.reduce_sum(tf.square(self.gradients_len), reduction_indices=[1]),
                        tf.reduce_sum(tf.square(self.gradients_node), reduction_indices=[1, 2]),
                        tf.reduce_sum(tf.square(self.gradients_tau), reduction_indices=[1, 2]),
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

    def discriminator_recurrent(self, x, t0_res, node_inputs, tau_inputs, length_inputs, length_discrete=None, reuse=None):
        """
        Discriminate real from fake random walks using LSTM.
        Parameters
        ----------
        inputs: tf.tensor, shape (None, rw_len, N)
                The inputs to process
        reuse: bool, default: None
               If True, discriminator variables will be reused.

        Returns
        -------
        final_score: tf.tensor, shape [None,], i.e. a scalar
                     A score measuring how "real" the input random walks are perceived.

        """

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

            with tf.name_scope('DISC_LENGTH'):
                length_inputs_reshape = tf.matmul(length_inputs, self.W_down_len_discriminator)
                # for ix, size in enumerate(self.D_layers):
                #     length_inputs = tf.layers.dense(length_inputs, size, name="Discriminator.length_{}".format(ix),
                #                                     reuse=reuse, activation=tf.nn.tanh,
                #                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                length_inputs_up = tf.layers.dense(length_inputs_reshape, int(self.W_down_discriminator.shape[-1]),
                                                  reuse=reuse, name="Discriminator.length_up_scale", activation=tf.nn.tanh,
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

            inputs = [x_input_reshape] + [t0_input_up] + [length_inputs_up]

            for i in range(self.rw_len):
                inputs += [node_output[i*2]] + [node_output[i*2+1]] + [tau_output[i]]

            with tf.name_scope('DISC_LSTM'):
                def lstm_cell(lstm_size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

                disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

                if length_discrete is not None:
                    output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell,
                                                                        inputs=inputs,
                                                                        dtype='float32',
                                                                        sequence_length=length_discrete,
                                                                        )
                else:
                    output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell,
                                                                        inputs=inputs,
                                                                        dtype='float32',
                                                                        )

            last_output = output_disc[-1]

            final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
            return final_score

    def generate_discrete(self, n_samples, n_eval_loop, reuse=True, z=None, gumbel=True, legacy=False):
        """
        Generate a random walk in index representation (instead of one hot). This is faster but prevents the gradients
        from flowing into the generator, so we only use it for evaluation purposes.

        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks.
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.

        Returns
        -------
                The generated random walks, shape [None, rw_len, N]


        """
        if self.start_x_0 is None:
            self.start_x_0 = tf.one_hot(tf.zeros(dtype=tf.int64, shape=[n_samples, ]), depth=2, name="start_x_0")
            self.start_x_1 = tf.one_hot(tf.ones(dtype=tf.int64, shape=[n_samples, ]), depth=2, name="start_x_1")
            self.start_t0 = tf.ones(dtype=tf.float32, shape=[n_samples, 1], name='start_t0')

        fake_x, fake_t0, fake_e, fake_tau, fake_len = [], [], [], [], []
        for i in range(n_eval_loop):
            if i == 0:
                fake_x_output, fake_t0_res_output, \
                fake_node_outputs, fake_tau_outputs, fake_length_outputs = self.generator_function(
                    n_samples, reuse, z, x_input=self.start_x_1, t0_input=self.start_t0,
                    gumbel=gumbel, legacy=legacy)
            else:
                t0_input = fake_tau_outputs[:, -2, :1] # second from end time is correct residual
                tau_input = fake_tau_outputs[:, -1, :]
                edge_input = fake_node_outputs[:, -2:, :] # first from end node is correct

                fake_x_output, fake_t0_res_output, \
                fake_node_outputs, fake_tau_outputs, fake_length_outputs = self.generator_function(
                    n_samples, reuse, z,
                    x_input=self.start_x_0, t0_input=t0_input, edge_input=edge_input, tau_input=tau_input,
                    gumbel=gumbel, legacy=legacy)

            fake_x_outputs_discrete = tf.argmax(fake_x_output, axis=-1)
            fake_node_outputs_discrete = tf.argmax(fake_node_outputs, axis=-1)
            # notice lengths >= 1, zero is not valid
            fake_length_outputs_discrete = tf.argmax(fake_length_outputs, axis=-1)

            fake_x.append(fake_x_outputs_discrete)
            fake_t0.append(fake_t0_res_output)
            fake_e.append(fake_node_outputs_discrete)
            fake_tau.append(fake_tau_outputs)
            fake_len.append(fake_length_outputs_discrete)

        return fake_x, fake_t0, fake_e, fake_tau, fake_len

    def generator_recurrent(self, n_samples, reuse=None, z=None,
                            x_input=None, t0_input=None, edge_input=None, tau_input=None, length_input=None,
                            gumbel=True, legacy=False):
        """
        Generate random walks using LSTM.
        Parameters
        ----------
        n_samples: int
                   The number of random walks to generate.
        reuse: bool, default: None
               If True, generator variables will be reused.
        z: None or tensor of shape (n_samples, noise_dim)
           The input noise. None means that the default noise generation function will be used.
        gumbel: bool, default: False
            Whether to use the gumbel softmax for generating discrete output.
        legacy: bool, default: False
            If True, the hidden and cell states of the generator LSTM are initialized by two separate feed-forward networks.
            If False (recommended), the hidden layer is shared, which has less parameters and performs just as good.
        Returns
        -------
        The generated random walks, shape [None, rw_len, N]

        """

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
                inputs = tf.zeros([n_samples, self.params['W_Down_Generator_size']])

            # LSTM steps
            node_outputs = []
            tau_outputs = []

            def lstm_cell(lstm_size, name):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse,
                                                    name="LSTM_{}".format(name))


            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size, "walks") for size in self.G_layers])
            # self.stacked_lstm_x = tf.contrib.rnn.MultiRNNCell([lstm_cell(size, "start") for size in self.G_layers])

            # LSTM tine steps
            for i in range(self.rw_len + 1):
                # generate the first three start elements: start x, residual time, and maximum possible length
                if i == 0:
                    with tf.variable_scope('LSTM_CELL'):
                        output, state = self.stacked_lstm.call(inputs, state)

                    with tf.name_scope('GEN_START_X'):
                        # generate start x, and its residual time
                        if x_input is not None:
                            x_output = x_input
                        else:
                            # generate start node binary if not need
                            x_logit = output
                            for ix, size in enumerate(self.G_x_up_layers):
                                x_logit = tf.layers.dense(x_logit, size, name="Generator.x_logit_{}".format(ix),
                                                          reuse=reuse, activation=tf.nn.tanh)
                            x_logit = tf.layers.dense(x_logit, 2, name="Generator.x_logit_last",
                                                      reuse=reuse, activation=None)

                            # Perform Gumbel softmax to ensure gradients flow for e, and end node y
                            if gumbel: x_output = gumbel_softmax(x_logit, temperature=self.temp, hard=True)
                            else:      x_output = tf.nn.softmax(x_logit)

                        x_down = tf.matmul(x_output, self.W_down_x_generator)
                        # convert to input
                        inputs = tf.layers.dense(x_down, self.params['W_Down_Discriminator_size'],
                                                 name="Generator.x_lstm_input",
                                                 reuse=reuse, activation=tf.nn.tanh)

                    output, state = self.stacked_lstm.call(inputs, state)
                    with tf.name_scope('GEN_START_TIME'):
                        if t0_input is not None: # for evaluation generation
                            t0_res_output = t0_input
                        else:
                            t0_res_output = self.generate_time_t0(output)
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

                    output, state = self.stacked_lstm.call(inputs, state)
                    with tf.name_scope('GEN_MAX_LENGTH'):
                        if length_input is not None:
                            max_lengths = length_input
                        else:
                            # generate start node binary if not need
                            len_logit = output
                            for ix, size in enumerate(self.G_x_up_layers):
                                len_logit = tf.layers.dense(len_logit, size, name="Generator.len_logit_{}".format(ix),
                                                          reuse=reuse, activation=tf.nn.tanh)
                            len_logit = tf.layers.dense(len_logit, self.n_length, name="Generator.len_logit_last",
                                                      reuse=reuse, activation=None)

                            # Perform Gumbel softmax to ensure gradients flow for e, and end node y
                            if gumbel:
                                max_lengths = gumbel_softmax(len_logit, temperature=self.temp, hard=True)
                            else:
                                max_lengths = tf.nn.softmax(len_logit)

                        len_down = tf.matmul(max_lengths, self.W_down_len_generator)
                        # convert to input
                        inputs = tf.layers.dense(len_down, self.params['W_Down_Discriminator_size'],
                                                 name="Generator.len_lstm_input",
                                                 reuse=reuse, activation=tf.nn.tanh)

                # generate temporal edge part
                else:
                    for j in range(2):
                        # LSTM for first node
                        output, state = self.stacked_lstm.call(inputs, state)

                        with tf.variable_scope('GEN_NODE'):
                            if i > 1: tf.get_variable_scope().reuse_variables()

                            if edge_input is not None and i == 1:  # for evaluation generation
                                output = edge_input[:, j]
                            else:
                                # Blow up to dimension N using W_up
                                logit = tf.matmul(output, self.W_up) + self.b_W_up

                                # Perform Gumbel softmax to ensure gradients flow
                                if gumbel: output = gumbel_softmax(logit, temperature=self.temp, hard=True)
                                else:      output = tf.nn.softmax(logit)

                            node_outputs.append(output)

                            # Back to dimension d
                            inputs = tf.matmul(output, self.W_down_generator)

                    # LSTM for tau
                    output, state = self.stacked_lstm.call(inputs, state)

                    with tf.variable_scope('GEN_TAU_TIME'):
                        if i > 1: tf.get_variable_scope().reuse_variables()

                        if tau_input is not None and i == 1:  # for evaluation generation
                            tau = tau_input
                        else:
                            tau = self.generate_time_tau(output)
                            tau = self.time_constraint(tau, method=self.params['constraint_method']) * res_time
                            self.tau = tau

                        res_time = tau
                        # res_time = tf.stop_gradient(res_time)

                        # save outputs
                        tau_outputs.append(tau)

                        # convert to input
                        inputs = tf.layers.dense(tau, int(self.W_down_generator.shape[-1]),
                                                 name="Generator.tau_input", activation=tf.nn.tanh)

            node_outputs = tf.stack(node_outputs, axis=1)
            tau_outputs = tf.stack(tau_outputs, axis=1)

            # print('x_output', x_output.shape)
            # print('t0_res_output', t0_res_output.shape)
            # print('node_outputs', node_outputs.shape)
            # print('tau_outputs', tau_outputs.shape)

        return x_output, t0_res_output, node_outputs, tau_outputs, max_lengths

    def time_constraint(self, t, method='min_max'):
        with tf.name_scope('time_constraint'):
            if method == 'relu':
                t = tf.nn.relu(t) - tf.nn.relu(t - 1.)
            elif method == 'l2_norm':
                t = (tf.nn.l2_normalize(t, axis=0) + 1.) / 2.
            elif method == 'min_max':
                min_ = tf.math.reduce_min(t, axis=0)[0]
                t = tf.case([
                    (tf.math.less(min_, 0.), lambda : t - min_)
                ], lambda : t)

                max_ = tf.math.reduce_max(t, axis=0)[0]
                t = tf.case([
                    (tf.math.less(1., max_), lambda: t / max_)
                ], lambda : t)

        return t

    def generate_time_t0(self, output):
        n_samples = int(output.shape[0])
        if self.params['use_decoder']:
            with tf.name_scope('t0_repara_decoder'):
                loc_t0 = output
                scale_t0 = output
                for ix, size in enumerate(self.G_t0_up_layers):
                    loc_t0 = tf.layers.dense(loc_t0, size,
                                             name="Generator.loc_t0_{}".format(ix),
                                             activation=tf.nn.tanh)
                    scale_t0 = tf.layers.dense(scale_t0, size,
                                               name="Generator.scale_t0_{}".format(ix),
                                               activation=tf.nn.tanh)
                loc_t0 = tf.layers.dense(loc_t0, 1, name="Generator.loc_t0_last",
                                         activation=None)
                scale_t0 = tf.layers.dense(scale_t0, 1, name="Generator.scale_t0_last",
                                           activation=None)
                # loc_t0 = tf.clip_by_value(loc_t0, 0, 1)
                # scale_t0 = tf.clip_by_value(scale_t0, 0, 1)
                if not self.params['use_beta']:
                    t0_wait = [tf.truncated_normal([1], mean=loc_t0[i, 0], stddev=scale_t0[i, 0])
                               for i in range(n_samples)]
                else:
                    t0_wait = self.beta_decoder(_alpha_param=loc_t0, _beta_param=scale_t0)

                t0_wait = tf.stack(t0_wait, axis=0)
        else:
            with tf.name_scope('t0_deep_decoder'):
                t0_wait = output
                for ix, size in enumerate(self.G_t0_up_layers):
                    t0_wait = tf.layers.dense(t0_wait, size,
                                              name="Generator.t0_up_{}".format(ix),
                                              activation=tf.nn.tanh)
                t0_wait = tf.expand_dims(t0_wait, axis=2)

                # deconvolutional
                n_strides = 2
                t0_wait = tf.nn.conv1d_transpose(
                    t0_wait, filters=self.t0_deconv_filter,
                    output_shape=[n_samples, int(t0_wait.shape[1]) * n_strides,
                                  self.G_t_deconv_output_depth],
                    strides=n_strides, padding='SAME',
                    name='Generator.t0_deconv')

                choice = tf.random_uniform([self.G_t_sample_n], maxval=t0_wait.shape[1],
                                           dtype=tf.int64)
                t0_wait = tf.gather(t0_wait, choice, axis=1)
                t0_wait = tf.reduce_mean(t0_wait, axis=1)
                t0_wait = tf.layers.dense(t0_wait, 1, name="Generator.t0_deconv_last",
                                          activation=None)

        return t0_wait

    def generate_time_tau(self, output):
        n_samples = int(output.shape[0])
        if self.params['use_decoder']:
            loc = output
            scale = output
            for ix, size in enumerate(self.G_tau_up_layers):
                loc = tf.layers.dense(loc, size, name="Generator.loc_tau_{}".format(ix),
                                      activation=tf.nn.tanh,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
                scale = tf.layers.dense(scale, size, name="Generator.scale_tau_{}".format(ix),
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            loc = tf.layers.dense(loc, 1, name="Generator.loc_tau_last", activation=None)
            scale = tf.layers.dense(scale, 1, name="Generator.scale_tau_last", activation=None)

            if not self.params['use_beta']:
                tau = [tf.truncated_normal(
                    [1], mean=loc[i, 0], stddev=scale[i, 0]) for i in range(n_samples)]
                tau = tf.stack(tau, axis=0)
            else:
                tau = self.beta_decoder(_alpha_param=loc, _beta_param=scale)
        else:
            tau = output
            for ix, size in enumerate(self.G_tau_up_layers):
                tau = tf.layers.dense(tau, size, name="Generator.tau_up_{}".format(ix),
                                      activation=tf.nn.tanh,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            tau = tf.expand_dims(tau, axis=2)

            # deconvolutional
            n_strides = 2
            tau = tf.nn.conv1d_transpose(tau, filters=self.tau_deconv_filter,
                                         output_shape=[n_samples,
                                                       self.G_tau_up_layers[-1] * n_strides,
                                                       self.G_t_deconv_output_depth],
                                         strides=n_strides, padding='SAME',
                                         name='Generator.tau_deconv')

            choice = tf.random_uniform([self.G_t_sample_n], maxval=tau.shape[1],
                                       dtype=tf.int64)
            tau = tf.gather(tau, choice, axis=1)
            tau = tf.reduce_mean(tau, axis=1)
            tau = tf.layers.dense(tau, 1, name="Generator.tau_deconv_last",
                                  activation=None)
        return tau

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

    def train(self, train_edges, test_edges, max_iters=1000, stopping=None,
              eval_transitions=1e6, n_eval_loop=3,
              max_patience=5, eval_every=500, plot_every=1000,
              output_directory='outputs', save_directory="snapshots",
              model_name=None, continue_training=False):
        """

        Parameters
        ----------
        A_orig: sparse matrix, shape: (N,N)
                Adjacency matrix of the original graph to be trained on.
        val_ones: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation edges
        val_zeros: np.array, shape (n_val, 2)
                  The indices of the hold-out set of validation non-edges
        max_iters: int, default: 50,000
                   The maximum number of training iterations if early stopping does not apply.
        stopping: float in (0,1] or None, default: None
                  The early stopping strategy. None means VAL criterion will be used (i.e. evaluation on the
                  validation set and stopping after there has not been an improvement for *max_patience* steps.
                  Set to a value in the interval (0,1] to stop when the edge overlap exceeds this threshold.
        eval_transitions: int, default: 15e6
                          The number of transitions that will be used for evaluating the validation performance, e.g.
                          if the random walk length is 5, each random walk contains 4 transitions.
        transitions_per_iter: int, default: 150000
                              The number of transitions that will be generated in one batch. Higher means faster
                              generation, but more RAM usage.
        max_patience: int, default: 5
                      Maximum evaluation steps without improvement of the validation accuracy to tolerate. Only
                      applies to the VAL criterion.
        eval_every: int, default: 500
                    Evaluate the model every X iterations.
        plot_every: int, default: -1
                    Plot the generator/discriminator losses every X iterations. Set to None or a negative number
                           to disable plotting.
        save_directory: str, default: "../snapshots"
                        The directory to save model snapshots to.
        model_name: str, default: None
                    Name of the model (will be used for saving the snapshots).
        continue_training: bool, default: False
                           Whether to start training without initializing the weights first. If False, weights will be
                           initialized.

        Returns
        -------
        log_dict: dict
                  A dictionary with the following values observed during training:
                  * The generator and discriminator losses
                  * The validation performances (ROC and AP)
                  * The edge overlap values between the generated and original graph
                  * The sampled graphs for all evaluation steps.

        """

        starting_time = time.time()
        saver = tf.train.Saver()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        tensorboard_log_path = './graph'
        model_number = 0

        if stopping == None:  # use VAL criterion
            best_performance = 0.0
            patience = max_patience
            log("**** Using VAL criterion for early stopping ****")

        else:  # use EO criterion
            assert "float" in str(type(stopping)) and stopping > 0 and stopping <= 1
            log("**** Using EO criterion of {} for early stopping".format(stopping))

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
        p = 100
        n_eval_iters = int(eval_transitions / self.batch_size / p)
        n_samples = self.batch_size * p
        sample_many = self.generate_discrete(n_samples, n_eval_loop=n_eval_loop, reuse=True)

        log("**** Starting training. ****")

        for _it in range(max_iters):

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

            gen_losses.append(gen_loss)
            disc_losses.append(np.mean(_disc_l))

            summ = self.session.run(self.performance_summaries,
                               feed_dict={self.tf_disc_cost_ph: gen_loss, self.tf_gen_cost_ph: np.mean(_disc_l)})
            summ_writer.add_summary(summ, _it)

            if (_it + 1) % 1000 == 0:
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

                fake_walks, fake_x_t0, real_walks, real_x_t0 = self.generate_samples(n_eval_iters, n_samples, sample_many)

                try:
                    real_walks, fake_graphs = self.my_eval(fake_walks, fake_x_t0, real_walks, real_x_t0,
                                                           output_directory, _it)

                    fake_graph_file = "{}/{}_assembled_graph_iter_{}.npz".format(output_directory, timestr, _it + 1)
                    np.savez_compressed(fake_graph_file, fake_graphs=fake_graphs, real_walks=real_walks)
                    fake_loss_file = "{}/{}_training_loss_iter_{}.npz".format(output_directory, timestr, _it + 1)
                    np.savez_compressed(fake_loss_file, disc_losses=disc_losses, gen_losses=gen_losses)
                    log('assembled graph to file: {} \nas array\n {}\n with shape: {}'.format(
                        fake_graph_file, fake_graphs[:2, :], fake_graphs.shape))
                    save_file = "{}/{}_iter_{}.ckpt".format(save_directory, model_name, _it + 1)
                    d = saver.save(self.session, save_file)
                    log("**** Saving snapshots into {} ****".format(save_file))
                except:
                    log('**** plotting function got error, continue training...')
                    t = time.time() - starting_time
                    log('**** end evaluation **** took {} seconds so far...'.format(int(t)))

            if plot_every > 0 and (_it + 1) % plot_every == 0:
                try:
                    if len(disc_losses) > 10:
                        plt.plot(disc_losses[10::10], label="Critic loss")
                        plt.plot(gen_losses[10::10], label="Generator loss")
                    else:
                        plt.plot(disc_losses, label="Critic loss")
                        plt.plot(gen_losses, label="Generator loss")
                    plt.legend()
                    plt.savefig('{}/iter_{}_loss_res_final.png'.format(output_directory, _it + 1))
                    plt.close()
                except:
                    log('plotting function got error, continue training...')

        self.session.close()

        log("**** Training completed after {} iterations. ****".format(_it + 1))
        try:
            plt.plot(disc_losses[10::10], label="Critic loss")
            plt.plot(gen_losses[10::10], label="Generator loss")
            plt.legend()
            plt.savefig('{}/{}_loss_res_final.png'.format(output_directory, timestr))
            plt.close()
        except:
            log('plotting function got error, continue training...')

        # if stopping is None:
        #     saver.restore(self.session, save_file)
        #### Training completed.
        log_dict = {"disc_losses": disc_losses, 'gen_losses': gen_losses, 'val_performances': val_performances,
                    'edge_overlaps': eo, 'generated_graphs': graphs}
        return log_dict

    def generate_samples(self, n_eval_iters, n_samples, sample_many):
        fake_walks = []
        fake_x_t0 = []
        real_walks = []
        real_x_t0 = []
        for _ in range(n_eval_iters):
            fake_x, fake_t0, fake_edges, fake_t, fake_length = self.session.run(sample_many, {self.temp: 0.5})
            # fake_x = np.ones((self.batch_size, 1))
            # fake_t0 = np.zeros((self.batch_size, 1))
            # fake_t = np.zeros((self.batch_size, self.rw_len, 1))
            # fake_x, fake_t0, fake_v, fake_u, fake_t = smpls[0]
            # # print('fake_t:\n {} \nfake_v: \n{} \nfake_u: \n{}'.format(fake_t, fake_v, fake_u))
            smpls = None
            stop = [False] * n_samples
            for i in range(n_eval_loop):
                x, t0, e, tau, le = fake_x[i], fake_t0[i], fake_edges[i], fake_t[i], fake_length[i]
                for j in range(self.rw_len):
                    if i == 0 and j == 0:
                        smpls = np.c_[e[:, j * 2:(j + 1) * 2], t0[:, :1]]
                    if i == 0 and j > 0:
                        smpls = np.c_[smpls, e[:, j * 2: (j + 1) * 2], t0[:, :1]]
                    if i > 0 and j > 0:  # ignore the first edge since it repeats last eval_loop
                        smpls = np.c_[smpls, e[:, j * 2: (j + 1) * 2], t0[:, :1]]
                # judge if reach max length
                for b in range(self.batch_size):
                    b_le = le[b]
                    if i == 0 and b_le < self.rw_len:  # end
                        smpls[b, (i * self.rw_len + b_le) * 3:] = -1
                        stop[b] = True

                    start = i * self.rw_len - i + 1
                    if i > 0 and not stop[b] and b_le <= 1:  # end
                        smpls[b, start * 3: (start + self.rw_len - 1) * 3] = -1
                        stop[b] = True
                    if i > 0 and not stop[b] and b_le > 1 and b_le < self.rw_len:
                        smpls[b, (start + b_le) * 3:] = -1
                        stop[b] = True
                    if i > 0 and stop[b]:
                        smpls[b, start * 3: (start + self.rw_len - 1) * 3] = -1

            fake_x = np.array(fake_x).reshape(-1, 1)
            fake_t0 = np.array(fake_t0).reshape(-1, 1)
            fake_start = np.c_[fake_x, fake_t0]
            fake_x_t0.append(fake_start)
            fake_walks.append(smpls)

            real_x, real_t0, real_edge, real_tau, \
            walk = self.session.run([
                self.real_x_input_discretes, self.real_t0_res_inputs,
                self.real_edge_inputs_discrete, self.real_tau_inputs,
                self.real_edge_inputs_discrete
            ], feed_dict={self.temp: 0.5})

            walk = np.c_[real_edge, real_tau]
            real_walks.append(walk)
            real_start = np.stack([real_x, real_t0[:, 0]], axis=1)
            real_x_t0.append(real_start)
        return fake_walks, fake_x_t0, real_walks, real_x_t0

    def my_eval(self, fake_walks, fake_x_t0, real_walks, real_x_t0, output_directory, _it):

        seq_len = 3 * (self.rw_len * n_eval_loop - n_eval_loop + 1)
        fake_graphs = np.array(fake_walks).reshape(-1, seq_len)

        fake_walks = fake_graphs.reshape(-1, 3)
        fake_mask = fake_walks[:, 0] > -1
        fake_walks = fake_walks[fake_mask]
        fake_x_t0 = np.array(fake_x_t0).reshape(-1, 2)

        real_walks = np.array(real_walks).reshape(-1, 3)
        real_mask = real_walks[:, 0] > -1
        real_walks = real_walks[real_mask]
        real_x_t0 = np.array(real_x_t0).reshape(-1, 2)

        # truth_train_walks = train_edges[:, 1:3]
        truth_train_time = train_edges[:, 3:]
        truth_train_res_time = self.params['t_end'] - truth_train_time
        truth_train_walks = np.concatenate([train_edges[:, 1:3], truth_train_res_time], axis=1)
        truth_train_x_t0 = np.array(real_x_t0).reshape(-1, 2)

        truth_test_time = test_edges[:, [3]]
        truth_test_res_time = self.params['t_end'] - truth_test_time
        truth_test_walks = np.c_[test_edges[:, 1:3], truth_test_res_time]
        truth_test_x_t0 = np.array(real_x_t0).reshape(-1, 2)
        # print('fake_walks: \n{} \nreal_walks: \n{}'.format(fake_walks, real_walks))

        # plot edges time series for qualitative evaluation
        print(fake_walks)
        fake_e_list, fake_e_counts = np.unique(fake_walks[:, 0:2], return_counts=True, axis=0)
        real_e_list, real_e_counts = np.unique(real_walks[:, 0:2], return_counts=True, axis=0)
        truth_train_e_list, truth_train_e_counts = np.unique(truth_train_walks[:, 0:2], return_counts=True, axis=0)
        truth_test_e_list, truth_test_e_counts = np.unique(truth_test_walks[:, 0:2], return_counts=True, axis=0)
        truth_e_list, truth_e_counts = np.unique(
            np.r_[truth_test_walks[:, 0:2], truth_test_walks[:, 0:2]], return_counts=True, axis=0)
        n_e = len(truth_e_list)

        real_x_list, real_x_counts = np.unique(real_x_t0[:, 0], return_counts=True)
        fake_x_list, fake_x_counts = np.unique(fake_x_t0[:, 0], return_counts=True)
        truth_train_x_list, truth_train_x_counts = real_x_list, real_x_counts
        truth_test_x_list, truth_test_x_counts = real_x_list, real_x_counts

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
        truth_ax.bar3d(truth_train_e_list[:, 0], truth_train_e_list[:, 1], zpos, dx, dy, truth_train_e_counts)
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

        fig, ax = plt.subplots(n_e + 3, 4, figsize=(4 * 6, (n_e + 3) * 4))
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
        truth_ax.bar(truth_train_x_list, truth_train_x_counts)
        truth_ax.set_xlim([-1, 2])
        truth_ax.set_title('truth start x number: {}'.format(len(truth_train_x_list)))
        truth_ax = ax[i, 3]
        truth_ax.bar(truth_train_x_list, truth_train_x_counts)
        truth_ax.set_xlim([-1, 2])
        truth_ax.set_title('truth start x number: {}'.format(len(truth_train_x_list)))

        i = 1
        for j, e in enumerate([0, 1]):
            real_ax = ax[i + j, 0]
            real_mask = real_x_t0[:, 0] == e
            real_times = real_x_t0[real_mask][:, 1]
            real_ax.hist(real_times, range=[0, 1], bins=100)
            real_ax.set_title('real x node: {} loc: {:.4f} scale: {:.4f}'.format(
                int(e), real_times.mean(), real_times.std()))

            fake_ax = ax[i + j, 1]
            fake_mask = fake_x_t0[:, 0] == e
            fake_times = fake_x_t0[fake_mask][:, 1]
            fake_ax.hist(fake_times, range=[0, 1], bins=100)
            fake_ax.set_title('fake x node: {} loc: {:.4f} scale: {:.4f}'.format(
                int(e), fake_times.mean(), fake_times.std()))

            truth_ax = ax[i + j, 2]
            truth_train_mask = truth_train_x_t0[:, 0] == e
            truth_train_times = truth_train_x_t0[truth_train_mask][:, 1]
            truth_ax.hist(truth_train_times, range=[0, 1], bins=100)
            truth_ax.set_title('truth x node: {} loc: {:.4f} scale: {:.4f}'.format(
                int(e), truth_train_times.mean(), truth_train_times.std()))
            truth_ax = ax[i + j, 3]
            truth_train_mask = truth_train_x_t0[:, 0] == e
            truth_train_times = truth_train_x_t0[truth_train_mask][:, 1]
            truth_ax.hist(truth_train_times, range=[0, 1], bins=100)
            truth_ax.set_title('truth x node: {} loc: {:.4f} scale: {:.4f}'.format(
                int(e), truth_train_times.mean(), truth_train_times.std()))

        i = 3
        for j, e in enumerate(truth_e_list):
            real_ax = ax[i + j, 0]
            real_mask = np.logical_and(real_walks[:, 0] == e[0], real_walks[:, 1] == e[1])
            real_times = real_walks[real_mask][:, 2]
            real_ax.hist(real_times, range=[0, 1], bins=100)
            real_ax.set_title('real start edge: {} loc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], real_times.mean(), real_times.std()))

            fake_ax = ax[i + j, 1]
            fake_mask = np.logical_and(fake_walks[:, 0] == e[0], fake_walks[:, 1] == e[1])
            fake_times = fake_walks[fake_mask][:, 2]
            fake_ax.hist(fake_times, range=[0, 1], bins=100)
            fake_ax.set_title('fake start edge: {} loc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], fake_times.mean(), fake_times.std()))

            truth_train_ax = ax[i + j, 2]
            truth_train_mask = np.logical_and(truth_train_walks[:, 0] == e[0], truth_train_walks[:, 1] == e[1])
            truth_train_times = truth_train_walks[truth_train_mask][:, 2]
            truth_test_mask = np.logical_and(truth_test_walks[:, 0] == e[0], truth_test_walks[:, 1] == e[1])
            truth_test_times = truth_test_walks[truth_test_mask][:, 2]
            truth_train_ax.hist(truth_train_times, range=[0, 1], bins=100)
            truth_train_ax.set_title('truth train start edge: {} loc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], truth_test_times.mean(), truth_test_times.std()))

            truth_test_ax = ax[i + j, 3]
            truth_test_mask = np.logical_and(truth_test_walks[:, 0] == e[0], truth_test_walks[:, 1] == e[1])
            truth_test_times = truth_test_walks[truth_test_mask][:, 2]
            truth_test_ax.hist(truth_test_times, range=[0, 1], bins=100)
            truth_test_ax.set_title('truth test start edge: {} loc: {:.4f} scale: {:.4f}'.format(
                [int(v) for v in e], truth_test_times.mean(), truth_test_times.std()))

        plt.tight_layout()
        plt.savefig('{}/iter_{}_validation.png'.format(output_directory, _it + 1))
        plt.close()

        return real_walks, fake_graphs

def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.

    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".

    Returns
    -------
    noise tensor

    """

    if type == "Gaussian":
        noise = tf.random_normal(shape)
    elif type == 'Uniform':
        noise = tf.random_uniform(shape, minval=-1, maxval=1)
    else:
        log("ERROR: Noise type {} not supported".format(type))
    return noise


def sample_gumbel(shape, eps=1e-20):
    """
    Sample from a uniform Gumbel distribution. Code by Eric Jang available at
    http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
    Parameters
    ----------
    shape: Shape of the Gumbel noise
    eps: Epsilon for numerical stability.

    Returns
    -------
    Noise drawn from a uniform Gumbel distribution.

    """
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
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

    n_nodes = 91
    n_edges = n_nodes * n_nodes
    scale = 0.1
    rw_len = 2
    batch_size = 8
    train_ratio = 0.8
    t_end = 1.
    embedding_size = 32
    lr = 0.0003
    gpu_id = 0

    # random data from metro
    file = 'data/metro_user_6.txt'
    edges = np.loadtxt(file)
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
                    # momentum=0.9
            )

    temperature = tggan.params['temp_start']

    tggan.session.run(tggan.init_op)

    # fake_x_inputs, fake_t0_res_inputs, fake_node_inputs, fake_tau_inputs, max_length, fake_lengths, \
    # real_data, real_x_inputs, real_t0_res_inputs, real_node_inputs, real_tau_inputs, real_length_discretes, real_lengths, \
    # disc_real, disc_fake, \
    # t0_res_output,tau\
    #     = tggan.session.run([
    #     tggan.fake_x_inputs, tggan.fake_t0_res_inputs,
    #     tggan.fake_node_inputs, tggan.fake_tau_inputs, tggan.fake_lengths, tggan.fake_lengths,
    #     tggan.real_data, tggan.real_x_inputs, tggan.real_t0_res_inputs,
    #     tggan.real_node_inputs, tggan.real_tau_inputs, tggan.real_length_discretes, tggan.real_lengths,
    #     tggan.disc_real, tggan.disc_fake,
    #     tggan.t0_res_output, tggan.tau,
    # ], feed_dict={tggan.temp: temperature})
    #
    # tggan.session.close()
    #
    # print('real_data:\n', real_data)
    # print('real_x_inputs:\n', real_x_inputs)
    # print('real_t0_res_inputs:\n', real_t0_res_inputs)
    # print('real_node_inputs:\n', np.argmax(real_node_inputs, axis=-1))
    # print('real_tau_inputs:\n', real_tau_inputs)
    # print('real_length_discretes:\n', real_length_discretes)
    # print('real_lengths:\n', real_lengths)
    #
    # print('fake_x_inputs:\n', fake_x_inputs)
    # print('t0_res_output:\n', t0_res_output)
    # print('fake_t0_res_inputs:\n', fake_t0_res_inputs)
    # print('fake_node_inputs:\n', np.argmax(fake_node_inputs, axis=-1))
    # print('tau:\n', tau)
    # print('fake_tau_inputs:\n', fake_tau_inputs)
    # print('max_length:\n', max_length)
    # print('fake_lengths:\n', fake_lengths)

    # print('disc_real:\n', disc_real)
    # print('disc_fake:\n', disc_fake)

    max_iters = 10
    eval_every = 5
    plot_every = 5
    n_eval_loop = 3
    transitions_per_iter = batch_size * n_eval_loop
    eval_transitions = transitions_per_iter * 10
    model_name='metro'

    log_dict = tggan.train(
        train_edges=train_edges, test_edges=test_edges,
        n_eval_loop=n_eval_loop,
        stopping=None,
        eval_transitions=eval_transitions,
        eval_every=eval_every, plot_every=plot_every,
        max_patience=20, max_iters=max_iters,
        model_name=model_name,
        )
    log('-'*40)