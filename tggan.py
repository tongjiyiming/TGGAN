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
                 generator_tau_up_layers=[128],
                 generator_time_deconv_output_depth=8,
                 generator_time_sample_num=4,
                 generator_layers=[40],
                 discriminator_layers=[30],
                 W_down_generator_size=128, W_down_discriminator_size=128, batch_size=128, noise_dim=16,
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
            'generator_tau_up_layers': generator_tau_up_layers,
            'generator_time_deconv_output_depth': generator_time_deconv_output_depth,
            'generator_time_sample_num': generator_time_sample_num,
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
            'use_beta': use_beta,
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

        self.G_tau_up_layers = self.params['generator_tau_up_layers']
        self.G_t_deconv_output_depth = self.params['generator_time_deconv_output_depth']
        self.G_t_sample_n = self.params['generator_time_sample_num']

        # W_down and W_up for generator and discriminator
        self.t_end = tf.constant(value=self.params['t_end'], name='t_end',
                                 dtype=tf.float32, shape=[1])
        self.empty_tau = tf.zeros([self.batch_size, 1], dtype=tf.float32, name="empty_tau")
        self.tau_deconv_filter = tf.get_variable('Generator.tau_deconv_filter',
                                                 shape=[3, self.G_t_deconv_output_depth, 1],
                                                 dtype=tf.float32,
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
        self.fake_node_inputs, self.fake_tau_inputs = self.generator_function(self.batch_size, reuse=False, gumbel=use_gumbel,
                                                   legacy=legacy_generator)
        self.fake_inputs_discrete = self.generate_discrete(self.params['batch_size'], reuse=True,
                                                           gumbel=use_gumbel, legacy=legacy_generator)

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
        self.real_lengths = self.get_real_input_lengths(self.real_data[:, 1:, 0])

        self.real_x_input_discretes = tf.cast(self.real_data[:, 0, 0], dtype=tf.int64)
        self.real_x_inputs = tf.one_hot(self.real_x_input_discretes, 2)
        self.real_t0_res_inputs = self.real_data[:, 0:1, 2]

        self.discriminator_function = self.discriminator_recurrent
        self.disc_real = self.discriminator_function(self.real_node_inputs, self.real_tau_inputs)
        self.disc_fake = self.discriminator_function(self.fake_node_inputs, self.fake_tau_inputs, reuse=True)

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        if use_wgan:
            # WGAN lipschitz-penalty
            alpha = tf.random_uniform(
                shape=[self.params['batch_size'], 1, 1],
                minval=0.,
                maxval=1.
            )

            self.differences = self.fake_node_inputs - self.real_node_inputs
            self.interpolates = self.real_node_inputs + (alpha * self.differences)
            self.gradients = tf.gradients(self.discriminator_function(self.interpolates, reuse=True), self.interpolates)[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2]))
            self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
            self.disc_cost += self.params['Wasserstein_penalty'] * self.gradient_penalty

        # weight regularization; we omit W_down from regularization
        self.disc_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                      if 'Disc' in v.name
                                      and not 'W_down' in v.name]) * self.params['l2_penalty_discriminator']
        self.disc_cost += self.disc_l2_loss

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

    def discriminator_recurrent(self, node_inputs, tau_inputs, reuse=None):
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

            node_input_reshape = tf.reshape(node_inputs, [-1, self.N])
            node_output = tf.matmul(node_input_reshape, self.W_down_discriminator)
            node_output = tf.reshape(node_output, [-1, self.rw_len*2, int(self.W_down_discriminator.shape[-1])])
            node_output = tf.unstack(node_output, axis=1)

            tau_input_reshape = tf.reshape(tau_inputs, [-1, 1])
            tau_output = tf.layers.dense(tau_input_reshape, int(self.W_down_discriminator.shape[-1]),
                                         reuse=reuse, name='Discriminator.tau_up')
            tau_output = tf.reshape(tau_output, [-1, self.rw_len, int(self.W_down_discriminator.shape[-1])])
            tau_output = tf.unstack(tau_output, axis=1)

            inputs = []
            for i in range(self.rw_len):
                inputs += [node_output[i*2]] + [node_output[i*2+1]] + [tau_output[i]]

            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

            disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

            output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell,
                                                                inputs=inputs,
                                                                dtype='float32')

            last_output = output_disc[-1]

            final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")
            return final_score

    def generate_discrete(self, n_samples, reuse=True, z=None, gumbel=True, legacy=False):
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
        fake_node_outputs, fake_tau_outputs = self.generator_function(
            n_samples, reuse, z, gumbel=gumbel, legacy=legacy)
        fake_node_outputs_discrete = tf.argmax(fake_node_outputs, axis=-1)

        return fake_node_outputs_discrete, fake_tau_outputs

    def generator_recurrent(self, n_samples, reuse=None, z=None, gumbel=True, legacy=False):
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
            with tf.name_scope('LSTM'):
                node_outputs = []
                tau_outputs = []

                def lstm_cell(lstm_size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)

                self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.G_layers])

                # LSTM tine steps
                for i in range(self.rw_len*2):

                    # Get LSTM output
                    with tf.variable_scope('LSTM_CELL') as lstm_cell_scope:
                        if i > 0: lstm_cell_scope.reuse_variables()
                        output, state = self.stacked_lstm.call(inputs, state)

                    with tf.name_scope('GEN_NODE'):
                        # Blow up to dimension N using W_up
                        output_bef = tf.matmul(output, self.W_up) + self.b_W_up

                        # Perform Gumbel softmax to ensure gradients flow
                        if gumbel: output = gumbel_softmax(output_bef, temperature=self.temp, hard=True)
                        else:      output = tf.nn.softmax(output_bef)

                    # generate \tau
                    if i % 2 == 1:
                        with tf.variable_scope('TAU_TIME') as tau_scope:
                            if i > 1: tau_scope.reuse_variables()
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
                                choice = tf.random_uniform([self.G_t_sample_n], maxval=int(tau.shape[1]),
                                                           dtype=tf.int64)
                                tau = tf.gather(tau, choice, axis=1)
                                tau = tf.reduce_mean(tau, axis=1)
                                tau = tf.layers.dense(tau, 1, name="Generator.tau_deconv_last",
                                                      activation=None)
                            # save outputs
                            tau_outputs.append(tau)

                    # Back to dimension d
                    inputs = tf.matmul(output, self.W_down_generator)

                    node_outputs.append(output)

                node_outputs = tf.stack(node_outputs, axis=1)
                tau_outputs = tf.stack(tau_outputs, axis=1)
        return node_outputs, tau_outputs

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

    def train(self, max_iters=50000, stopping=None, eval_transitions=15e6,
              max_patience=5, eval_every=500, plot_every=-1,
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
        n_eval_iters = int(eval_transitions / self.batch_size)
        sample_many = self.generate_discrete(self.batch_size, reuse=True)

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
                log('{:<7}/{:<8} training iterations, took {} seconds so far...'.format(_it, max_iters, int(t)))
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

                fake_walks = []
                fake_x_t0 = []
                real_walks = []
                real_x_t0 = []
                for _ in range(n_eval_iters):
                    edges, fake_t = self.session.run(sample_many, {self.temp: 0.5})
                    fake_x = np.ones((self.batch_size, 1))
                    fake_t0 = np.zeros((self.batch_size, 1))
                    # fake_t = np.zeros((self.batch_size, self.rw_len, 1))
                    # fake_x, fake_t0, fake_v, fake_u, fake_t = smpls[0]
                    # # print('fake_t:\n {} \nfake_v: \n{} \nfake_u: \n{}'.format(fake_t, fake_v, fake_u))

                    smpls = np.c_[edges, fake_t[:, :, 0]]

                    fake_walks.append(smpls)
                    fake_start = np.c_[fake_x, fake_t0]
                    fake_x_t0.append(fake_start)

                    real_x, real_t0, real_edge, real_tau, \
                    walk = self.session.run([
                        self.real_x_input_discretes, self.real_t0_res_inputs,
                        self.real_edge_inputs_discrete, self.real_tau_inputs,
                        self.real_edge_inputs_discrete,
                    ],feed_dict={self.temp: 0.5})

                    walk = np.c_[real_edge, real_tau]
                    real_walks.append(walk)
                    real_start = np.stack([real_x, real_t0[:, 0]], axis=1)
                    real_x_t0.append(real_start)

                fake_walks = np.array(fake_walks).reshape(-1, 3)
                fake_x_t0 = np.array(fake_x_t0).reshape(-1, 2)
                real_walks = np.array(real_walks).reshape(-1, 3)
                real_x_t0 = np.array(real_x_t0).reshape(-1, 2)
                # print('fake_walks: \n{} \nreal_walks: \n{}'.format(fake_walks, real_walks))

                # plot edges time series for qualitative evaluation
                fake_e_list, fake_e_counts = np.unique(fake_walks[:, 0:2], return_counts=True, axis=0)
                real_e_list, real_e_counts = np.unique(real_walks[:, 0:2], return_counts=True, axis=0)

                real_x_list, real_x_counts = np.unique(real_x_t0[:, 0], return_counts=True)
                fake_x_list, fake_x_counts = np.unique(fake_x_t0[:, 0], return_counts=True)
                n_e = len(real_e_list)

                fig = plt.figure(figsize=(9, 2*9))
                fig.suptitle('Real and Fake edges comparisons')
                dx = 0.3
                dy = dx
                zpos = 0
                real_ax = fig.add_subplot(211, projection='3d')
                real_ax.bar3d(real_e_list[:, 0], real_e_list[:, 1], zpos, dx, dy, real_e_counts)
                real_ax.set_xlim([0, self.N])
                real_ax.set_ylim([0, self.N])
                real_ax.set_xticks(range(self.N))
                real_ax.set_yticks(range(self.N))
                real_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                real_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                real_ax.set_title('real edges number: {}'.format(len(real_e_list)))

                fake_ax = fig.add_subplot(212, projection='3d')
                fake_ax.bar3d(fake_e_list[:, 0], fake_e_list[:, 1], zpos, dx, dy, fake_e_counts)
                fake_ax.set_xlim([0, self.N])
                fake_ax.set_ylim([0, self.N])
                fake_ax.set_xticks(range(self.N))
                fake_ax.set_yticks(range(self.N))
                fake_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                fake_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(self.N)])
                fake_ax.set_title('fake edges number: {}'.format(len(fake_e_list)))

                plt.tight_layout()
                plt.savefig('{}/iter_{}_edges_counts_validation.png'.format(output_directory, _it+1), dpi=90)
                plt.close()

                fig, ax = plt.subplots(n_e+3, 2, figsize=(2*9, (n_e+3)*4))
                i = 0
                real_ax = ax[i, 0]
                real_ax.bar(real_x_list, real_x_counts)
                real_ax.set_xlim([-1, 2])
                real_ax.set_title('real start x number: {}'.format(len(real_x_list)))
                fake_ax = ax[i, 1]
                fake_ax.bar(fake_x_list, fake_x_counts)
                fake_ax.set_xlim([-1, 2])
                fake_ax.set_title('fake start x number: {}'.format(len(fake_x_list)))
                i = 1
                for j, e in enumerate([0, 1]):
                    real_ax = ax[i+j, 0]
                    real_mask = real_x_t0[:, 0] == e
                    real_times = real_x_t0[real_mask][:, 1]
                    real_ax.hist(real_times, range=[-0.5, 1.5], bins=200)
                    real_ax.set_title('real x node: {} loc: {:.4f} scale: {:.4f}'.format(
                        int(e), real_times.mean(), real_times.std()))

                    fake_ax = ax[i+j, 1]
                    fake_mask = fake_x_t0[:, 0] == e
                    fake_times = fake_x_t0[fake_mask][:, 1]
                    fake_ax.hist(fake_times, range=[-0.5, 1.5], bins=200)
                    fake_ax.set_title('fake x node: {} loc: {:.4f} scale: {:.4f}'.format(
                        int(e), fake_times.mean(), fake_times.std()))
                i = 3
                for j, e in enumerate(real_e_list):
                    real_ax = ax[i+j, 0]
                    real_mask = np.logical_and(real_walks[:, 0] == e[0], real_walks[:, 1] == e[1])
                    real_times = real_walks[real_mask][:, 2]
                    real_ax.hist(real_times, range=[-0.5, 1.5], bins=200)
                    real_ax.set_title('real start edge: {} loc: {:.4f} scale: {:.4f}'.format(
                        [int(v) for v in e], real_times.mean(), real_times.std()))

                    fake_ax = ax[i+j, 1]
                    fake_mask = np.logical_and(fake_walks[:, 0] == e[0], fake_walks[:, 1] == e[1])
                    fake_times = fake_walks[fake_mask][:, 2]
                    fake_ax.hist(fake_times, range=[-0.5, 1.5], bins=200)
                    fake_ax.set_title('fake start edge: {} loc: {:.4f} scale: {:.4f}'.format(
                        [int(v) for v in e], fake_times.mean(), fake_times.std()))

                plt.tight_layout()
                plt.savefig('{}/iter_{}_validation.png'.format(output_directory, _it+1))
                plt.close()
                # except: log('plotting function got error, continue training...')

                fake_graph_file = "{}/{}_assembled_graph_iter_{}.npz".format(output_directory, timestr, _it + 1)
                np.savez_compressed(fake_graph_file, fake_walks=fake_walks, real_walks=real_walks)
                log('assembled graph to file: {} \nas array\n {}\n with shape: {}'.format(
                    fake_graph_file, fake_walks[:10, :], fake_walks.shape))
                save_file = "{}/{}_iter_{}.ckpt".format(save_directory, model_name, _it+1)
                d = saver.save(self.session, save_file)
                log("**** Saving snapshots into {} ****".format(save_file))

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
    rw_len = 1
    batch_size = 8
    train_ratio = 0.8
    t_end = 1.
    embedding_size = 32
    lr = 0.000003
    gpu_id = 0

    # random data from metro
    file = 'data/metro_user_4.txt'
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
                    # momentum=0.9
            )

    temperature = tggan.params['temp_start']

    tggan.session.run(tggan.init_op)

    # fake_x_inputs, fake_t0_res_inputs, fake_v_inputs, fake_u_inputs, fake_tau_inputs, fake_lengths, \
    # real_data, real_x_inputs, real_t0_res_inputs, real_v_inputs, real_u_inputs, real_tau_inputs, real_lengths, \
    # disc_real, disc_fake \
    #     = tggan.session.run([
    #     tggan.fake_x_inputs, tggan.fake_t0_res_inputs,
    #     tggan.fake_v_inputs, tggan.fake_u_inputs, tggan.fake_tau_inputs, tggan.fake_lengths,
    #     tggan.real_data, tggan.real_x_inputs, tggan.real_t0_res_inputs,
    #     tggan.real_v_inputs, tggan.real_u_inputs, tggan.real_tau_inputs,
    #     tggan.real_lengths,
    #     tggan.disc_real, tggan.disc_fake
    # ], feed_dict={tggan.temp: temperature})
    #
    # tggan.session.close()
    #
    # print('real_data:\n', real_data)
    # print('real_x_inputs:\n', real_x_inputs)
    # print('real_t0_res_inputs:\n', real_t0_res_inputs)
    # print('real_v_inputs:\n', np.argmax(real_v_inputs, axis=-1))
    # print('real_u_inputs:\n', np.argmax(real_u_inputs, axis=-1))
    # print('real_tau_inputs:\n', real_tau_inputs)
    # print('real_lengths:\n', real_lengths)
    #
    # print('fake_x_inputs:\n', fake_x_inputs)
    # print('fake_t0_res_inputs:\n', fake_t0_res_inputs)
    # print('fake_v_inputs:\n', np.argmax(fake_v_inputs, axis=-1))
    # print('fake_u_inputs:\n', np.argmax(fake_u_inputs, axis=-1))
    # print('fake_tau_inputs:\n', fake_tau_inputs)
    # print('fake_lengths:\n', fake_lengths)

    # print('disc_real:\n', disc_real)
    # print('disc_fake:\n', disc_fake)

    max_iters = 10
    eval_every = 5
    plot_every = 5
    n_eval_loop = 1
    transitions_per_iter = batch_size * n_eval_loop
    eval_transitions = transitions_per_iter * 100
    model_name='metro'

    log_dict = tggan.train(
        # n_eval_loop=n_eval_loop,
       stopping=None,
        eval_transitions=eval_transitions,
        eval_every=eval_every, plot_every=plot_every,
        max_patience=20, max_iters=max_iters,
        model_name=model_name,
        )
    log('-'*40)