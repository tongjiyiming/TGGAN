import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time
import pickle

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

# # create tf logger
# logger = logging.getLogger('tensorflow')
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

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
# def log(str): print(str)

import numpy as np
import tensorflow as tf
log('is GPU available? {}'.format(tf.test.is_gpu_available(cuda_only=True)))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

from evaluation import *
from utils import *


class TGGAN:
    """
    NetGAN class, an implicit generative model for graphs using random walks.
    """

    def __init__(self, N, rw_len, walk_generator,
                 t_end,
                 generator_start_layers=[30],
                 generator_layers=[40],
                 discriminator_layers=[30],
                 W_down_generator_size=128,
                 W_down_discriminator_size=128,
                 batch_size=128, noise_dim=16,
                 noise_type="Gaussian", learning_rate=0.0003, disc_iters=3, wasserstein_penalty=10,
                 l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, temp_start=5.0, min_temperature=0.5,
                 temperature_decay=1 - 5e-5, seed=15, gpu_id=0, use_gumbel=True, legacy_generator=False):
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
            't_end': t_end,
            'noise_dim': noise_dim,
            'noise_type': noise_type,
            'generator_start_layers': generator_start_layers,
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
            'legacy_generator': legacy_generator
        }

        # assert rw_len > 1, "Random walk length must be > 1."

        tf.set_random_seed(seed)

        self.N = N
        self.rw_len = rw_len
        self.batch_size = self.params['batch_size']
        self.noise_dim = self.params['noise_dim']
        self.G_t_layers = self.params['generator_start_layers']
        self.G_layers = self.params['Generator_Layers']
        self.D_layers = self.params['Discriminator_Layers']
        self.temp = tf.placeholder(1.0, shape=(), name="temperature")

        # W_down and W_up for generator and discriminator
        # self.max_length = tf.constant(value=self.rw_len, name='Generator.max_length', dtype=tf.int64, shape=[1])
        self.t_end = tf.constant(value=self.params['t_end'], name='Generator.t_end', dtype=tf.float32, shape=[1])
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

        self.W_tau = tf.get_variable("Generator.W_tau", shape=[self.params['W_Down_Discriminator_size'], 1],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        self.b_W_tau = tf.get_variable("Generator.W_tau_bias", dtype=tf.float32, initializer=tf.zeros_initializer,
                                      shape=1)

        self.W_down_x_generator = tf.get_variable('Generator.W_Down_x',
                                                shape=[2, self.params['W_Down_Discriminator_size']],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())

        self.generator_function = self.generator_recurrent
        self.discriminator_function = self.discriminator_recurrent

        self.fake_v_inputs, self.fake_t_inputs = self.generator_function(
            self.batch_size, reuse=False, gumbel=use_gumbel, legacy=legacy_generator)

        self.fake_lengths = tf.ones(name='Generator.fake_lengths', dtype=tf.int64, shape=[self.batch_size,])

        self.fake_inputs_discrete = self.generate_discrete(self.batch_size, reuse=True,
                                                           gumbel=use_gumbel, legacy=legacy_generator)

        # Pre-fetch real random walks
        dataset = tf.data.Dataset.from_generator(generator=walk_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[self.batch_size, self.rw_len+1, 3])

        # dataset_batch = dataset.prefetch(2).batch(self.params['batch_size'])
        dataset_batch = dataset.prefetch(100)
        batch_iterator = dataset_batch.make_one_shot_iterator()
        self.real_data = batch_iterator.get_next()
        self.real_v_inputs_discrete = tf.cast(self.real_data[:, 1:, 0], dtype=tf.int64)
        self.real_v_inputs = tf.one_hot(self.real_v_inputs_discrete, self.N)
        self.real_t0_inputs = self.real_data[:, 0:1, 0]
        self.real_t_inputs = tf.expand_dims(self.real_data[:, 1:, 2], axis=2)
        self.real_lengths = self.get_real_input_lengths(self.real_v_inputs_discrete)

        self.disc_real = self.discriminator_function(self.real_v_inputs, self.real_t_inputs, self.real_lengths)
        # self.disc_fake = self.discriminator_function(self.fake_inputs, self.fake_x, self.fake_y, self.fake_lengths,
        #                                              reuse=False)
        self.disc_fake = self.discriminator_function(self.fake_v_inputs, self.fake_t_inputs, self.fake_lengths, reuse=True)

        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        # WGAN lipschitz-penalty
        # alpha_v = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0.,maxval=1.)
        #
        # self.differences_v = self.fake_v_inputs - self.real_v_inputs
        # self.interpolates_v = self.real_v_inputs + (alpha_v * self.differences_v)
        # self.gradients_v = tf.gradients(
        #     self.discriminator_function(self.interpolates_v, self.fake_lengths, reuse=True),
        #     [self.interpolates_v])[0]
        # self.slopes = tf.sqrt(
        #     tf.reduce_sum(tf.stack([
        #         tf.reduce_sum(tf.square(self.gradients_v), reduction_indices=[1, 2]),
        #     ]), reduction_indices=[1])
        # )

        # alpha_v = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0.,maxval=1.)
        # alpha_t0 = tf.random_uniform(shape=[self.params['batch_size'], 1], minval=0.,maxval=1.)
        # self.differences_v = self.fake_v_inputs - self.real_v_inputs
        # self.interpolates_v = self.real_v_inputs + (alpha_v * self.differences_v)
        # self.differences_t0 = self.fake_t0_inputs - self.real_t0_inputs
        # self.interpolates_t0 = self.real_t0_inputs + (alpha_t0 * self.differences_t0)
        # self.gradients_v, self.gradients_t0 = tf.gradients(
        #     self.discriminator_function(self.interpolates_v, self.interpolates_t0, self.fake_lengths, reuse=True),
        #     [self.interpolates_v, self.interpolates_t0])
        # self.slopes = tf.sqrt(
        #     tf.reduce_sum(tf.stack([
        #         tf.reduce_sum(tf.square(self.gradients_v), reduction_indices=[1, 2]),
        #         tf.reduce_sum(tf.square(self.gradients_t0), reduction_indices=[1]),
        #     ]), reduction_indices=[1])
        # )

        # alpha_v = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0.,maxval=1.)
        # alpha_t = tf.random_uniform(shape=[self.params['batch_size'], 1, 1], minval=0.,maxval=1.)
        # self.differences_v = self.fake_v_inputs - self.real_v_inputs
        # self.interpolates_v = self.real_v_inputs + (alpha_v * self.differences_v)
        # self.differences_t = self.fake_t_inputs - self.real_t_inputs
        # self.interpolates_t = self.real_t_inputs + (alpha_t * self.differences_t)
        # self.gradients_v, self.gradients_t = tf.gradients(
        #     self.discriminator_function(self.interpolates_v, self.interpolates_t, self.fake_lengths, reuse=True),
        #     [self.interpolates_v, self.interpolates_t])
        # self.slopes = tf.sqrt(
        #     tf.reduce_sum(tf.stack([
        #         tf.reduce_sum(tf.square(self.gradients_v), reduction_indices=[1, 2]),
        #         tf.reduce_sum(tf.square(self.gradients_t), reduction_indices=[1, 2]),
        #     ]), reduction_indices=[1])
        # )
        #
        # self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        # self.disc_cost += self.params['Wasserstein_penalty'] * self.gradient_penalty

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

        if gpu_id is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.InteractiveSession(config=config)
        self.init_op = tf.global_variables_initializer()

    def discriminator_recurrent(self, edges, ts, lengths, reuse=None):
        """
        Discriminate real from fake random walks using LSTM.
        Parameters
        ----------
        edges: tf.tensor, shape (None, rw_len, N)
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

            def lstm_cell(lstm_size):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
            # disc_t0_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])
            disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])

            # t0_input = tf.layers.dense(t0, self.D_layers[-1],
            #                            name='Discriminator.t0_inputs', reuse=reuse, activation=tf.nn.tanh)
            # t0_input = [t0_input]
            # output_t0, state_t0 = tf.nn.static_rnn(cell=disc_t0_lstm_cell,
            #                                                     inputs=[t0],
            #                                                     dtype='float32')

            v_input_reshape = tf.reshape(edges, [-1, self.N])
            v_input_reshape = tf.matmul(v_input_reshape, self.W_down_discriminator)
            v_input_reshape = tf.reshape(v_input_reshape, [-1, self.rw_len, int(self.W_down_discriminator.get_shape()[-1])])

            t_inputs = ts[:, :, 0]
            for ix, size in enumerate(self.D_layers):
                t_inputs = tf.layers.dense(t_inputs, size, reuse=reuse, name="Discriminator.t_{}".format(ix),
                                           activation=tf.nn.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            t_inputs = tf.expand_dims(t_inputs, axis=1)

            input_reshape = tf.concat([v_input_reshape, t_inputs], axis=2)
            inputs = tf.unstack(input_reshape, axis=1)

            output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell,
                                                                inputs=inputs,
                                                                dtype='float32',
                                                                sequence_length=lengths,
                                                                )

            last_output = output_disc[-1]
            # last_output = tf.concat([last_output, t0_input], axis=1)

            final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")

        # with tf.variable_scope('Discriminator') as scope:
        #     if reuse == True:
        #         scope.reuse_variables()
        #
        #     v_input_reshape = tf.reshape(edges, [-1, self.N])
        #     v_input_reshape = tf.matmul(v_input_reshape, self.W_down_discriminator)
        #     v_input_reshape = tf.reshape(v_input_reshape, [-1, self.rw_len, int(self.W_down_discriminator.get_shape()[-1])])
        #     inputs = tf.unstack(v_input_reshape, axis=1)
        #
        #     def lstm_cell(lstm_size):
        #         return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
        #     disc_lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in self.D_layers])
        #
        #     output_disc, state_disc = tf.contrib.rnn.static_rnn(cell=disc_lstm_cell,
        #                                                         inputs=inputs,
        #                                                         dtype='float32',
        #                                                         sequence_length=lengths,
        #                                                         )
        #
        #     last_output = output_disc[-1]
        #     final_score = tf.layers.dense(last_output, 1, reuse=reuse, name="Discriminator.Out")

            return final_score

    def to_discrete(self, x):
        return tf.argmax(x, axis=-1)

    def get_real_input_lengths(self, inputs_discrete):
        lengths = tf.math.reduce_sum(tf.cast(tf.math.less(-1, tf.cast(inputs_discrete, dtype=tf.int32)), dtype=tf.int64), axis=1)
        return lengths

    def generator_recurrent(self, n_samples, reuse=None, z=None, gumbel=True, decoder=True, legacy=False):
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

            # initial states h and c are randomly sampled for each lstm cell
            if z is None:
                initial_states_noise = self.make_noise([n_samples, self.noise_dim], self.params['noise_type'])
            else:
                initial_states_noise = z

            # # generate start node binary if not need
            # if t0: t0_res = t0
            # else:
            #     # t_start_wait = initial_states_noise
            #     # for ix, size in enumerate(self.G_t_layers):
            #     #     t_start_wait = tf.layers.dense(t_start_wait, size, name="Generator.t0_{}".format(ix + 1), reuse=reuse,
            #     #                                activation=tf.nn.tanh)
            #     # t_start_wait = tf.layers.dense(t_start_wait, 1, name="Generator.t0_wait", reuse=reuse,
            #     #                      activation=tf.nn.sigmoid)
            #     # t0_res = tf.nn.relu(self.t_end - t_start_wait, name='Generator.t0_res')
            #
            #     ### Perform variational decoder to ensure gradients flow for t
            #     if decoder:
            #         mu_start = initial_states_noise
            #         sigma_start = initial_states_noise
            #         for ix, size in enumerate(self.G_t_layers):
            #             mu_start = tf.layers.dense(mu_start, size, name="Generator.mu_t0_{}".format(ix),
            #                                        reuse=reuse, activation=tf.nn.tanh)
            #             sigma_start = tf.layers.dense(sigma_start, size, name="Generator.sigma_t0_{}".format(ix),
            #                                           reuse=reuse, activation=tf.nn.tanh)
            #         mu_start = tf.layers.dense(mu_start, 1, name="Generator.mu_t0_last", reuse=reuse,
            #                                   activation=tf.nn.tanh)
            #         sigma_start = tf.layers.dense(sigma_start, 1, name="Generator.sigma_t0_last", reuse=reuse,
            #                                      activation=tf.nn.tanh)
            #         # t_start_wait = tf.nn.relu(self.var_decoder(mu_start, sigma_start), name="Generator.t0_wait")
            #         t_start_wait = self.var_decoder(mu_start, sigma_start)
            #     else:
            #         t_start_wait = tf.nn.relu(tf.layers.dense(initial_states_noise, 1, name="Generator.start_t_decoder",
            #                                           activation=tf.nn.sigmoid))
            #
            #     t0_res = tf.nn.relu(self.t_end - t_start_wait, name='Generator.t0_res')

            # generate input vector for lstm from x
            inputs = tf.zeros([n_samples, self.params['W_Down_Generator_size']])
            # inputs = tf.layers.dense(t0_res, self.params['W_Down_Generator_size'],
            #                          name='Generator.t0_res_inputs', reuse=reuse, activation=tf.nn.tanh)

            # initial lstm states from noise
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

            # start to generate edge and end node y
            v_outputs = []
            t_outputs = []

            def lstm_cell(lstm_size, name):
                return tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse, 
                                                    name="Generator.{}".format(name))

            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell(size, "lstm_{}".format(size)) for size in self.G_layers])

            # LSTM steps
            state = initial_states
            for i in range(self.rw_len):
                if i > 0: tf.get_variable_scope().reuse_variables()

                # Get LSTM output
                output, state = self.stacked_lstm.call(inputs, state)

                # Blow up to dimension N x N using W_up
                v_logit = tf.matmul(output, self.W_up) + self.b_W_up

                # Perform Gumbel softmax to ensure gradients flow for e, and end node y
                if gumbel:
                    v_output = self.gumbel_softmax(v_logit, temperature=self.temp, hard=True)
                    # v_input = tf.one_hot(v_output, self.N)
                else:
                    v_output = tf.nn.softmax(v_logit)
                    # v_input = v_output

                # Back to dimension d
                inputs = tf.matmul(v_output, self.W_down_generator)

                # generate \tau time
                # tau = tf.nn.sigmoid(tf.matmul(inputs, self.W_tau) + self.b_W_tau)
                # generate \tau with decoder sampling
                if decoder:
                    loc = inputs
                    scale = inputs
                    for ix, size in enumerate(self.G_t_layers):
                        loc = tf.layers.dense(loc, size, name="Generator.loc_tau_{}".format(ix),
                                              reuse=reuse, activation=tf.nn.tanh,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())
                        scale = tf.layers.dense(scale, size, name="Generator.scale_tau_{}".format(ix),
                                                reuse=reuse, activation=tf.nn.tanh,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
                    loc = tf.layers.dense(loc, 1, name="Generator.loc_tau_last", reuse=reuse, activation=None)
                    scale = tf.layers.dense(scale, 1, name="Generator.scale_tau_last", reuse=reuse, activation=None)
                    # tau = self.var_decoder(mu_start, sigma_start)
                    tau = tf.random_normal([self.batch_size,], mean=loc, stddev=scale)
                else:
                    tau = tf.layers.dense(initial_states_noise, 1, name="Generator.tau_decoder", reuse=reuse, activation=None)
                t_outputs.append(tau)

                # if i == 0: t_outputs.append(tau)
                # else: t_outputs.append(tf.math.add(tau, t_outputs[-1]))

                v_outputs.append(v_output)

            v_outputs = tf.stack(v_outputs, axis=1)
            t_outputs = tf.stack(t_outputs, axis=1)
        return v_outputs, t_outputs

    def var_decoder(self, mu, sigma):
        shape = tf.shape(mu)
        eps = tf.truncated_normal(shape, mean=[0.], stddev=[1.])
        # eps = tf.random_normal(shape, mean=[0.0], stddev=[1.0])
        t = mu + sigma*eps
        return t

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
        fake_v_outputs, fake_t_outputs = self.generator_function(n_samples, reuse, z,
                                                                 gumbel=gumbel, legacy=legacy)
        fake_v_outputs_discrete = tf.argmax(fake_v_outputs, axis=-1)

        return fake_v_outputs_discrete, fake_t_outputs

    def train(self, n_eval_loop, max_iters=50000, stopping=None, eval_transitions=15e6,
              transitions_per_iter=150000, max_patience=5, eval_every=500, plot_every=-1,
              save_directory="snapshots", output_directory='output',
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

        if stopping == None:  # use VAL criterion
            best_performance = 0.0
            patience = max_patience
            log("**** Using VAL criterion for early stopping ****")

        else:  # use EO criterion
            assert "float" in str(type(stopping)) and stopping > 0 and stopping <= 1
            log("**** Using EO criterion of {} for early stopping".format(stopping))

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # if model_name is None:
        #     # Find the file corresponding to the lowest vacant model number to store the snapshots into.
        #     model_number = 0
        #     while os.path.exists("{}/model_best_{}.ckpt".format(save_directory, model_number)):
        #         model_number += 1
        #     save_file = "{}/model_best_{}.ckpt".format(save_directory, model_number)
        #     open(save_file, 'a').close()  # touch file
        # else:
        #     save_file = "{}/{}_best.ckpt".format(save_directory, model_name)
        # print("**** Saving snapshots into {} ****".format(save_file))

        starting_time = time.time()
        saver = tf.train.Saver()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_number = 0

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

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # # Validation labels
        # actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))

        # Some lists to store data into.
        gen_losses = []
        disc_losses = []
        # graphs = []
        # val_performances = []
        # eo = []
        temperature = self.params['temp_start']

        # transitions_per_walk = self.rw_len - 1
        # # Sample lots of random walks, used for evaluation of model.
        # sample_many_count = int(np.round(transitions_per_iter / transitions_per_walk))
        # sample_many = self.generate_discrete(sample_many_count, reuse=True)
        # n_eval_walks = eval_transitions / transitions_per_walk
        # n_eval_iters = int(np.round(n_eval_walks / sample_many_count))

        n_eval_iters = int(eval_transitions / self.batch_size)
        sample_many = self.generate_discrete(self.batch_size, reuse=True)

        log("**** Starting training. ****")

        for _it in range(max_iters):

            if _it > 0 and _it % (1000) == 0:
                t = time.time() - starting_time
                log('{:<7}/{:<8} training iterations, took {} seconds so far...'.format(_it, max_iters, int(t)))

            # Generator training iteration
            gen_loss, _ = self.session.run([self.gen_cost, self.gen_train_op],
                                           feed_dict={self.temp: temperature})

            _disc_l = []
            # Multiple discriminator training iterations.
            for _ in range(self.params['disc_iters']):
                disc_fake, disc_real, \
                disc_loss, _ = self.session.run([
                    self.disc_fake, self.disc_real,
                     self.disc_cost, self.disc_train_op],
                    feed_dict={self.temp: temperature}
                )
                # print('iteration - {} \ndisc_fake: \n{} \ndisc_real: \n{}'.format(_it+1, disc_fake, disc_real))
                _disc_l.append(disc_loss)
            gen_losses.append(gen_loss)
            disc_losses.append(np.mean(_disc_l))

            if (_it + 1) % 1000 == 0:
                log('gen_loss: {} disc_loss: {}'.format(gen_loss, np.mean(_disc_l)))
                log('disc_fake max:{} min:{} shape:{}'.format(disc_fake.max(), disc_fake.min(), disc_fake.shape))
                log('disc_real max:{} min:{} shape:{}'.format(disc_real.max(), disc_real.min(), disc_real.shape))

            # Evaluate the model's progress.
            if (_it+1) % eval_every == 0:

                # Update Gumbel temperature
                temperature = np.maximum(
                    self.params['temp_start'] * np.exp(-(1 - self.params['temperature_decay']) * _it),
                    self.params['min_temperature'])

                log('**** Starting Evaluation ****')

                fake_walks = []
                real_walks = []
                for _ in range(n_eval_iters):
                    smpls = self.session.run([sample_many], {self.temp: 0.5})
                    fake_v, fake_t = smpls[0]
                    # print('fake_t:\n {} \nfake_v: \n{}'.format(fake_t, fake_v))

                    smpls = np.stack([fake_v[:, 0], fake_t[:, 0, 0]], axis=1)
                    fake_walks.append(smpls)

                    real_v, real_t0, real_t = self.session.run([
                        self.real_v_inputs_discrete, self.real_t0_inputs, self.real_t_inputs
                    ],feed_dict={self.temp: 0.5})
                    # walk = np.stack([real_t0[:, 0], real_v[:, 0]], axis=1)
                    walk = np.stack([real_v[:, 0], real_t[:, 0, 0]], axis=1)
                    real_walks.append(walk)
                fake_walks = np.array(fake_walks).reshape(-1, 2)
                real_walks = np.array(real_walks).reshape(-1, 2)

                fake_graph_file = "{}/{}_assembled_graph_iter_{}.npz".format(output_directory, timestr, _it + 1)
                np.savez_compressed(fake_graph_file, fake_walks=fake_walks, real_walks=real_walks)
                log('assembled graph to file: {} \nas array\n {}\n with shape: {}'.format(
                    fake_graph_file, fake_walks[:10, :], fake_walks.shape))
                save_file = "{}/{}_iter_{}.ckpt".format(save_directory, model_name, _it+1)
                d = saver.save(self.session, save_file)
                log("**** Saving snapshots into {} ****".format(save_file))

                # plot edges time series for qualitative evaluation
                real_v_list, real_v_counts = np.unique(real_walks[:, 0], return_counts=True)
                fake_v_list, fake_v_counts = np.unique(fake_walks[:, 0], return_counts=True)
                n_v = len(real_v_list)

                fig, ax = plt.subplots(n_v+1, 2, figsize=(2*9, (n_v+1)*4))

                real_ax = ax[0, 0]
                real_ax.bar(real_v_list, real_v_counts)
                real_ax.set_xlim([-1, self.N+1])
                real_ax.set_title('real nodes number: {}'.format(len(real_v_list)))
                fake_ax = ax[0, 1]
                fake_ax.bar(fake_v_list, fake_v_counts)
                fake_ax.set_xlim([-1, self.N+1])
                fake_ax.set_title('fake nodes number: {}'.format(len(fake_v_list)))

                for i, e in enumerate(real_v_list):
                    real_ax = ax[i+1, 0]
                    real_mask = real_walks[:, 0] == e
                    real_times = real_walks[real_mask][:, 1]
                    real_ax.hist(real_times, range=[-0.5, 1.5], bins=200)
                    real_ax.set_title('start node: {} loc: {:.4f} scale: {:.4f}'.format(
                        int(e), real_times.mean(), real_times.std()))

                    fake_ax = ax[i+1, 1]
                    fake_mask = fake_walks[:, 0] == e
                    fake_times = fake_walks[fake_mask][:, 1]
                    fake_ax.hist(fake_times, range=[-0.5, 1.5], bins=200)
                    fake_ax.set_title('start node: {} loc: {:.4f} scale: {:.4f}'.format(
                        int(e), fake_times.mean(), fake_times.std()))

                plt.tight_layout()
                plt.savefig('{}/iter_{}_validation.png'.format(output_directory, _it+1))
                plt.close()

            #     # Sample lots of random walks.
            #     smpls = []
            #     for _ in range(n_eval_iters):
            #         smpls.append(self.session.run(sample_many, {self.temp: 0.5}))
            #
            #     # Compute score matrix
            #     gr = utils.score_matrix_from_random_walks(np.array(smpls).reshape([-1, self.rw_len]), self.N)
            #     gr = gr.tocsr()
            #
            #     # Assemble a graph from the score matrix
            #     _graph = utils.graph_from_scores(gr, A_orig.sum())
            #     # Compute edge overlap
            #     edge_overlap = utils.edge_overlap(A_orig.toarray(), _graph)
            #     graphs.append(_graph)
            #     eo.append(edge_overlap)
            #
            #     edge_scores = np.append(gr[tuple(val_ones.T)].A1, gr[tuple(val_zeros.T)].A1)
            #
            #     # Compute Validation ROC-AUC and average precision scores.
            #     val_performances.append((roc_auc_score(actual_labels_val, edge_scores),
            #                              average_precision_score(actual_labels_val, edge_scores)))
            #
            #
            #     print("**** Iter {:<6} Val ROC {:.3f}, AP: {:.3f}, EO {:.3f} ****".format(_it,
            #                                                                               val_performances[-1][0],
            #                                                                               val_performances[-1][1],
            #                                                                               edge_overlap / A_orig.sum()))
            #
            #     if stopping is None:  # Evaluate VAL criterion
            #         if np.sum(val_performances[-1]) > best_performance:
            #             # New "best" model
            #             best_performance = np.sum(val_performances[-1])
            #             patience = max_patience
            #             _ = saver.save(self.session, save_file)
            #         else:
            #             patience -= 1
            #
            #         if patience == 0:
            #             print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
            #             break
            #     elif edge_overlap / A_orig.sum() >= stopping:  # Evaluate EO criterion
            #         print("**** EARLY STOPPING AFTER {} ITERATIONS ****".format(_it))
            #         break

            if plot_every > 0 and (_it + 1) % plot_every == 0:
                if len(disc_losses) > 10:
                    plt.plot(disc_losses[1000::100], label="Critic loss")
                    plt.plot(gen_losses[1000::100], label="Generator loss")
                else:
                    plt.plot(disc_losses, label="Critic loss")
                    plt.plot(gen_losses, label="Generator loss")
                plt.legend()
                plt.savefig('{}/iter_{}_loss_res_final.png'.format(output_directory, _it+1))
                plt.close()

        log("**** Training completed after {} iterations. ****".format(_it+1))
        plt.plot(disc_losses[1000::], label="Critic loss")
        plt.plot(gen_losses[1000::], label="Generator loss")
        plt.legend()
        plt.savefig('{}/{}_loss_res_final.png'.format(output_directory, timestr))
        plt.close()

        #### Training completed.
        log_dict = {"disc_losses": disc_losses,
                    'gen_losses': gen_losses,
                    'fake_walks': fake_walks}
        return log_dict

    def make_noise(self, shape, type="Gaussian"):
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


    def sample_gumbel(self, shape, eps=1e-20):
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


    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature)


    def gumbel_softmax(self, logits, temperature, hard=False):
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
        y = self.gumbel_softmax_sample(logits, temperature)
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
    # print(walker.walk().__next__())

    tggan = TGGAN(N=n_nodes, rw_len=rw_len,
                    t_end=t_end,
                    walk_generator=walker.walk, batch_size=batch_size, gpu_id=gpu_id,
                    use_gumbel=True,
                    disc_iters=3,
                    W_down_discriminator_size=embedding_size,
                    W_down_generator_size=embedding_size,
                    l2_penalty_generator=1e-7,
                    l2_penalty_discriminator=5e-5,
                    generator_start_layers=[40, 10],
                    generator_layers=[50, 10],
                    discriminator_layers=[40, 10],
                    # varDecoder_layers=[30],
                    temp_start=5,
                    learning_rate=lr,
                    # momentum=0.9
            )

    temperature = tggan.params['temp_start']

    tggan.session.run(tggan.init_op)

    # fake_v_inputs, fake_t0_inputs, fake_inputs_discrete, \
    # real_data, real_v_inputs, real_t0_inputs, real_lengths, \
    # disc_real, disc_fake \
    #     = tggan.session.run([
    #     tggan.fake_v_inputs, tggan.fake_t0_inputs, tggan.fake_inputs_discrete,
    #     tggan.real_data, tggan.real_v_inputs, tggan.real_t0_inputs, tggan.real_lengths,
    #     tggan.disc_real, tggan.disc_fake,
    # ], feed_dict={tggan.temp: temperature})
    #
    # tggan.session.close()
    #
    # print('fake_v_inputs:\n', np.argmax(fake_v_inputs, axis=-1))
    # print('fake_t0_inputs:\n', fake_t0_inputs)
    # print('fake_inputs_discrete: \n{}'.format(fake_inputs_discrete))
    # print('real_data:\n', real_data)
    # print('real_v_inputs:\n', np.argmax(real_v_inputs, axis=-1))
    # print('real_t0_inputs:\n', real_t0_inputs)
    # print('real_lengths:\n', real_lengths)
    # print('disc_real:\n', disc_real)
    # print('disc_fake:\n', disc_fake)
    # print('fake_lengths:', fake_lengths)
    # print('real_x_discrete:\n', real_x_discrete)
    # print('real_v1_discrete:\n', real_v1_discrete)
    # print('real_u1_discrete:\n', real_u1_discrete)
    # print('real_t_inputs:\n', real_t_inputs)
    # print('real_v_discrete:\n', real_v_discrete.shape)
    # print('real_u_discrete:\n', real_u_discrete.shape)
    # print('real_tau_inputs:\n', real_tau_inputs)

    # print('x shape:\n', x.shape)
    # print('v1 shape:\n', v1.shape)
    # print('u1 shape:\n', u1.shape)
    # print('x :\n', x)
    # print('v1 :\n', v1)
    # print('u1 :\n', u1)
    # print('mu_last shape:\n', mu_last.shape)
    # print('sigma_last shape:\n', sigma_last.shape)
    # print('t shape:\n', t.shape)
    # print('x_noise shape:\n', x_noise.shape)
    # print('v1_noise shape:\n', v1_noise.shape)
    # print('u1_noise shape:\n', u1_noise.shape)
    # print('initial_states_noise shape:\n', initial_states_noise.shape)
    # print('lstm states shape:\n', np.array(state[0]).shape, np.array(state[1]).shape, len(state))
    # print('lstm inputs shape:\n', inputs.shape)
    # print('output_bef shape:\n', output_bef.shape)
    # print('leng, x:\n', leng[0], '\nv\n', leng[1], '\nu\n', leng[2], '\nt\n', leng[3],
    #       '\nlengths\n', leng[4])
    # print('normal_outputs:\n', normal_outputs)
    # print('fake_discrete:\n', fake_discrete)
    # log('-'*40)

    max_iters = 10
    eval_every = 5
    plot_every = 5000
    n_eval_loop = 1
    transitions_per_iter = batch_size * n_eval_loop
    eval_transitions = transitions_per_iter * 100
    model_name='metro'

    log_dict = tggan.train(n_eval_loop=n_eval_loop,
                           stopping=None,
                            transitions_per_iter=transitions_per_iter, eval_transitions=eval_transitions,
                            eval_every=eval_every, plot_every=plot_every,
                            max_patience=20, max_iters=max_iters,
                            model_name=model_name, save_directory="snapshots",
                           output_directory='outputs',
                            )
    log('-'*40)