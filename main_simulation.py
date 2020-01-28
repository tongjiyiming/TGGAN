from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('Agg')

from main import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TGGAN paper")
    assert tf.__version__.startswith('1.14.0')

    models = ['tggan', 'graphrnn', 'graphvae', 'netgan', 'RNN', 'dsbm', 'markovian']
    parser.add_argument("-m", "--model", default="tggan", type=str,
                        help="one of: {}".format(", ".join(sorted(models))))
    parser.add_argument("-re", "--runEvaluation", default=False, type=bool,
                        help="if this run should run all evaluations")
    datasets = ['simulation', 'metro', 'auth']
    parser.add_argument("-d", "--dataset", default="simulation", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-ui", "--userid", default=0, type=int,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-f", "--file", default="data/auth_user_0.txt", type=str,
                        help="file path of data in format [[d, i, j, t], ...]")
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

    # DeepTemporalWalk
    parser.add_argument("-bs", "--batch_size", default=32, type=int,
                        help="random walks batch size in DeepTemporalWalk")
    parser.add_argument("-lr", "--learningrate", default=0.0003, type=float,
                        help="if this run should run all evaluations")
    parser.add_argument("-rl", "--rw_len", default=10, type=int,
                        help="random walks maximum length in DeepTemporalWalk")
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
    parser.add_argument("-ne", "--n_eval_loop", default=40, type=int,
                        help="number of walk loops")

    # parser.add_argument("-mi", "--max_iters", default=3000, type=int,
    #                     help="max iterations")
    # parser.add_argument("-ev", "--eval_every", default=3000, type=int,
    #                     help="evaluation interval of epochs")
    # parser.add_argument("-pe", "--plot_every", default=3000, type=int,
    #                     help="plot generated graph interval of epochs")
    parser.add_argument("-mi", "--max_iters", default=2, type=int,
                        help="max iterations")
    parser.add_argument("-ev", "--eval_every", default=1, type=int,
                        help="evaluation interval of epochs")
    parser.add_argument("-pe", "--plot_every", default=1, type=int,
                        help="plot generated graph interval of epochs")
    parser.add_argument("-te", "--is_test", default=True, type=bool,
                        help="if this is a testing period.")

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
    else:
        node_list = [100, 500, 2500]
    for n_nodes in node_list:
        args.numberNode = n_nodes
        args.numberSamples = 1000
        tf.compat.v1.reset_default_graph()
        main(args)
    log('finish execution')
