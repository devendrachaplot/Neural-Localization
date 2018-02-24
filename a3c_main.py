import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import signal
import torch
import torch.multiprocessing as mp

from maze2d import *
from model import *
from a3c_train import train
from a3c_test import test

import logging


parser = argparse.ArgumentParser(description='Active Neural Localization')

# Environment arguments
parser.add_argument('-l', '--max-episode-length', type=int,
                    default=30, metavar='L',
                    help='maximum length of an episode (default: 30)')
parser.add_argument('-m', '--map-size', type=int, default=7,
                    help='''m: Size of the maze m x m (default: 7),
                            must be an odd natural number''')

# A3C and model arguments
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--num-iters', type=int, default=1000000, metavar='NS',
                    help='''number of training iterations per training thread
                            (default: 10000000)''')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-n', '--num-processes', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 8)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--hist-size', type=int, default=5,
                    help='action history size (default: 5)')
parser.add_argument('--load', type=str, default="0",
                    help='model path to load, 0 to not reload (default: 0)')
parser.add_argument('-e', '--evaluate', type=int, default=0,
                    help='0:Train, 1:Evaluate on test data (default: 0)')
parser.add_argument('-d', '--dump-location', type=str, default="./saved/",
                    help='path to dump models and log (default: ./saved/)')
parser.add_argument('-td', '--test-data', type=str,
                    default="./test_data/m7_n1000.npy",
                    help='''Test data filepath
                            (default: ./test_data/m7_n1000.npy)''')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if not os.path.exists(args.dump_location):
        os.makedirs(args.dump_location)

    logging.basicConfig(
        filename=args.dump_location +
        'train.log',
        level=logging.INFO)

    assert args.evaluate == 0 or args.num_processes == 0, \
        "Can't train while evaluating, either n=0 or e=0"

    shared_model = Localization_2D_A3C(args)

    if args.load != "0":
        shared_model.load_state_dict(torch.load(args.load))
    shared_model.share_memory()

    signal.signal(signal.SIGINT, signal.signal(signal.SIGINT, signal.SIG_IGN))
    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping training. " +
              "Best model stored at {}model_best".format(args.dump_location))
        for p in processes:
            p.terminate()
