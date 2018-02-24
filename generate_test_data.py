import argparse
import os
from maze2d import *

import logging


parser = argparse.ArgumentParser(description='Generate test mazes')
parser.add_argument('-n', '--num-mazes', type=int, default=1000,
                    help='Number of mazes to generate (default: 1000)')
parser.add_argument('-m', '--map-size', type=int, default=7,
                    help='''m: Size of the maze m x m (default: 7),
                            must be an odd natural number''')
parser.add_argument('-tdl', '--test-data-location', type=str,
                    default="./test_data/",
                    help='Data location (default: ./test_data/)')
parser.add_argument('-tdf', '--test-data-filename', type=str,
                    default="m7_n1000.npy",
                    help='Data location (default: m7_n1000.npy)')


if __name__ == '__main__':
    args = parser.parse_args()
    test_mazes = []

    if not os.path.exists(args.test_data_location):
        os.makedirs(args.test_data_location)

    while len(test_mazes) < args.num_mazes:
        map_design = generate_map(args.map_size)
        position = np.array(get_random_position(map_design))
        orientation = np.array([np.random.randint(4)])

        maze = np.concatenate((map_design.flatten(), position, orientation))

        # Make sure the maze doesn't exist in the test mazes already
        if not any((maze == x).all() for x in test_mazes):
            # Make sure map is not symmetric
            if not (map_design == np.rot90(map_design)).all() and \
                    not (map_design == np.rot90(np.rot90(map_design))).all():
                test_mazes.append(maze)

    filepath = os.path.join(args.test_data_location, args.test_data_filename)
    np.save(filepath, test_mazes)
