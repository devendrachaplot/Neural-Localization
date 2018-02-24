import os
import numpy as np
from utils.maze import *
from utils.localization import *


class Maze2D(object):

    def __init__(self, args):
        self.args = args
        self.test_mazes = np.load(args.test_data)
        self.test_maze_idx = 0
        return

    def reset(self):

        # Load a test maze during evaluation
        if self.args.evaluate != 0:
            maze_in_test_data = False
            maze = self.test_mazes[self.test_maze_idx]
            self.orientation = int(maze[-1])
            self.position = (int(maze[-3]), int(maze[-2]))
            self.map_design = maze[:-3].reshape(self.args.map_size,
                                                self.args.map_size)
            self.test_maze_idx += 1
        else:
            maze_in_test_data = True

        # Generate a maze
        while maze_in_test_data:
            # Generate a map design
            self.map_design = generate_map(self.args.map_size)

            # Get random initial position and orientation of the agent
            self.position = get_random_position(self.map_design)
            self.orientation = np.random.randint(4)

            maze = np.concatenate((self.map_design.flatten(),
                                   np.array(self.position),
                                   np.array([self.orientation])))

            # Make sure the maze doesn't exist in the test mazes
            if not any((maze == x).all() for x in self.test_mazes):
                # Make sure map is not symmetric
                if not (self.map_design ==
                        np.rot90(self.map_design)).all() \
                    and not (self.map_design ==
                             np.rot90(np.rot90(self.map_design))).all():
                    maze_in_test_data = False

        # Pre-compute likelihoods of all observations on the map for efficiency
        self.likelihoods = get_all_likelihoods(self.map_design)

        # Get current observation and likelihood
        curr_depth = get_depth(self.map_design, self.position,
                               self.orientation)
        curr_likelihood = self.likelihoods[int(curr_depth) - 1]

        # Posterior is just the likelihood as prior is uniform
        self.posterior = curr_likelihood

        # Renormalization of the posterior
        self.posterior /= np.sum(self.posterior)
        self.t = 0

        # next state for the policy model
        self.state = np.concatenate((self.posterior, np.expand_dims(
                                     self.map_design, axis=0)), axis=0)
        return self.state, int(curr_depth)

    def step(self, action_id):
        # Get the observation before taking the action
        curr_depth = get_depth(self.map_design, self.position,
                               self.orientation)

        # Posterior from last step is the prior for this step
        prior = self.posterior

        # Transform the prior according to the action taken
        prior = transition_function(prior, curr_depth, action_id)

        # Calculate position and orientation after taking the action
        self.position, self.orientation = get_next_state(
            self.map_design, self.position, self.orientation, action_id)

        # Get the observation and likelihood after taking the action
        curr_depth = get_depth(
            self.map_design, self.position, self.orientation)
        curr_likelihood = self.likelihoods[int(curr_depth) - 1]

        # Posterior = Prior * Likelihood
        self.posterior = np.multiply(curr_likelihood, prior)

        # Renormalization of the posterior
        self.posterior /= np.sum(self.posterior)

        # Calculate the reward
        reward = self.posterior.max()

        self.t += 1
        if self.t == self.args.max_episode_length:
            is_final = True
        else:
            is_final = False

        # next state for the policy model
        self.state = np.concatenate(
            (self.posterior, np.expand_dims(
                self.map_design, axis=0)), axis=0)

        return self.state, reward, is_final, int(curr_depth)
