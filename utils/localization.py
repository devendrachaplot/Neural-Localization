import os
import numpy as np

from .maze import *


def transition_function(belief_map, depth, action):
    (o, m, n) = belief_map.shape
    if action == 'TURN_RIGHT' or action == 1:
        belief_map = np.append(belief_map,
                               belief_map[0, :, :]
                               ).reshape(o + 1, m, n)
        belief_map = np.delete(belief_map, 0, axis=0)
    elif action == "TURN_LEFT" or action == 0:
        belief_map = np.insert(belief_map, 0,
                               belief_map[-1, :, :],
                               axis=0).reshape(o + 1, m, n)
        belief_map = np.delete(belief_map, -1, axis=0)
    elif action == "MOVE_FORWARD" or action == 2:
        if depth != 1:
            new_belief = np.zeros(belief_map.shape)
            for orientation in range(belief_map.shape[0]):
                B = belief_map[orientation]
                Bcap = shift_belief(B, orientation)
                new_belief[orientation, :, :] = Bcap
            belief_map = new_belief
    return belief_map


def get_all_likelihoods(map_design):
    num_orientations = 4
    all_likelihoods = np.zeros(
        [map_design.shape[0] - 2, num_orientations,
         map_design.shape[0], map_design.shape[1]])
    for orientation in range(num_orientations):
        for i, element in np.ndenumerate(all_likelihoods[0, orientation]):
            depth = get_depth(map_design, i, orientation)
            if depth > 0:
                all_likelihoods[int(depth) - 1][orientation][i] += 1
    return all_likelihoods


def shift_belief(B, orientation):
    if orientation == 0 or orientation == "east":
        Bcap = np.insert(
            B, 0, np.zeros(
                B.shape[1]), axis=1).reshape(
            B.shape[0], B.shape[1] + 1)
        Bcap = np.delete(Bcap, -1, axis=1)
    elif orientation == 2 or orientation == "west":
        Bcap = np.append(B, np.zeros([B.shape[1], 1]), axis=1).reshape(
            B.shape[0], B.shape[1] + 1)
        Bcap = np.delete(Bcap, 0, axis=1)
    elif orientation == 1 or orientation == "north":
        Bcap = np.append(B, np.zeros([1, B.shape[1]]), axis=0).reshape(
            B.shape[0] + 1, B.shape[1])
        Bcap = np.delete(Bcap, 0, axis=0)
    elif orientation == 3 or orientation == "south":
        Bcap = np.insert(
            B, 0, np.zeros(
                B.shape[1]), axis=0).reshape(
            B.shape[0] + 1, B.shape[1])
        Bcap = np.delete(Bcap, -1, axis=0)
    else:
        assert False, "Invalid orientation"
    return Bcap
