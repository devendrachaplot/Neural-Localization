import numpy as np
import numpy.random as npr
import itertools


def generate_map(maze_size, decimation=0.):
    """
    Generates a maze using Kruskal's algorithm
    https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    m = (maze_size - 1)//2
    n = (maze_size - 1)//2

    maze = np.ones((maze_size, maze_size))
    for i, j in list(itertools.product(range(m), range(n))):
        maze[2*i+1, 2*j+1] = 0
    m = m - 1
    L = np.arange(n+1)
    R = np.arange(n)
    L[n] = n-1

    while m > 0:
        for i in range(n):
            j = L[i+1]
            if (i != j and npr.randint(3) != 0):
                R[j] = R[i]
                L[R[j]] = j
                R[i] = i + 1
                L[R[i]] = i
                maze[2*(n-m)-1, 2*i+2] = 0
            if (i != L[i] and npr.randint(3) != 0):
                L[R[i]] = L[i]
                R[L[i]] = R[i]
                L[i] = i
                R[i] = i
            else:
                maze[2*(n-m), 2*i+1] = 0
        m -= 1

    for i in range(n):
        j = L[i+1]
        if (i != j and (i == L[i] or npr.randint(3) != 0)):
            R[j] = R[i]
            L[R[j]] = j
            R[i] = i+1
            L[R[i]] = i
            maze[2*(n-m)-1, 2*i+2] = 0
        L[R[i]] = L[i]
        R[L[i]] = R[i]
        L[i] = i
        R[i] = i
    return maze


def get_depth(map_design, position, orientation):
    m, n = map_design.shape
    depth = 0
    new_tuple = position
    while(compare_tuples(new_tuple, tuple([m - 1, n - 1])) and
            compare_tuples(tuple([0, 0]), new_tuple)):
        if map_design[new_tuple] != 0:
            break
        else:
            new_tuple = get_tuple(new_tuple, orientation)
            depth += 1
    return depth


def get_next_state(map_design, position, orientation, action):
    m, n = map_design.shape
    if action == 'TURN_LEFT' or action == 0:
        orientation = (orientation + 1) % 4
    elif action == "TURN_RIGHT" or action == 1:
        orientation = (orientation - 1) % 4
    elif action == "MOVE_FORWARD" or action == 2:
        new_tuple = get_tuple(position, orientation)
        if compare_tuples(new_tuple, tuple([m - 1, n - 1])) and \
           compare_tuples(tuple([0, 0]), new_tuple) and \
           map_design[new_tuple] == 0:
            position = new_tuple
    return position, orientation


def get_random_position(map_design):
    m, n = map_design.shape
    while True:
        index = tuple([np.random.randint(m), np.random.randint(n)])
        if map_design[index] == 0:
            return index


def get_tuple(i, orientation):
    if orientation == 0 or orientation == "east":
        new_tuple = tuple([i[0], i[1] + 1])
    elif orientation == 2 or orientation == "west":
        new_tuple = tuple([i[0], i[1] - 1])
    elif orientation == 1 or orientation == "north":
        new_tuple = tuple([i[0] - 1, i[1]])
    elif orientation == 3 or orientation == "south":
        new_tuple = tuple([i[0] + 1, i[1]])
    else:
        assert False, "Invalid orientation"
    return new_tuple


def compare_tuples(a, b):
    """
    Returns true if all elements of a are less than
    or equal to b
    """
    assert len(a) == len(b), "Unequal lengths of tuples for comparison"
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
    return True


if __name__ == '__main__':
    print(generate_map(7))
