import numpy as np
import random
import sys
from Env_Builder import World


def isConnected(world0):
    sys.setrecursionlimit(10000)
    world0 = world0.copy()

    def firstFree(world0):
        for x in range(world0.shape[0]):
            for y in range(world0.shape[1]):
                if world0[x, y] == 0:
                    return x, y

    def floodfill(world, i, j):
        sx, sy = world.shape[0], world.shape[1]
        if i < 0 or i >= sx or j < 0 or j >= sy:  # out of bounds, return
            return
        if world[i, j] == -1: return
        world[i, j] = -1
        floodfill(world, i + 1, j)
        floodfill(world, i, j + 1)
        floodfill(world, i - 1, j)
        floodfill(world, i, j - 1)

    i, j = firstFree(world0)
    floodfill(world0, i, j)
    if np.any(world0 == 0):
        return False
    else:
        return True


def GetConnectedRegion(world, regions_dict, x, y):
    sys.setrecursionlimit(1000000)
    '''returns a list of tuples of connected squares to the given tile
    this is memorized with a dict'''
    if (x, y) in regions_dict:
        return regions_dict[(x, y)]
    visited = set()
    sx, sy = world.shape[0], world.shape[1]
    work_list = [(x, y)]
    while len(work_list) > 0:
        (i, j) = work_list.pop()
        if i < 0 or i >= sx or j < 0 or j >= sy:  # out of bounds, return
            continue
        if world[i, j] == -1:
            continue  # crashes
        if world[i, j] > 0:
            regions_dict[(i, j)] = visited
        if (i, j) in visited:
            continue
        visited.add((i, j))
        work_list.append((i + 1, j))
        work_list.append((i, j + 1))
        work_list.append((i - 1, j))
        work_list.append((i, j - 1))
    regions_dict[(x, y)] = visited
    return visited


def maze_generator(env_size=(10, 70), wall_components=(1, 8), obstacle_density=None,
                   go_straight=0.8):
    # allow both one-dimensional args and tuple args
    if obstacle_density is None:
        obstacle_density = (0, 1)
    if isinstance(env_size, tuple):
        assert len(env_size) == 2 and min(env_size) > 5
        world_size = np.random.randint(min(env_size), max(env_size))
    else:
        world_size = env_size
    if isinstance(wall_components, tuple):
        assert len(wall_components) == 2
        num_components = np.random.randint(low=min(wall_components), high=max(wall_components) + 1)
    else:
        num_components = wall_components
    if isinstance(obstacle_density, tuple):
        assert len(obstacle_density) == 2
        obs_dense = np.random.uniform(min(obstacle_density), max(obstacle_density))
    else:
        obs_dense = obstacle_density

    def maze(h, w, total_density=0):
        assert h > 0 and w > 0, "You are giving non-positive width and height"
        shape = ((h // 2) * 2 + 3, (w // 2) * 2 + 3)
        # Adjust num_components and density relative to maze world_size
        density = int(shape[0] * shape[1] * total_density // num_components) if num_components != 0 else 0

        # Build actual maze
        Z = np.zeros(shape, dtype='int')
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        for i in range(density):
            x, y = np.random.randint(0, shape[1] // 2) * 2, np.random.randint(0, shape[
                0] // 2) * 2  # pick a random position
            Z[y, x] = 1
            last_dir = 0
            for j in range(num_components):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    if last_dir == 0:
                        y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                        if Z[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            Z[y_, x_] = 1
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
                    else:
                        index_F = -1
                        index_B = -1
                        diff = []
                        for k in range(len(neighbours)):
                            diff.append((neighbours[k][0] - y, neighbours[k][1] - x))
                            if diff[k] == last_dir:
                                index_F = k
                            elif diff[k][0] + last_dir[0] == 0 and diff[k][1] + last_dir[1] == 0:
                                index_B = k
                        assert (index_B >= 0)
                        if index_F + 1:
                            p = (1 - go_straight) * np.ones(len(neighbours)) / (len(neighbours) - 2)
                            p[index_B] = 0
                            p[index_F] = go_straight
                            # assert(p.sum() == 1)
                        else:
                            if len(neighbours) == 1:
                                p = 1
                            else:
                                p = np.ones(len(neighbours)) / (len(neighbours) - 1)
                                p[index_B] = 0
                            assert (p.sum() == 1)

                        I = np.random.choice(range(len(neighbours)), p=p)
                        (y_, x_) = neighbours[I]
                        if Z[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            Z[y_, x_] = 1
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
        return Z

    def generator():
        world = -maze(int(world_size), int(world_size),
                      total_density=obs_dense,
                      ).astype(int)
        world = np.array(world)
        return world, None

    return generator


def DummyGenerator():
    state_map = np.array([[1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
    goals_map = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1]])

    def generator():
        return state_map, goals_map

    return generator


def manual_generator(state_map, goals_map=None):
    state_map = np.array(state_map)

    assert state_map is not None
    assert len(state_map.shape) == 2
    assert min(state_map.shape) >= 5
    if goals_map is not None:
        goals_map = np.array(goals_map)
        assert goals_map.shape[0] == state_map.shape[0] and goals_map.shape[1] == state_map.shape[1]

    def generator():
        return state_map, goals_map

    return generator


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    print("testing randomized map generation")
    plt.ion()
    for _ in range(1000):
        generator = maze_generator()
        world = generator()
        plt.imshow(world)
        plt.pause(0.1)
    plt.ioff()
    plt.show()
