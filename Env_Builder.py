import copy
from operator import sub, add
import gym
import numpy as np
import math
import warnings
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from od_mstar3 import od_mstar
from GroupLock import Lock
from matplotlib.colors import *
from gym.envs.classic_control import rendering
import imageio
from gym import spaces


def make_gif(images, fname):
    gif = imageio.mimwrite(fname, images, subrectangles=True)
    print("wrote gif")
    return gif


def opposite_actions(action, isDiagonal=False):
    if isDiagonal:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
        raise NotImplemented
    else:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
    return checking_table[action]


def action2dir(action):
    checking_table = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    return checking_table[action]


def dir2action(direction):
    checking_table = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
    return checking_table[direction]


def tuple_plus(a, b):
    """ a + b """
    return tuple(map(add, a, b))


def tuple_minus(a, b):
    """ a - b """
    return tuple(map(sub, a, b))


def _heap(ls, max_length):
    while True:
        if len(ls) > max_length:
            ls.pop(0)
        else:
            return ls


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def getAstarDistanceMap(map: np.array, start: tuple, goal: tuple, isDiagonal: bool = False):
    """
    returns a numpy array of same dims as map with the distance to the goal from each coord
    :param map: a n by m np array, where -1 denotes obstacle
    :param start: start_position
    :param goal: goal_position
    :return: optimal distance map
    """

    def lowestF(fScore, openSet):
        # find entry in openSet with lowest fScore
        assert (len(openSet) > 0)
        minF = 2 ** 31 - 1
        minNode = None
        for (i, j) in openSet:
            if (i, j) not in fScore: continue
            if fScore[(i, j)] < minF:
                minF = fScore[(i, j)]
                minNode = (i, j)
        return minNode

    def getNeighbors(node):
        # return set of neighbors to the given node
        n_moves = 9 if isDiagonal else 5
        neighbors = set()
        for move in range(1, n_moves):  # we dont want to include 0 or it will include itself
            direction = action2dir(move)
            dx = direction[0]
            dy = direction[1]
            ax = node[0]
            ay = node[1]
            if (ax + dx >= map.shape[0] or ax + dx < 0 or ay + dy >= map.shape[
                1] or ay + dy < 0):  # out of bounds
                continue
            if map[ax + dx, ay + dy] == -1:  # collide with static obstacle
                continue
            neighbors.add((ax + dx, ay + dy))
        return neighbors

    # NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
    start, goal = goal, start
    start, goal = tuple(start), tuple(goal)
    # The set of nodes already evaluated
    closedSet = set()

    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    openSet = set()
    openSet.add(start)

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    cameFrom = dict()

    # For each node, the cost of getting from the start node to that node.
    gScore = dict()  # default value infinity

    # The cost of going from start to start is zero.
    gScore[start] = 0

    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    fScore = dict()  # default infinity

    # our heuristic is euclidean distance to goal
    heuristic_cost_estimate = lambda x, y: math.hypot(x[0] - y[0], x[1] - y[1])

    # For the first node, that value is completely heuristic.
    fScore[start] = heuristic_cost_estimate(start, goal)

    while len(openSet) != 0:
        # current = the node in openSet having the lowest fScore value
        current = lowestF(fScore, openSet)

        openSet.remove(current)
        closedSet.add(current)
        for neighbor in getNeighbors(current):
            if neighbor in closedSet:
                continue  # Ignore the neighbor which is already evaluated.

            if neighbor not in openSet:  # Discover a new node
                openSet.add(neighbor)

            # The distance from start to a neighbor
            # in our case the distance between is always 1
            tentative_gScore = gScore[current] + 1
            if tentative_gScore >= gScore.get(neighbor, 2 ** 31 - 1):
                continue  # This is not a better path.

            # This path is the best until now. Record it!
            cameFrom[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal)

            # parse through the gScores
    Astar_map = map.copy()
    for (i, j) in gScore:
        Astar_map[i, j] = gScore[i, j]
    return Astar_map


class Agent:
    """
    The agent object that contains agent's position, direction dict and position dict,
    currently only supporting 4-connected region.
    self.distance_map is None here. Assign values in upper class.
    ###########
    WARNING: direction_history[i] means the action taking from i-1 step, resulting in the state of step i,
    such that len(direction_history) == len(position_history)
    ###########
    """

    def __init__(self, isDiagonal=False):
        self._path_count = -1
        self.IsDiagonal = isDiagonal
        self.position, self.position_history, self.ID, self.direction, self.direction_history, \
        self.action_history, self.goal_pos, self.distanceMap, self.dones, self.status, self.next_goal, self.next_distanceMap \
            = None, [], None, None, [(None, None)], [(None, None)], None, None, 0, None, None, None

    def reset(self):
        self._path_count = -1
        self.position, self.position_history, self.ID, self.direction, self.direction_history, \
        self.action_history, self.goal_pos, self.distanceMap, self.dones, self.status, self.next_goal, self.next_distanceMap \
            = None, [], None, None, [(None, None)], [(None, None)], None, None, 0, None, None, None

    def move(self, pos, status=None):
        if pos is None:
            pos = self.position
        if self.position is not None:
            assert pos in [self.position,
                           tuple_plus(self.position, (0, 1)), tuple_plus(self.position, (0, -1)),
                           tuple_plus(self.position, (1, 0)), tuple_plus(self.position, (-1, 0)), ], \
                "only 1 step 1 cell allowed. Previous pos:" + str(self.position)
        self.add_history(pos, status)

    def add_history(self, position, status):
        assert len(position) == 2
        self.status = status
        self._path_count += 1
        self.position = tuple(position)
        if self._path_count != 0:
            direction = tuple_minus(position, self.position_history[-1])
            action = dir2action(direction)
            assert action in list(range(4 + 1)), \
                "direction not in actionDir, something going wrong"
            self.direction_history.append(direction)
            self.action_history.append(action)
        self.position_history.append(tuple(position))

        self.position_history = _heap(self.position_history, 30)
        self.direction_history = _heap(self.direction_history, 30)
        self.action_history = _heap(self.action_history, 30)


class World:
    """
    Include: basic world generation rules, blank map generation and collision checking.
    reset_world:
    Do not add action pruning, reward structure or any other routine for training in this class. Pls add in upper class MAPFEnv
    """

    def __init__(self, map_generator, num_agents, isDiagonal=False):
        self.num_agents = num_agents
        self.manual_world = False
        self.manual_goal = False
        self.goal_generate_distance = 2

        self.map_generator = map_generator
        self.isDiagonal = isDiagonal

        self.agents_init_pos, self.goals_init_pos = None, None
        self.reset_world()
        self.init_agents_and_goals()

    def reset_world(self):
        """
        generate/re-generate a world map, and compute its corridor map
        """

        def scan_for_agents(state_map):
            agents = {}
            for i in range(state_map.shape[0]):
                for j in range(state_map.shape[1]):
                    if state_map[i, j] > 0:
                        agentID = state_map[i, j]
                        agents.update({agentID: (i, j)})
            return agents

        self.state, self.goals_map = self.map_generator()
        # detect manual world
        if (self.state > 0).any():
            self.manual_world = True
            self.agents_init_pos = scan_for_agents(self.state)
            if self.num_agents is not None and self.num_agents != len(self.agents_init_pos.keys()):
                warnings.warn("num_agent does not match the actual agent number in manual map! "
                              "num_agent has been set to be consistent with manual map.")
            self.num_agents = len(self.agents_init_pos.keys())
            self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        else:
            assert self.num_agents is not None
            self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        # detect manual goals_map
        if self.goals_map is not None:
            self.manual_goal = True
            self.goals_init_pos = scan_for_agents(self.goals_map) if self.manual_goal else None

        else:
            self.goals_map = np.zeros([self.state.shape[0], self.state.shape[1]])

        self.corridor_map = {}
        self.restrict_init_corridor = True
        self.visited = []
        self.corridors = {}
        self.get_corridors()

    def reset_agent(self):
        """
        remove all the agents (with their travel history) and goals in the env, rebase the env into a blank one
        """
        self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        self.state[self.state > 0] = 0  # remove agents in the map

    def get_corridors(self):
        """
        in corridor_map , output = list:
            list[0] : if In corridor, corridor id , else -1 
            list[1] : If Inside Corridor = 1
                      If Corridor Endpoint = 2
                      If Free Cell Outside Corridor = 0   
                      If Obstacle = -1 
        """
        corridor_count = 1
        # Initialize corridor map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state[i, j] >= 0:
                    self.corridor_map[(i, j)] = [-1, 0]
                else:
                    self.corridor_map[(i, j)] = [-1, -1]
        # Compute All Corridors and End-points, store them in self.corridors , update corridor_map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                positions = self.blank_env_valid_neighbor(i, j)
                if (positions.count(None)) == 2 and (i, j) not in self.visited:
                    allowed = self.check_for_singular_state(positions)
                    if not allowed:
                        continue
                    self.corridors[corridor_count] = {}
                    self.corridors[corridor_count]['Positions'] = [(i, j)]
                    self.corridor_map[(i, j)] = [corridor_count, 1]
                    self.corridors[corridor_count]['EndPoints'] = []
                    self.visited.append((i, j))
                    for num in range(4):
                        if positions[num] is not None:
                            self.visit(positions[num][0], positions[num][1], corridor_count)
                    corridor_count += 1
        # Get Delta X , Delta Y for the computed corridors ( Delta= Displacement to corridor exit)       
        for k in range(1, corridor_count):
            if k in self.corridors:
                if len(self.corridors[k]['EndPoints']) == 2:
                    self.corridors[k]['DeltaX'] = {}
                    self.corridors[k]['DeltaY'] = {}
                    pos_a = self.corridors[k]['EndPoints'][0]
                    pos_b = self.corridors[k]['EndPoints'][1]
                    self.corridors[k]['DeltaX'][pos_a] = (pos_a[0] - pos_b[0])  # / (max(1, abs(pos_a[0] - pos_b[0])))
                    self.corridors[k]['DeltaX'][pos_b] = -1 * self.corridors[k]['DeltaX'][pos_a]
                    self.corridors[k]['DeltaY'][pos_a] = (pos_a[1] - pos_b[1])  # / (max(1, abs(pos_a[1] - pos_b[1])))
                    self.corridors[k]['DeltaY'][pos_b] = -1 * self.corridors[k]['DeltaY'][pos_a]
            else:
                print('Weird2')

                # Rearrange the computed corridor list such that it becomes easier to iterate over the structure
        # Basically, sort the self.corridors['Positions'] list in a way that the first element of the list is
        # adjacent to Endpoint[0] and the last element of the list is adjacent to EndPoint[1] 
        # If there is only 1 endpoint, the sorting doesn't matter since blocking is easy to compute
        for t in range(1, corridor_count):
            positions = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][0][0],
                                                      self.corridors[t]['EndPoints'][0][1])
            for position in positions:
                if position is not None and self.corridor_map[position][0] == t:
                    break
            index = self.corridors[t]['Positions'].index(position)

            if index == 0:
                pass
            if index != len(self.corridors[t]['Positions']) - 1:
                temp_list = self.corridors[t]['Positions'][0:index + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)

            elif index == len(self.corridors[t]['Positions']) - 1 and len(self.corridors[t]['EndPoints']) == 2:
                positions2 = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][1][0],
                                                           self.corridors[t]['EndPoints'][1][1])
                for position2 in positions2:
                    if position2 is not None and self.corridor_map[position2][0] == t:
                        break
                index2 = self.corridors[t]['Positions'].index(position2)
                temp_list = self.corridors[t]['Positions'][0:index2 + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index2 + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)
                self.corridors[t]['Positions'].reverse()
            else:
                if len(self.corridors[t]['EndPoints']) == 2:
                    print("Weird3")

            self.corridors[t]['StoppingPoints'] = []
            if len(self.corridors[t]['EndPoints']) == 2:
                position_first = self.corridors[t]['Positions'][0]
                position_last = self.corridors[t]['Positions'][-1]
                self.corridors[t]['StoppingPoints'].append([position_first[0], position_first[1]])
                self.corridors[t]['StoppingPoints'].append([position_last[0], position_last[1]])
            else:
                position_first = self.corridors[t]['Positions'][0]
                self.corridors[t]['StoppingPoints'].append([position[0], position[1]])
                self.corridors[t]['StoppingPoints'].append(None)
        return

    def check_for_singular_state(self, positions):
        counter = 0
        for num in range(4):
            if positions[num] is not None:
                new_positions = self.blank_env_valid_neighbor(positions[num][0], positions[num][1])
                if new_positions.count(None) in [2, 3]:
                    counter += 1
        return counter > 0

    def visit(self, i, j, corridor_id):
        positions = self.blank_env_valid_neighbor(i, j)
        if positions.count(None) in [0, 1]:
            self.corridors[corridor_id]['EndPoints'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 2]
            return
        elif positions.count(None) in [2, 3]:
            self.visited.append((i, j))
            self.corridors[corridor_id]['Positions'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 1]
            for num in range(4):
                if positions[num] is not None and positions[num] not in self.visited:
                    self.visit(positions[num][0], positions[num][1], corridor_id)
        else:
            print('Weird')

    def blank_env_valid_neighbor(self, i, j):
        possible_positions = [None, None, None, None]
        move = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        if self.state[i, j] == -1:
            return possible_positions
        else:
            for num in range(4):
                x = i + move[num][0]
                y = j + move[num][1]
                if 0 <= x < self.state.shape[0] and 0 <= y < self.state.shape[1]:
                    if self.state[x, y] != -1:
                        possible_positions[num] = (x, y)
                        continue
        return possible_positions

    def getPos(self, agent_id):
        return tuple(self.agents[agent_id].position)

    def getDone(self, agentID):
        # get the number of goals that an agent has finished
        return self.agents[agentID].dones

    def get_history(self, agent_id, path_id=None):
        """
        :param: path_id: if None, get the last step
        :return: past_pos: (x,y), past_direction: int
        """

        if path_id is None:
            path_id = self.agents[agent_id].path_count - 1 if self.agents[agent_id].path_count > 0 else 0
        try:
            return self.agents[agent_id].position_history[path_id], self.agents[agent_id].direction_history[path_id]
        except IndexError:
            print("you are giving an invalid path_id")

    def getGoal(self, agent_id):
        return tuple(self.agents[agent_id].goal_pos)

    def init_agents_and_goals(self):
        """
        place all agents and goals in the blank env. If turning on corridor population restriction, only 1 agent is
        allowed to be born in each corridor.
        """

        def corridor_restricted_init_poss(state_map, corridor_map, goal_map, id_list=None):
            """
            generate agent init positions when corridor init population is restricted
            return a dict of positions {agentID:(x,y), ...}
            """
            if id_list is None:
                id_list = list(range(1, self.num_agents + 1))

            free_space1 = list(np.argwhere(state_map == 0))
            free_space1 = [tuple(pos) for pos in free_space1]
            corridors_visited = []
            manual_positions = {}
            break_completely = False
            for idx in id_list:
                if break_completely:
                    return None
                pos_set = False
                agentID = idx
                while not pos_set:
                    try:
                        assert (len(free_space1) > 1)
                        random_pos = np.random.choice(len(free_space1))
                    except AssertionError or ValueError:
                        print('wrong agent')
                        self.reset_world()
                        self.init_agents_and_goals()
                        break_completely = True
                        if idx == id_list[-1]:
                            return None
                        break
                    position = free_space1[random_pos]
                    cell_info = corridor_map[position[0], position[1]][1]
                    if cell_info in [0, 2]:
                        if goal_map[position[0], position[1]] != agentID:
                            manual_positions.update({idx: (position[0], position[1])})
                            free_space1.remove(position)
                            pos_set = True
                    elif cell_info == 1:
                        corridor_id = corridor_map[position[0], position[1]][0]
                        if corridor_id not in corridors_visited:
                            if goal_map[position[0], position[1]] != agentID:
                                manual_positions.update({idx: (position[0], position[1])})
                                corridors_visited.append(corridor_id)
                                free_space1.remove(position)
                                pos_set = True
                        else:
                            free_space1.remove(position)
                    else:
                        print("Very Weird")
                        # print('Manual Positions' ,manual_positions)
            return manual_positions

        # no corridor population restriction
        if not self.restrict_init_corridor or (self.restrict_init_corridor and self.manual_world):
            self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            self._put_agents(list(range(1, self.num_agents + 1)), self.agents_init_pos)
        # has corridor population restriction
        else:
            check = self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            if check is not None:
                manual_positions = corridor_restricted_init_poss(self.state, self.corridor_map, self.goals_map)
                if manual_positions is not None:
                    self._put_agents(list(range(1, self.num_agents + 1)), manual_positions)

    def _put_agents(self, id_list, manual_pos=None):
        """
        put some agents in the blank env, saved history data in self.agents and self.state
        get distance map for the agents
        :param id_list: a list of agent_id
                manual_pos: a dict of manual positions {agentID: (x,y),...}
        """
        if manual_pos is None:
            # randomly init agents everywhere
            free_space = np.argwhere(np.logical_or(self.state == 0, self.goals_map == 0) == 1)
            new_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
            init_poss = [free_space[idx] for idx in new_idx]
        else:
            assert len(manual_pos.keys()) == len(id_list)
            init_poss = [manual_pos[agentID] for agentID in id_list]
        assert len(init_poss) == len(id_list)
        self.agents_init_pos = {}
        for idx, agentID in enumerate(id_list):
            self.agents[agentID].ID = agentID
            if self.state[init_poss[idx][0], init_poss[idx][1]] in [0, agentID] \
                    and self.goals_map[init_poss[idx][0], init_poss[idx][1]] != agentID:
                self.state[init_poss[idx][0], init_poss[idx][1]] = agentID
                self.agents_init_pos.update({agentID: (init_poss[idx][0], init_poss[idx][1])})
            else:
                print(self.state)
                print(init_poss)
                raise ValueError('invalid manual_pos for agent' + str(agentID) + ' at: ' + str(init_poss[idx]))
            self.agents[agentID].move(init_poss[idx])
            self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                   self.agents[agentID].goal_pos)

    def put_goals(self, id_list, manual_pos=None):
        """
        put a goal of single agent in the env, if the goal already exists, remove that goal and put a new one
        :param manual_pos: a dict of manual_pos {agentID: (x, y)}
        :param id_list: a list of agentID
        :return: an Agent object
        """

        def random_goal_pos(previous_goals=None, distance=None):
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0)
            # print(previous_goals)
            if not all(previous_goals.values()):  # they are new born agents
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals
            else:
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(self.state > 0, self.state != agentID)
                    free_spaces_for_previous_goal = np.logical_or(free_on_agents, free_for_all)
                    if distance > 0:
                        previous_x, previous_y = previous_goals[agentID]
                        x_lower_bound = (previous_x - distance) if (previous_x - distance) > 0 else 0
                        x_upper_bound = previous_x + distance + 1
                        y_lower_bound = (previous_y - distance) if (previous_x - distance) > 0 else 0
                        y_upper_bound = previous_y + distance + 1
                        free_spaces_for_previous_goal[x_lower_bound:x_upper_bound, y_lower_bound:y_upper_bound] = False
                    free_spaces_for_previous_goal = list(np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        init_idx = np.random.choice(len(free_spaces_for_previous_goal))
                        init_pos = free_spaces_for_previous_goal[init_idx]
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print('wrong goal')
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals

        previous_goals = {agentID: self.agents[agentID].goal_pos for agentID in id_list}
        if manual_pos is None:
            new_goals = random_goal_pos(previous_goals, distance=self.goal_generate_distance)
        else:
            new_goals = manual_pos
        if new_goals is not None:  # recursive breaker
            refresh_distance_map = False
            for agentID in id_list:
                if self.state[new_goals[agentID][0], new_goals[agentID][1]] >= 0:
                    if self.agents[agentID].next_goal is None:  # no next_goal to use
                        # set goals_map
                        self.goals_map[new_goals[agentID][0], new_goals[agentID][1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = (new_goals[agentID][0], new_goals[agentID][1])
                        # set agent.next_goal
                        new_next_goals = random_goal_pos(new_goals, distance=self.goal_generate_distance)
                        if new_next_goals is None:
                            return None
                        self.agents[agentID].next_goal = (new_next_goals[agentID][0], new_next_goals[agentID][1])
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                    else:  # use next_goal as new goal
                        # set goals_map
                        self.goals_map[self.agents[agentID].next_goal[0], self.agents[agentID].next_goal[1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = self.agents[agentID].next_goal
                        # set agent.next_goal
                        self.agents[agentID].next_goal = (
                            new_goals[agentID][0], new_goals[agentID][1])  # store new goal into next_goal
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                else:
                    print(self.state)
                    print(self.goals_map)
                    raise ValueError('invalid manual_pos for goal' + str(agentID) + ' at: ', str(new_goals[agentID]))
                if previous_goals[agentID] is not None:  # it has a goal!
                    if previous_goals[agentID] != self.agents[agentID].position:
                        print(self.state)
                        print(self.goals_map)
                        print(previous_goals)
                        raise RuntimeError("agent hasn't finished its goal but asking for a new goal!")

                    refresh_distance_map = True

                # compute distance map
                self.agents[agentID].next_distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].goal_pos,
                                                                            self.agents[agentID].next_goal)
                if refresh_distance_map:
                    self.agents[agentID].distanceMap = getAstarDistanceMap(self.state, self.agents[agentID].position,
                                                                           self.agents[agentID].goal_pos)
            return 1
        else:
            return None

    def CheckCollideStatus(self, movement_dict):
        """
        WARNING: ONLY NON-DIAGONAL IS IMPLEMENTED
        return collision status and predicted next positions, do not move agent directly
        return:
         2: (only in oneShot mode) action not executed, agents has done its target and has been removed from the env.
         1: action executed, and agents standing on its goal.
         0: action executed
        -1: collision with env (obstacles, out of bound)
        -2: collision with robot, swap
        -3: collision with robot, cell-wise
        """

        if self.isDiagonal is True:
            raise NotImplemented
        Assumed_newPos_dict = {}
        newPos_dict = {}
        status_dict = {agentID: None for agentID in range(1, self.num_agents + 1)}
        not_checked_list = list(range(1, self.num_agents + 1))

        # detect env collision
        for agentID in range(1, self.num_agents + 1):
            direction_vector = action2dir(movement_dict[agentID])
            newPos = tuple_plus(self.getPos(agentID), direction_vector)
            Assumed_newPos_dict.update({agentID: newPos})
            if newPos[0] < 0 or newPos[0] > self.state.shape[0] or newPos[1] < 0 \
                    or newPos[1] > self.state.shape[1] or self.state[newPos] == -1:
                status_dict[agentID] = -1
                newPos_dict.update({agentID: self.getPos(agentID)})
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
                # collide, stay at the same place

        # detect swap collision

        for agentID in copy.deepcopy(not_checked_list):
            collided_ID = self.state[Assumed_newPos_dict[agentID]]
            if collided_ID != 0 and Assumed_newPos_dict[agentID] != self.getGoal(
                    agentID):  # some one is standing on the assumed pos
                if Assumed_newPos_dict[collided_ID] == self.getPos(agentID):  # he wants to swap
                    if status_dict[agentID] is None:
                        status_dict[agentID] = -2
                        newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                        Assumed_newPos_dict[agentID] = self.getPos(agentID)
                        not_checked_list.remove(agentID)
                    if status_dict[collided_ID] is None:
                        status_dict[collided_ID] = -2
                        newPos_dict.update({collided_ID: self.getPos(collided_ID)})  # stand still
                        Assumed_newPos_dict[collided_ID] = self.getPos(collided_ID)
                        not_checked_list.remove(collided_ID)

        # detect cell-wise collision
        for agentID in copy.deepcopy(not_checked_list):
            other_agents_dict = copy.deepcopy(Assumed_newPos_dict)
            other_agents_dict.pop(agentID)
            ignore_goal_agents_dict = copy.deepcopy(newPos_dict)
            for agent in range(1, self.num_agents + 1):
                if agent != agentID:
                    if Assumed_newPos_dict[agent] == self.getGoal(agent):
                        other_agents_dict.pop(agent)
                        try:
                            ignore_goal_agents_dict.pop(agent)
                        except:
                            pass
            if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos:
                continue
            if Assumed_newPos_dict[agentID] in ignore_goal_agents_dict.values():
                status_dict[agentID] = -3
                newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
            elif Assumed_newPos_dict[agentID] in other_agents_dict.values():
                other_coming_agents = get_key(Assumed_newPos_dict, Assumed_newPos_dict[agentID])
                other_coming_agents.remove(agentID)
                # if the agentID is the biggest among all other coming agents,
                # allow it to move. Else, let it stand still
                if agentID < min(other_coming_agents):
                    status_dict[agentID] = 1 if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
                    newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
                    not_checked_list.remove(agentID)
                else:
                    status_dict[agentID] = -3
                    newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                    Assumed_newPos_dict[agentID] = self.getPos(agentID)
                    not_checked_list.remove(agentID)

        # the rest are valid actions
        for agentID in copy.deepcopy(not_checked_list):
            status_dict[agentID] = 1 if Assumed_newPos_dict[agentID] == self.agents[agentID].goal_pos else 0
            newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
            not_checked_list.remove(agentID)
        assert not not_checked_list

        return status_dict, newPos_dict

class MAPFEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, observer, map_generator, num_agents,
                 IsDiagonal=False,isOneShot=False):
        self.observer = observer
        self.map_generator = map_generator
        self.viewer = None

        self.isOneShot = isOneShot
        self.num_agents = num_agents
        self.IsDiagonal = IsDiagonal
        self.set_world()
        self.obs_size = self.observer.observation_size
        self.isStandingOnGoal = {i: False for i in range(1, self.num_agents + 1)}

        self.individual_rewards = {i: 0 for i in range(1, self.num_agents + 1)}
        self.done = False
        self.GIF_frame = []
        if IsDiagonal:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])

        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = -0.3, 5., -2.

    def getObstacleMap(self):
        return (self.world.state == -1).astype(int)

    def getGoals(self):
        return {i: self.world.agents[i].goal_pos for i in range(1, self.num_agents + 1)}

    def getStatus(self):
        return {i: self.world.agents[i].status for i in range(1, self.num_agents + 1)}

    def getPositions(self):
        return {i: self.world.agents[i].position for i in range(1, self.num_agents + 1)}

    def getLastMovements(self):
        return {i: self.world.agents[i].position_history(-1) for i in range(1, self.num_agents + 1)}

    def set_world(self):

        self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)

    def _reset(self, *args, **kwargs):
        raise NotImplementedError

    def isInCorridor(self, agentID):
        """
        :param agentID: start from 1 not 0!
        :return: isIn: bool, corridor_ID: int
        """
        agent_pos = self.world.getPos(agentID)
        if self.world.corridor_map[(agent_pos[0], agent_pos[1])][1] in [-1, 2]:
            return False, None
        else:
            return True, self.world.corridor_map[(agent_pos[0], agent_pos[1])][0]

    def _observe(self, handles=None):
        """
        Returns Dict of observation {agentid:[], ...}
        """
        if handles is None:
            self.obs_dict = self.observer.get_many(list(range(1, self.num_agents + 1)))
        elif handles in list(range(1, self.num_agents + 1)):
            self.obs_dict = self.observer.get_many([handles])
        elif set(handles) == set(handles) & set(list(range(1, self.num_agents + 1))):
            self.obs_dict = self.observer.get_many(handles)
        else:
            raise ValueError("Invalid agent_id given")
        return self.obs_dict

    def step_all(self, movement_dict):
        """
        The new goal will be generated at the FIRST step it remains on its goal.

        :param movement_dict: {agentID_starting_from_1: action:int 0-4, ...}
                              unmentioned agent will be considered as taking standing still
        :return: obs_of_all:dict, reward_of_single_step:dict
        """

        for agentID in range(1, self.num_agents + 1):
            if self.world.getDone(agentID) > 0 and self.isOneShot:
                movement_dict.update({agentID: 0})
            if agentID not in movement_dict.keys() :
                movement_dict.update({agentID: 0})
            else:
                assert movement_dict[agentID] in list(range(5)) if self.IsDiagonal else list(range(9)), \
                    'action not in action space'

        status_dict, newPos_dict = self.world.CheckCollideStatus(movement_dict)
        self.world.state[self.world.state > 0] = 0  # remove agents in the map
        put_goal_list = []
        self.done = True
        for agentID in range(1, self.num_agents + 1):
            if self.isOneShot and self.world.getDone(agentID) > 0:
                continue

            self.done = False
            newPos = newPos_dict[agentID]
            if self.isOneShot:
                if status_dict[agentID] not in [1, 2]:
                    self.world.state[newPos] = agentID
                # else: don't place agents on state map
            else:
                self.world.state[newPos] = agentID

            self.world.agents[agentID].move(newPos, status_dict[agentID])
            self.give_moving_reward(agentID)
            if status_dict[agentID] == 1:
                if not self.isOneShot:
                    put_goal_list.append(agentID)
                else:
                    if self.world.state[newPos] == 0:
                        self.world.state[newPos] = 0
                    self.world.agents[agentID].status = 2  # status=2 means done and removed from the env
                    self.world.goals_map[newPos] = 0
        free_agents = list(range(1, self.num_agents + 1))

        if put_goal_list and not self.isOneShot:
            self.world.put_goals(put_goal_list)

        return self._observe(free_agents), self.individual_rewards

    def give_moving_reward(self, agentID):
        raise NotImplementedError

    def listValidActions(self, agent_ID, agent_obs):
        raise NotImplementedError

    def expert_until_first_goal(self, inflation=2.0, time_limit=180.0):
        world = self.getObstacleMap()
        start_positions = []
        goals = []
        start_positions_dir = self.getPositions()
        goals_dir = self.getGoals()
        for i in range(1, self.world.num_agents + 1):
            start_positions.append(start_positions_dir[i])
            goals.append(goals_dir[i])
        mstar_path = None
        try:
            mstar_path = od_mstar.find_path(world, start_positions, goals,
                                            inflation=inflation, time_limit=time_limit)
        except OutOfTimeError:
            # M* timed out
            print("timeout")
        except NoSolutionError:
            print("nosol????")
        return mstar_path

    def _add_rendering_entry(self, entry, permanent=False):
        if permanent:
            self.viewer.add_geom(entry)
        else:
            self.viewer.add_onetime(entry)

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800):

        def painter(state_map, agents_dict, goals_dict):
            def initColors(num_agents):
                c = {a + 1: hsv_to_rgb(np.array([a / float(num_agents), 1, 1])) for a in range(num_agents)}
                return c

            def create_rectangle(x, y, width, height, fill):
                ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
                rect = rendering.FilledPolygon(ps)
                rect.set_color(fill[0], fill[1], fill[2])
                rect.add_attr(rendering.Transform())
                return rect

            def drawStar(centerX, centerY, diameter, numPoints, color):
                entry_list = []
                outerRad = diameter // 2
                innerRad = int(outerRad * 3 / 8)
                # fill the center of the star
                angleBetween = 2 * math.pi / numPoints  # angle between star points in radians
                for i in range(numPoints):
                    # p1 and p3 are on the inner radius, and p2 is the point
                    pointAngle = math.pi / 2 + i * angleBetween
                    p1X = centerX + innerRad * math.cos(pointAngle - angleBetween / 2)
                    p1Y = centerY - innerRad * math.sin(pointAngle - angleBetween / 2)
                    p2X = centerX + outerRad * math.cos(pointAngle)
                    p2Y = centerY - outerRad * math.sin(pointAngle)
                    p3X = centerX + innerRad * math.cos(pointAngle + angleBetween / 2)
                    p3Y = centerY - innerRad * math.sin(pointAngle + angleBetween / 2)
                    # draw the triangle for each tip.
                    poly = rendering.FilledPolygon([(p1X, p1Y), (p2X, p2Y), (p3X, p3Y)])
                    poly.set_color(color[0], color[1], color[2])
                    poly.add_attr(rendering.Transform())
                    entry_list.append(poly)
                return entry_list

            def create_circle(x, y, diameter, world_size, fill, resolution=20):
                c = (x + world_size / 2, y + world_size / 2)
                dr = math.pi * 2 / resolution
                ps = []
                for i in range(resolution):
                    x = c[0] + math.cos(i * dr) * diameter / 2
                    y = c[1] + math.sin(i * dr) * diameter / 2
                    ps.append((x, y))
                circ = rendering.FilledPolygon(ps)
                circ.set_color(fill[0], fill[1], fill[2])
                circ.add_attr(rendering.Transform())
                return circ

            assert len(goals_dict) == len(agents_dict)
            num_agents = len(goals_dict)
            world_shape = state_map.shape
            world_size = screen_width / max(*world_shape)
            colors = initColors(num_agents)
            if self.viewer is None:
                self.viewer = rendering.Viewer(screen_width, screen_height)
                rect = create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6))
                self._add_rendering_entry(rect, permanent=True)
                for i in range(world_shape[0]):
                    start = 0
                    end = 1
                    scanning = False
                    write = False
                    for j in range(world_shape[1]):
                        if state_map[i, j] != -1 and not scanning:  # free
                            start = j
                            scanning = True
                        if (j == world_shape[1] - 1 or state_map[i, j] == -1) and scanning:
                            end = j + 1 if j == world_shape[1] - 1 else j
                            scanning = False
                            write = True
                        if write:
                            x = i * world_size
                            y = start * world_size
                            rect = create_rectangle(x, y, world_size, world_size * (end - start), (1, 1, 1))
                            self._add_rendering_entry(rect, permanent=True)
                            write = False
            for agent in range(1, num_agents + 1):
                i, j = agents_dict[agent]
                x = i * world_size
                y = j * world_size
                try:
                    color = colors[agent]
                except:
                    continue
                rect = create_rectangle(x, y, world_size, world_size, color)
                self._add_rendering_entry(rect)

                i, j = goals_dict[agent]
                x = i * world_size
                y = j * world_size
                color = colors[agent]
                circ = create_circle(x, y, world_size, world_size, color)
                self._add_rendering_entry(circ)
                if agents_dict[agent][0] == goals_dict[agent][0] and agents_dict[agent][1] == goals_dict[agent][1]:
                    color = (0, 0, 0)
                    circ = create_circle(x, y, world_size, world_size, color)
                    self._add_rendering_entry(circ)
            # if self.action_probs is not None:
            #     n_moves = 9 if self.IsDiagonal else 5
            #     for agent in range(1, num_agents + 1):
            #         # take the a_dist from the given data and draw it on the frame
            #         a_dist = self.action_probs[agent - 1]
            #         if a_dist is not None:
            #             for m in range(n_moves):
            #                 dx, dy = action2dir(m)
            #                 x = (agents_dict(agent)[0] + dx) * world_size
            #                 y = (agents_dict(agent)[1] + dy) * world_size
            #                 circ = create_circle(x, y, world_size, world_size, (0, 0, 0))
            #                 self._add_rendering_entry(circ)
            result = self.viewer.render(return_rgb_array=1)
            return result

        frame = painter(self.world.state, self.getPositions(), self.getGoals())
        return frame


if __name__ == "__main__":
    from PRIMAL2Observer import PRIMAL2Observer
    from Map_Generator import *
    from PRIMAL2Env import PRIMAL2Env
    import numpy as np
    from tqdm import tqdm

    for _ in tqdm(range(2000)):
        n_agents = np.random.randint(low=25, high=30)
        env = PRIMAL2Env(num_agents=n_agents,
                          observer=PRIMAL2Observer(observation_size=3),
                          map_generator=maze_generator(env_size=(10, 30),
                                                       wall_components=(3, 8), obstacle_density=(0.5, 0.7)),
                          IsDiagonal=False)
        for agentID in range(1, n_agents + 1):
            pos = env.world.agents[agentID].position
            goal = env.world.agents[agentID].goal_pos
            assert agentID == env.world.state[pos]
            assert agentID == env.world.goals_map[goal]
        assert len(np.argwhere(env.world.state > 0)) == n_agents
        assert len(np.argwhere(env.world.goals_map > 0)) == n_agents
