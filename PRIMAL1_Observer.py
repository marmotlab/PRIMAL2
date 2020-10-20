from Observer_Builder import ObservationBuilder
import numpy as np
import copy
from Env_Builder import *
from PRIMAL2_Env import PRIMAL2_Env


class PRIMAL1_observer(ObservationBuilder):
    """
    obs shape: (4 * obs_size * obs_size )
    map order: [poss_map, goal_map, goals_map obs_map]
    for listvalidaction: [blocking_map, deltax_map, deltay_map]
    """

    def __init__(self, observation_size=11, num_future_steps=3):
        super(PRIMAL1_observer, self).__init__()
        self.observation_size = observation_size
        self.num_future_steps = num_future_steps
        self.NUM_CHANNELS = 4

    def set_world(self, world):
        super().set_env(world)

    def get_next_positions(self, agent_id):
        agent_pos = self.world.getPos(agent_id)
        positions = []
        current_pos = [agent_pos[0], agent_pos[1]]
        next_positions = self.world.blank_env_valid_neighbor(current_pos[0], current_pos[1])
        for position in next_positions:
            if position is not None and position != agent_pos:
                positions.append([position[0], position[1]])
                next_next_positions = self.world.blank_env_valid_neighbor(position[0], position[1])
                for pos in next_next_positions:
                    if pos is not None and pos not in positions and pos != agent_pos:
                        positions.append([pos[0], pos[1]])

        return positions

    def _get(self, agent_id, all_astar_maps):

        assert (agent_id > 0)
        top_left = (self.world.getPos(agent_id)[0] - self.observation_size // 2,
                    self.world.getPos(agent_id)[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        centre = (self.observation_size - 1) / 2
        obs_shape = (self.observation_size, self.observation_size)

        goal_map = np.zeros(obs_shape)
        poss_map = np.zeros(obs_shape)
        goals_map = np.zeros(obs_shape)
        obs_map = np.zeros(obs_shape)
        astar_map = np.zeros([self.num_future_steps, self.observation_size, self.observation_size])
        astar_map_unpadded = np.zeros([self.num_future_steps, self.world.state.shape[0], self.world.state.shape[1]])
        pathlength_map = np.zeros(obs_shape)
        deltax_map = np.zeros(obs_shape)
        deltay_map = np.zeros(obs_shape)
        blocking_map = np.zeros(obs_shape)

        # concatenate all_astar maps
        for step in range(self.num_future_steps):
            other_agents = list(range(1, self.world.num_agents + 1))
            other_agents.remove(agent_id)
            for other in other_agents:
                astar_map_unpadded[step, :, :] = all_astar_maps[other][step, :, :] + astar_map_unpadded[step, :, :]

        # original layers from PRIMAL1
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):
                for step in range(self.num_future_steps):
                    if 0 <= i < self.world.state.shape[0] and 0 <= j < self.world.state.shape[1]:
                        astar_map[step, i - top_left[0], j - top_left[1]] = astar_map_unpadded[step, i, j]
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    pathlength_map[i - top_left[0], j - top_left[1]] = -1
                    continue
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] == agent_id:
                    # agent's position
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.goals_map[i, j] == agent_id:
                    # agent's goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions
                    visible_agents.append(self.world.state[i, j])
                    poss_map[i - top_left[0], j - top_left[1]] = 1

                # we can keep this map even if on goal,
                # since observation is computed after the refresh of new distance map
                pathlength_map[i - top_left[0], j - top_left[1]] = self.world.agents[agent_id].distanceMap[i, j]

        distance = lambda x1, y1, x2, y2: ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5
        for agent in visible_agents:
            x, y = self.world.getGoal(agent)
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # out of observation
                min_node = (-1, -1)
                min_dist = 1000
                for i in range(top_left[0], top_left[0] + self.observation_size):
                    for j in range(top_left[1], top_left[1] + self.observation_size):
                        d = distance(i, j, x, y)
                        if d < min_dist:
                            min_node = (i, j)
                            min_dist = d
                goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1
            else:
                goals_map[x - top_left[0], y - top_left[1]] = 1
        dx = self.world.getGoal(agent_id)[0] - self.world.getPos(agent_id)[0]
        dy = self.world.getGoal(agent_id)[1] - self.world.getPos(agent_id)[1]
        mag = (dx ** 2 + dy ** 2) ** .5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag

        current_corridor_id = -1
        current_corridor = self.world.corridor_map[self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]][1]
        if current_corridor == 1:
            current_corridor_id = \
                self.world.corridor_map[self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]][0]

        positions = self.get_next_positions(agent_id)
        for position in positions:
            cell_info = self.world.corridor_map[position[0], position[1]]
            if cell_info[1] == 1:
                corridor_id = cell_info[0]
                if corridor_id != current_corridor_id:
                    if len(self.world.corridors[corridor_id]['EndPoints']) == 1:
                        if [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][0]:
                            blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                                corridor_id,
                                0, agent_id,
                                1)
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][0]:
                        end_point_pos = self.world.corridors[corridor_id]['EndPoints'][0]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            0, agent_id,
                            2)
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][1]:
                        end_point_pos = self.world.corridors[corridor_id]['EndPoints'][1]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            1, agent_id,
                            2)
                    else:
                        pass

        state = np.array([poss_map, goal_map, goals_map, obs_map])
        return state, [dx, dy, mag], [[[], [], [], [], [], blocking_map, deltax_map, deltay_map]]

    def get_many(self, handles=None):
        observations = {}
        all_astar_maps = self.get_astar_map()
        if handles is None:
            handles = list(range(1, self.world.num_agents + 1))
        for h in handles:
            observations[h] = self._get(h, all_astar_maps)
        return observations

    def get_astar_map(self):
        """

        :return: a dict of 3D np arrays. Each astar_maps[agentID] is a num_future_steps * obs_size * obs_size matrix.
        """

        def get_single_astar_path(distance_map, start_position, path_len):
            """
            :param distance_map:
            :param start_position:
            :param path_len:
            :return: [[(x,y), ...],..] a list of lists of positions from start_position, the length of the return can be
            smaller than num_future_steps. Index of the return: list[step][0-n] = tuple(x, y)
            """

            def get_astar_one_step(position):
                next_astar_cell = []
                h = self.world.state.shape[0]
                w = self.world.state.shape[1]
                for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    # print(position, direction)
                    new_pos = tuple_plus(position, direction)
                    if 0 < new_pos[0] <= h and 0 < new_pos[1] <= w:
                        if distance_map[new_pos] == distance_map[position] - 1 \
                                and distance_map[new_pos] >= 0:
                            next_astar_cell.append(new_pos)
                return next_astar_cell

            path_counter = 0
            astar_list = [[start_position]]
            while path_counter < path_len:
                last_step_cells = astar_list[-1]
                next_step_cells = []
                for cells_per_step in last_step_cells:
                    new_cell_list = get_astar_one_step(cells_per_step)
                    if not new_cell_list:  # no next step, should be standing on goal
                        astar_list.pop(0)
                        return astar_list
                    next_step_cells.extend(new_cell_list)
                next_step_cells = list(set(next_step_cells))  # remove repeated positions
                astar_list.append(next_step_cells)
                path_counter += 1
            astar_list.pop(0)
            return astar_list

        astar_maps = {}
        for agentID in range(1, self.world.num_agents + 1):
            astar_maps.update(
                {agentID: np.zeros([self.num_future_steps, self.world.state.shape[0], self.world.state.shape[1]])})

            distance_map0, start_pos0 = self.world.agents[agentID].distanceMap, self.world.agents[agentID].position
            astar_path = get_single_astar_path(distance_map0, start_pos0, self.num_future_steps)

            if not len(astar_path) == self.num_future_steps:  # this agent reaches its goal during future steps
                distance_map1, start_pos1 = self.world.agents[agentID].next_distanceMap, \
                                            self.world.agents[agentID].goal_pos
                astar_path.extend(
                    get_single_astar_path(distance_map1, start_pos1, self.num_future_steps - len(astar_path)))

            for i in range(self.num_future_steps - len(astar_path)):  # only happen when min_distance not sufficient
                astar_path.extend([[astar_path[-1][-1]]])  # stay at the last pos

            assert len(astar_path) == self.num_future_steps
            for step in range(self.num_future_steps):
                for cell in astar_path[step]:
                    astar_maps[agentID][step, cell[0], cell[1]] = agentID

        return astar_maps

    def get_blocking(self, corridor_id, reverse, agent_id, dead_end):
        def get_last_pos(agentID, position):
            history_list = copy.deepcopy(self.world.agents[agentID].position_history)
            history_list.reverse()
            assert (history_list[0] == self.world.getPos(agentID))
            history_list.pop(0)
            if history_list == []:
                return None
            for pos in history_list:
                if pos != position:
                    return pos
            return None

        positions_to_check = copy.deepcopy(self.world.corridors[corridor_id]['Positions'])
        if reverse:
            positions_to_check.reverse()
        idx = -1
        for position in positions_to_check:
            idx += 1
            state = self.world.state[position[0], position[1]]
            if state > 0 and state != agent_id:
                if dead_end == 1:
                    return 1
                if idx == 0:
                    return 1
                last_pos = get_last_pos(state, position)
                if last_pos == None:
                    return 1
                if idx == len(positions_to_check) - 1:
                    if last_pos != positions_to_check[idx - 1]:
                        return 1
                    break
                if last_pos == positions_to_check[idx + 1]:
                    return 1
        if dead_end == 2:
            if not reverse:
                other_endpoint = self.world.corridors[corridor_id]['EndPoints'][1]
            else:
                other_endpoint = self.world.corridors[corridor_id]['EndPoints'][0]
            state_endpoint = self.world.state[other_endpoint[0], other_endpoint[1]]
            if state_endpoint > 0 and state_endpoint != agent_id:
                return -1
        return 0


class PRIMAL1_env(PRIMAL2_Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, observer, map_generator, num_agents=None,
                 IsDiagonal=False, frozen_steps=0):
        super(PRIMAL1_env, self).__init__(observer=observer, map_generator=map_generator,
                                          num_agents=num_agents,
                                          IsDiagonal=IsDiagonal, frozen_steps=frozen_steps)

    def listValidActions(self, agent_ID, agent_obs):
        """
        :return: action:int, pos:(int,int)
        in non-corridor states:
            return all valid actions
        in corridor states:
            if standing on goal: Only going 'forward' allowed
            if not standing on goal: only going 'forward' allowed
        """

        available_actions = []
        pos = self.world.getPos(agent_ID)
        # if the agent is inside a corridor
        available_actions.append(0)  # standing still always allowed
        num_actions = 4 + 1 if not self.IsDiagonal else 8 + 1
        for action in range(1, num_actions):
            direction = action2dir(action)
            new_pos = tuple_plus(direction, pos)
            if self.world.state[new_pos[0], new_pos[1]] == 0:
                available_actions.append(action)

        return available_actions
