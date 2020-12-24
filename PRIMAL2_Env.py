from Env_Builder import *
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError

try:
    # from od_mstar3 import od_mstar
    from od_mstar3 import cpp_mstar as od_mstar
except ImportError:
    raise ImportError('cpp_mstar not compiled. Please refer to README')
from GroupLock import Lock
import random
from gym import spaces


class PRIMAL2_Env(MAPFEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, observer, map_generator, num_agents=None,
                 IsDiagonal=False, frozen_steps=0, isOneShot=False):
        super(PRIMAL2_Env, self).__init__(observer=observer, map_generator=map_generator,
                                          num_agents=num_agents,
                                          IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=isOneShot)

    def _reset(self, new_generator=None, *args):
        if new_generator is None:
            self.set_world()
        else:
            self.map_generator = new_generator
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
            self.num_agents = self.world.num_agents
            self.observer.set_env(self.world)

        self.fresh = True
        if self.viewer is not None:
            self.viewer = None

    def give_moving_reward(self, agentID):
        """
        WARNING: ONLY CALL THIS AFTER MOVING AGENTS!
        Only the moving agent that encounters the collision is penalized! Standing still agents
        never get punishment.
        """
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        """
        :return: action:int, pos:(int,int)
        in non-corridor states:
            return all valid actions
        in corridor states:
            if standing on goal: Only going 'forward' allowed
            if not standing on goal: only going 'forward' allowed
        """

        def get_last_pos(agentID, position):
            """
            get the last different position of an agent
            """
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

        VANILLA_VALID_ACTIONS = True

        if VANILLA_VALID_ACTIONS == True:
            available_actions = []
            pos = self.world.getPos(agent_ID)
            available_actions.append(0)  # standing still always allowed 
            num_actions = 4 + 1 if not self.IsDiagonal else 8 + 1
            for action in range(1, num_actions):
                direction = action2dir(action)
                new_pos = tuple_plus(direction, pos)
                lastpos = None
                try:
                    lastpos = self.world.agents[agent_ID].position_history[-2]
                except:
                    pass
                if new_pos == lastpos:
                    continue
                if self.world.state[new_pos[0], new_pos[1]] == 0:
                    available_actions.append(action)

            return available_actions

        available_actions = []
        pos = self.world.getPos(agent_ID)
        # if the agent is inside a corridor
        if self.world.corridor_map[pos[0], pos[1]][1] == 1:
            corridor_id = self.world.corridor_map[pos[0], pos[1]][0]
            if [pos[0], pos[1]] not in self.world.corridors[corridor_id]['StoppingPoints']:
                possible_moves = self.world.blank_env_valid_neighbor(*pos)
                last_position = get_last_pos(agent_ID, pos)
                for possible_position in possible_moves:
                    if possible_position is not None and possible_position != last_position \
                            and self.world.state[possible_position[0], possible_position[1]] == 0:
                        available_actions.append(dir2action(tuple_minus(possible_position, pos)))

                    elif len(self.world.corridors[corridor_id]['EndPoints']) == 1 and possible_position is not None \
                            and possible_moves.count(None) == 3:
                        available_actions.append(dir2action(tuple_minus(possible_position, pos)))

                if not available_actions:
                    available_actions.append(0)
            else:
                possible_moves = self.world.blank_env_valid_neighbor(*pos)
                last_position = get_last_pos(agent_ID, pos)
                if last_position in self.world.corridors[corridor_id]['Positions']:
                    available_actions.append(0)
                    for possible_position in possible_moves:
                        if possible_position is not None and possible_position != last_position \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            available_actions.append(dir2action(tuple_minus(possible_position, pos)))
                else:
                    for possible_position in possible_moves:
                        if possible_position is not None \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            available_actions.append(dir2action(tuple_minus(possible_position, pos)))
                    if not available_actions:
                        available_actions.append(0)
        else:
            available_actions.append(0)  # standing still always allowed 
            num_actions = 4 + 1 if not self.IsDiagonal else 8 + 1
            for action in range(1, num_actions):
                direction = action2dir(action)
                new_pos = tuple_plus(direction, pos)
                lastpos = None
                blocking_valid = self.get_blocking_validity(agent_obs, agent_ID, new_pos)
                if not blocking_valid:
                    continue
                try:
                    lastpos = self.world.agents[agent_ID].position_history[-2]
                except:
                    pass
                if new_pos == lastpos:
                    continue
                if self.world.corridor_map[new_pos[0], new_pos[1]][1] == 1:
                    valid = self.get_convention_validity(agent_obs, agent_ID, new_pos)
                    if not valid:
                        continue
                if self.world.state[new_pos[0], new_pos[1]] == 0:
                    available_actions.append(action)

        return available_actions

    def get_blocking_validity(self, observation, agent_ID, pos):
        top_left = (self.world.getPos(agent_ID)[0] - self.obs_size // 2,
                    self.world.getPos(agent_ID)[1] - self.obs_size // 2)
        blocking_map = observation[0][5]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 1:
            return 0
        return 1

    def get_convention_validity(self, observation, agent_ID, pos):
        top_left = (self.world.getPos(agent_ID)[0] - self.obs_size // 2,
                    self.world.getPos(agent_ID)[1] - self.obs_size // 2)
        blocking_map = observation[0][5]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == -1:
            deltay_map = observation[0][7]
            if deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] > 0:
                return 1
            elif deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 0:
                deltax_map = observation[0][6]
                if deltax_map[pos[0] - top_left[0], pos[1] - top_left[1]] > 0:
                    return 1
                else:
                    return 0
            elif deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] < 0:
                return 0
            else:
                print('Weird')
        else:
            return 1


class DummyEnv(PRIMAL2_Env):
    def __init__(self, observer, map_generator, num_agents=None, IsDiagonal=False):
        super(DummyEnv, self).__init__(observer=observer, map_generator=map_generator,
                                       num_agents=num_agents,
                                       IsDiagonal=IsDiagonal)

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800):
        pass


if __name__ == '__main__':
    from matplotlib import pyplot
    from PRIMAL2_Observer import PRIMAL2_Observer
    from Map_Generator import maze_generator
    from Map_Generator import manual_generator

    state0 = [[-1, -1, -1, -1, -1, -1, -1],
              [-1, 1, -1, 0, 0, 0, -1],
              [-1, 0, -1, -1, -1, 0, -1],
              [-1, 0, 0, 0, -1, 0, -1],
              [-1, 0, -1, 0, 0, 0, -1],
              [-1, 2, -1, 0, 0, 0, -1],
              [-1, -1, -1, -1, -1, -1, -1]]
    n_agents = 3
    env = PRIMAL2_Env(num_agents=n_agents,
                      observer=PRIMAL2_Observer(observation_size=5),
                      map_generator=maze_generator(env_size=(8, 10),
                                                   wall_components=(3, 8), obstacle_density=(0.3, 0.7)),
                      # map_generator=manual_generator(state_map=state0, goals_map=None),
                      IsDiagonal=False)
    # for i in range(4):
    #    env.step_all({1: 1, 2: 1, 3: 1})
    #    print(env._render())
    print(env.world.state)
    print(env.world.goals_map)
    c = 0
    a = c
    b = c
    for j in range(0, 50):
        movement = {1: a, 2: b, 3: c, 4: c, 5: c, 6: c, 7: c, 8: c}
        env.step_all(movement)
        print(env.world.state)
