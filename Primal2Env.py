from Env_Builder import *
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from od_mstar3 import od_mstar
from GroupLock import Lock
import random
from gym import spaces

'''
    Observation: 
    Action space: (Tuple)
        agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
                 5:NE, 6:SE, 7:SW, 8:NW, 5,6,7,8 not used in non-diagonal world}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''

## New Action Space: {0,1,2,3} -> {static, forward, CW, CCW}


class Primal2Env(MAPFEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, observer, map_generator, num_agents=None,
                 IsDiagonal=False, frozen_steps=0, isOneShot=False):
        super(Primal2Env, self).__init__(observer=observer, map_generator=map_generator,
                                          num_agents=num_agents,
                                          IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=isOneShot)

    def _reset(self, new_generator=None):
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

    # DONE change how to find and check for valid actions from given position
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


        # corridor_map[x,y][0] = corridor ID
        # corridor_map[x,y][1] = is agent inside corridor

        available_actions = []
        pos = self.world.getPos(agent_ID)
        # if the agent is inside a corridor
        if self.world.corridor_map[pos[0], pos[1]][1] == 1:
            corridor_id = self.world.corridor_map[pos[0], pos[1]][0]
            if [pos[0], pos[1]] not in self.world.corridors[corridor_id]['StoppingPoints']:
                possible_moves = self.world.valid_neighbors_oriented(pos) # DONE edit for orientation in Env_Builder
                last_position = get_last_pos(agent_ID, pos)
                for possible_position in possible_moves:
                    # Here: In corridor, not on a stopping point
                    if possible_position is not None and possible_position != last_position \
                            and self.world.state[possible_position[0], possible_position[1]] == 0:
                        # Here not last position and valid state
                        # DONE create 2 tuple action from 3 tuple position (repeated below)
                        temp_action = (tuple_minus(possible_position, pos))
                        available_actions.append(dir2action(temp_action[0], temp_action[1]))

                    # TODO What does corridors[ID][Endpoints] ==1 mean... end of a corridor? 
                    elif len(self.world.corridors[corridor_id]['EndPoints']) == 1 and possible_position is not None \
                            and possible_moves.count(None) == 3: # where there is only 1 possible move and 3 "None" returned 
                        temp_action = (tuple_minus(possible_position, pos))
                        available_actions.append(dir2action(temp_action[0], temp_action[1]))

                if not available_actions:
                    available_actions.append(0)
            else: # Here: In corridor, on a stopping point
                possible_moves = self.world.valid_neighbors_oriented(pos)
                last_position = get_last_pos(agent_ID, pos)
                if last_position in self.world.corridors[corridor_id]['Positions']:
                    available_actions.append(0)
                    for possible_position in possible_moves:
                        if possible_position is not None and possible_position != last_position \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            temp_action = (tuple_minus(possible_position, pos))
                            available_actions.append(dir2action(temp_action[0], temp_action[1]))
                else:
                    for possible_position in possible_moves:
                        if possible_position is not None \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            temp_action = (tuple_minus(possible_position, pos))
                            available_actions.append(dir2action(temp_action[0], temp_action[1]))
                    if not available_actions:
                        available_actions.append(0)
        # agent not in corridor
        else:
            available_actions.append(0)  # standing still always allowed when not in corridor
            # DONE change logic for available_actions for orientaion
            num_actions = 4  # now only 0-3
            for action in range(0, num_actions): 
                # use new action2position(action, current_position) to get each of the potential new_positions
                new_position = action2position(action, pos)

                lastpos = None
                blocking_valid = self.get_blocking_validity(agent_obs, agent_ID, new_position)
                if not blocking_valid:
                    continue
                try:
                    lastpos = self.world.agents[agent_ID].position_history[-2]
                except:
                    pass
                if new_position == lastpos:
                    continue
                if self.world.corridor_map[new_position[0], new_position[1]][1] == 1:
                    valid = self.get_convention_validity(agent_obs, agent_ID, new_position)
                    if not valid:
                        continue
                if self.world.state[new_position[0], new_position[1]] == 0:
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


class DummyEnv(Primal2Env):
    def __init__(self, observer, map_generator, num_agents=None, IsDiagonal=False):
        super(DummyEnv, self).__init__(observer=observer, map_generator=map_generator,
                                       num_agents=num_agents,
                                       IsDiagonal=IsDiagonal)

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800):
        pass


if __name__ == '__main__':
    from matplotlib import pyplot
    from Primal2Observer import Primal2Observer
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
    env = Primal2Env(num_agents=n_agents,
                      observer=Primal2Observer(observation_size=5),
                      map_generator=maze_generator(env_size=(8, 10),
                                                   wall_components=(3, 8), obstacle_density=(0.3, 0.7)),
                      IsDiagonal=False)
    print(env.world.state)
    print(env.world.goals_map)
    c = 0
    a = c
    b = c
    for j in range(0, 50):
          movement = {1: a, 2: b, 3: c, 4: c, 5: c, 6: c, 7: c, 8: c} 
          env.step_all(movement)
          obs = env._observe()

          print(env.world.state)
          a = int(input())
          b = int(input())
