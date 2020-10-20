from Observer_Builder import ObservationBuilder
import numpy as np
import copy
from Env_Builder import *


class PRIMALObserver(ObservationBuilder):
    """
    obs shape: (8 + num_future_steps * obs_size * obs_size )
    map order: poss_map, goal_map, goals_map, obs_map, pathlength_map, blocking_map, deltax_map, deltay_map, astar maps
    """

    def __init__(self, observation_size=11, num_future_steps=3):
        super(PRIMALObserver, self).__init__()
        self.observation_size = observation_size
        self.num_future_steps = num_future_steps
        self.NUM_CHANNELS =4 

    def set_world(self, world):
        super().set_env(world)

    def _get(self, agent_id):

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

        # original layers from PRIMAL1
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    obs_map[i - top_left[0], j - top_left[1]] = 1
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


        for agent in visible_agents:
            x, y = self.world.getGoal(agent)
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1
                 
        dx = self.world.getGoal(agent_id)[0] - self.world.getPos(agent_id)[0]
        dy = self.world.getGoal(agent_id)[1] - self.world.getPos(agent_id)[1]
        mag = (dx ** 2 + dy ** 2) ** .5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        if mag >60 :
            mag=60    

        state = np.array([poss_map, goal_map, goals_map, obs_map])
        return state, [dx, dy, mag]

    def get_many(self, handles=None):
        observations = {}
        if handles is None:
            handles = list(range(1, self.world.num_agents + 1))
        for h in handles:
            observations[h] = self._get(h)
        return observations

if __name__ == "__main__":
    pass
