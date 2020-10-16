import numpy as np


def _get_one_hot_for_agent_direction(agent):
    """Retuns the agent's direction to one-hot encoding."""
    direction = np.zeros(4)
    direction[agent.direction] = 1
    return direction


class ObservationBuilder:
    """
    ObservationBuilder base class.
    """

    def __init__(self):
        self.world = None
        self.NUM_CHANNELS = None

    def set_env(self, env):
        self.world = env

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

    def get_many(self, handles):
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.

        Parameters
        ----------
        handles : list of handles, optional
            List with the handles of the agents for which to compute the observation vector.

        Returns
        -------
        function
            A dictionary of observation structures, specific to the corresponding environment, with handles from
            `handles` as keys.
        """
        raise NotImplementedError


class DummyObserver(ObservationBuilder):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def __init__(self):
        super().__init__()
        self.observation_size = 1

    def reset(self):
        pass

    def get_many(self, handles) -> bool:
        return True

    def get(self, handle: int = 0) -> bool:
        return True
