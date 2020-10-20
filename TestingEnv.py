import warnings
import json
import multiprocessing
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=Warning)
from PRIMAL2_Env import PRIMAL2_Env
from PRIMAL2_Observer import PRIMAL2_Observer
from OriginalPrimal_Observer import PRIMALObserver
from Map_Generator import *
from Observer_Builder import DummyObserver

import tensorflow as tf
from ACNet import ACNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Env_Builder import *


class RL_Planner(MAPFEnv):
    def __init__(self, observer, model_path, gpu_fraction=0.04):
        super().__init__(observer=observer, map_generator=DummyGenerator(), num_agents=1, isOneShot=True)

        self._set_testType()
        self._set_tensorflow(model_path, gpu_fraction)

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.method = '_oneshotPRIMAL2'

    def _set_tensorflow(self, model_path, gpu_fraction):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.sess = tf.Session(config=config)

        self.num_channels = 18  # HAS TO BE ENTERED MANUALLY TO MATCH THE MODEL, to be read from DRLMAPF...
        self.network = ACNet("global", a_size=5, trainer=None, TRAINING=False,
                             NUM_CHANNEL=self.num_channels,
                             OBS_SIZE=self.observer.observation_size,
                             GLOBAL_NET_SCOPE="global")

        # load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def set_world(self):
        return

    def give_moving_reward(self, agentID):
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
        return

    def _reset(self, map_generator=None, worldInfo=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal)
        else:
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
            raise UserWarning('you are using re-computing env mode')
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None
        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def step_greedily(self, o):
        def run_network(o):
            inputs, goal_pos, rnn_out = [], [], []

            for agentID in range(1, self.num_agents + 1):
                agent_obs = o[agentID]
                inputs.append(agent_obs[0])
                goal_pos.append(agent_obs[1])
            # compute up to LSTM in parallel
            h3_vec = self.sess.run([self.network.h3],
                                   feed_dict={self.network.inputs  : inputs,
                                              self.network.goal_pos: goal_pos})
            h3_vec = h3_vec[0]
            # now go all the way past the lstm sequentially feeding the rnn_state
            for a in range(0, self.num_agents):
                rnn_state = self.agent_states[a]
                lstm_output, state = self.sess.run([self.network.rnn_out, self.network.state_out],
                                                   feed_dict={self.network.inputs     : [inputs[a]],
                                                              self.network.h3         : [h3_vec[a]],
                                                              self.network.state_in[0]: rnn_state[0],
                                                              self.network.state_in[1]: rnn_state[1]})
                rnn_out.append(lstm_output[0])
                self.agent_states[a] = state
            # now finish in parallel
            policy_vec = self.sess.run([self.network.policy],
                                       feed_dict={self.network.rnn_out: rnn_out})
            policy_vec = policy_vec[0]
            action_dict = {agentID: np.argmax(policy_vec[agentID - 1]) for agentID in range(1, self.num_agents + 1)}
            return action_dict

        numCrashedAgents, computing_time = 0, 0

        start_time = time.time()
        action_dict = run_network(o)
        computing_time = time.time() - start_time

        next_o, reward = self.step_all(action_dict)

        for agentID in reward.keys():
            if reward[agentID] // 1 != 0:
                numCrashedAgents += 1
        assert numCrashedAgents <= self.num_agents

        return numCrashedAgents, computing_time, next_o

    def find_path(self, max_length, saveImage=True, time_limit=np.Inf):
        assert max_length > 0
        step_count, num_crash, computing_time_list, frames = 0, 0, [], []
        episode_status = 'no early stop'

        obs = self._observe()
        for step in range(1, max_length + 1):
            if saveImage:
                frames.append(self._render(mode='rgb_array'))
            numCrash_AStep, computing_time, obs = self.step_greedily(obs)

            computing_time_list.append(computing_time)
            num_crash += numCrash_AStep
            step_count = step

            end_episode = self.done
            if end_episode:  # done early
                episode_status = 'succeed'
                break

            if time_limit < computing_time:
                episode_status = "timeout"
                break

        if saveImage:
            frames.append(self._render(mode='rgb_array'))

        return step_count, num_crash, \
               True if episode_status == 'succeed' else False, \
               self.num_agents if episode_status == 'succeed' \
                   else sum([1 if self.world.agents[agentID].status > 0
                             else 0 for agentID in range(1, self.num_agents + 1)]), \
               frames


class MstarOneshotPlanner(MAPFEnv):
    def __init__(self, ):
        super(MstarOneshotPlanner, self).__init__(observer=DummyObserver(), map_generator=DummyGenerator(),
                                                  num_agents=1, IsDiagonal=False, isOneShot=True)
        self._set_testType()

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.method = '_oneshotmstar'

    def set_world(self):
        return

    def give_moving_reward(self, agentID):
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
        return

    def find_path(self, max_length, saveImage=False, time_limit=300):
        world = self.getObstacleMap()
        start_positions = []
        goals = []
        start_positions_dir = self.getPositions()
        goals_dir = self.getGoals()
        for i in range(1, self.world.num_agents + 1):
            start_positions.append(start_positions_dir[i])
            goals.append(goals_dir[i])
        try:
            mstar_path = od_mstar.find_path(world, start_positions, goals,
                                            inflation=3.0, time_limit=5 * time_limit)
        except OutOfTimeError:
            return 0, 0, False, 0, []
        except NoSolutionError:
            return -1, 0, False, -1, []
        if saveImage:
            Warning("oneshot mstar doesn't support GIF generation for remaining a high testing speed")
        #      step_count, num_crash, succeed, num_succeeded, frames
        return len(mstar_path), 0, True, self.world.num_agents, []

    def _reset(self, map_generator=None, worldInfo=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal)
        else:
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None


class OneShotTestsRunner:
    def __init__(self, env_path, result_path, Planner, resume_testing=False, GIF_prob=0.):
        self.env_path = env_path
        self.result_path = result_path
        self.resume_testing = resume_testing
        self.GIF_prob = float(GIF_prob)

        self.worker = Planner

        self.test_type = self.worker.method

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

    def read_single_env(self, name):
        root = self.env_path
        assert name.split('.')[-1] == 'npy'

        if self.resume_testing:
            env_name = name[:name.rfind('.')]
            if os.path.exists(self.result_path + env_name + "_continuousPRIMAL2.txt"):
                return None
        maps = np.load(root + name, allow_pickle=True)
        return maps

    def run_1_test(self, name, maps):
        def get_maxLength(env_size):
            if env_size <= 40:
                return 320  # 256
            elif env_size <= 80:
                return 480  # 384
            return 640  # 512

        self.worker._reset(map_generator=manual_generator(maps[0][0], maps[0][1]),
                           worldInfo=maps)
        env_name = name[:name.rfind('.')]
        results = dict()
        start_time = time.time()
        print("working on " + env_name)

        env_size = int(env_name[env_name.find("_") + 1:env_name.find("size")])
        max_length = get_maxLength(env_size)
        result = self.worker.find_path(max_length=int(max_length), saveImage=np.random.rand() < self.GIF_prob)

        step_count, num_crash, succeed, num_succeeded, frames = result
        results['time'] = time.time() - start_time
        results['steps'] = str(step_count) + '/' + str(max_length)
        results['crashed'] = num_crash
        results['succeed'] = succeed
        results['num_succeeded'] = num_succeeded

        self.make_gif(frames, env_name, self.worker.method)
        self.write_files(results, env_name, self.worker.method)
        return

    def make_gif(self, image, env_name, ext):
        if image:
            gif_name = self.result_path + env_name + ext + ".gif"
            images = np.array(image)
            make_gif(images, gif_name)

    def write_files(self, results, env_name, ext):
        txt_filename = self.result_path + env_name + ext + ".txt"
        f = open(txt_filename, 'w')
        f.write(json.dumps(results))
        f.close()


if __name__ == "__main__":
    import time

    model_path = './model_astar10_continuous/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./testing_result/")
    parser.add_argument("--env_path", default='./primal2_testing_envs50/')
    parser.add_argument("-r", "--resume_testing", default=True, help="resume testing")
    parser.add_argument("-g", "--GIF_prob", default=0., help="write GIF")
    parser.add_argument("-p", "--planner", default='mstar', help="choose between mstar and RL")
    parser.add_argument("-n", "--mapName", default=None, help="single map name for multiprocessing")
    args = parser.parse_args()

    # set a tester--------------------------------------------
    if args.planner == 'mstar':
        tester = OneShotTestsRunner(args.env_path,
                                    args.result_path,
                                    Planner=MstarOneshotPlanner(),
                                    resume_testing=args.resume_testing,
                                    GIF_prob=args.GIF_prob
                                    )

    elif args.planner == 'RL':
        tester = OneShotTestsRunner(args.env_path,
                                    args.result_path,
                                    Planner=RL_Planner(
                                        observer=PRIMAL2_Observer(observation_size=11, num_future_steps=10),
                                        model_path=model_path),
                                    resume_testing=args.resume_testing,
                                    GIF_prob=args.GIF_prob
                                    )
    else:
        raise NameError('invalid planner type')
    # run the tests---------------------------------------------------------

    maps = tester.read_single_env(args.mapName)
    if maps is None:
        print(args.mapName, " already completed")
    else:
        tester.run_1_test(args.mapName, maps)
