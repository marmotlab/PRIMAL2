import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import multiprocessing as mp
from pathos.multiprocessing import ProcessPool as Pool
# from pathos import multiprocessing as mp
import os
import argparse
from PRIMAL2_Env import *
from Observer_Builder import DummyObserver
from PRIMAL2_Observer import PRIMAL2_Observer
from Map_Generator import *
from tqdm import tqdm
from Env_Builder import *


class MazeTestGenerator:
    def __init__(self, env_dir, printInfo=False, pureMaps=False):
        self.env_dir = env_dir + '/' if env_dir[-1] != '/' else env_dir
        self.num_core = mp.cpu_count()
        self.parallel_pool = Pool()
        self.printInfo = printInfo
        self.pureMaps = pureMaps  # if true, only save state_map and goals_map, else save the whole world

    def make_name(self, n, s, d, w, id, extension, dirname, extra=""):
        if dirname[-1] == '/':
            dirname = dirname[:-1]
        if extra == "":
            return dirname + '/' + "{}agents_{}size_{}density_{}wall_id{}{}".format(n, s, d, w, id, extension)
        else:
            return dirname + '/' + "{}agents_{}size_{}density_{}wall_id{}_{}{}".format(n, s, d, w, id, extra, extension)

    def create_map(self, num_agents, env_size, obs_dense, wall_component, id):
        num_agents, env_size, obs_dense, \
        wall_component, id = int(num_agents), int(env_size), round(obs_dense, 2), round(wall_component, 2), int(id)
        file_name = self.make_name(num_agents, env_size, obs_dense, wall_component, id,
                                   dirname=self.env_dir,
                                   extension='.npy')
        if os.path.exists(file_name):
            if self.printInfo:
                print('skip env:' + file_name)
            return

        gameEnv = DummyEnv(num_agents=num_agents,
                           observer=DummyObserver(),
                           map_generator=maze_generator(
                               env_size=env_size,
                               wall_components=wall_component,
                               obstacle_density=obs_dense),
                           IsDiagonal=False)
        state = np.array(gameEnv.world.state)
        goals = np.array(gameEnv.world.goals_map)
        if self.pureMaps:
            info = np.array([state, goals])
        else:
            agents_init_pos = gameEnv.world.agents_init_pos
            # goals_init_pos = gameEnv.world.goals_init_pos
            corridor_map = np.array(gameEnv.world.corridor_map)
            corridors = np.array(gameEnv.world.corridors)
            agents_object = gameEnv.world.agents
            info = np.array([[state, goals],
                             agents_init_pos, corridor_map, corridors, agents_object])

        np.save(file_name, info)
        # self.parallel_pool.apipe(self.write_map, (file_name, maps))
        if self.printInfo:
            print('finish env:' + file_name)
        return None

    def run_mazeMap_creator(self, num_agents_list, env_size_list, obs_dense_list,
                            wall_component_list, num_tests, multiProcessing=True):

        if not os.path.exists(self.env_dir):
            os.makedirs(self.env_dir)
        if multiProcessing:
            print("Multi-processing activated, you are using {:d} processes".format(self.num_core))
        else:
            print("Single-processing activated, you are using 1 processes")

        print("There are " + format(len(num_agents_list) * len(env_size_list) * len(obs_dense_list) *
                                    len(wall_component_list) * num_tests, ',') + " tests in total. Start Working!")

        allResults = []
        for num_agents in num_agents_list:
            for env_size in env_size_list:
                for obs_dense in obs_dense_list:
                    for wall_component in wall_component_list:
                        for i in range(num_tests):
                            if env_size <= 20 and num_agents >= 128:
                                continue
                            if env_size <= 40 and num_agents >= 256:
                                continue

                            if multiProcessing:
                                result = self.parallel_pool.apipe(self.create_map, num_agents, env_size, obs_dense,
                                                                  wall_component, i)
                                allResults.append(result)
                            else:
                                self.create_map(num_agents, env_size, obs_dense, wall_component, i)

        totalJobs = len(allResults)
        jobsCompleted = 0
        while len(allResults) > 0:
            for i in range(len(allResults)):
                if allResults[i].ready():
                    jobsCompleted += 1
                    print("{} / {}".format(jobsCompleted, totalJobs))
                    allResults[i].get()
                    allResults.pop(i)
                    break
        self.parallel_pool.close()
        print('finish all envs!')


class MazeTestInfoAdder(MazeTestGenerator):
    """
    add info to previous testing env.
    Info in the npy file  in FIXED ORDER!!!:
    [
    [state_map, goal_map], <----- previous info
    agents_init_pos
    goals_init_pos
    corridor_map
    corridors
    world.agents
    ]
    """

    def __init__(self, env_dir, printInfo=False):
        super(MazeTestInfoAdder, self).__init__(env_dir=env_dir, printInfo=printInfo)

    def read_envs(self):
        # assert self.worker.test_type == self.test_type
        print('loading testing env...')
        maps_dict = {}

        for root, dirs, files in os.walk(self.env_dir, topdown=False):
            for name in files:
                if name.split('.')[-1] != 'npy':
                    continue
                try:
                    maps = np.load(root + name, allow_pickle=True)
                    if len(maps) > 2:
                        continue
                except ValueError:
                    print(root + name, 'is a broken file that numpy cannot read, possibly due to the forced '
                                       'suspension of generation code. Automatically skip this env...')
                    continue
                if len(maps) != 2:  # notice that only pure maps will be processed
                    continue
                maps_dict.update({name: maps})
        print('There are ' + str(len(maps_dict.keys())) + ' tests detected')
        return maps_dict

    def add_info(self, state_map, goals_map, file_name):

        gameEnv = DummyEnv(num_agents=1,
                           observer=DummyObserver(),
                           map_generator=manual_generator(state_map=state_map, goals_map=goals_map),
                           IsDiagonal=False)
        state = np.array(gameEnv.world.state)
        goals = np.array(gameEnv.world.goals_map)
        agents_init_pos = gameEnv.world.agents_init_pos
        corridor_map = np.array(gameEnv.world.corridor_map)
        corridors = np.array(gameEnv.world.corridors)
        agents_object = gameEnv.world.agents

        info = np.array([[state, goals],
                         agents_init_pos, corridor_map, corridors, agents_object])

        np.save(self.env_dir + file_name, info)
        # self.parallel_pool.apipe(self.write_map, (file_name, maps))
        if self.printInfo:
            print('finish env:' + file_name)
        return None

    def run_mazeMap_infoAdder(self, multiProcessing=True):

        if not os.path.exists(self.env_dir):
            os.makedirs(self.env_dir)
        if multiProcessing:
            print("Multi-processing activated, you are using {:d} processes".format(self.num_core))
        else:
            print("Single-processing activated, you are using 1 processes")

        map_dict = self.read_envs()
        print("There are " + str(len(map_dict.keys())) + " tests in total. Start Working!")

        allResults = []
        for file_name, maps in map_dict.items():
            if multiProcessing:
                result = self.parallel_pool.apipe(self.add_info, maps[0], maps[1], file_name)
                allResults.append(result)
                # result.get()
            else:
                self.add_info(maps[0], maps[1], file_name)

        totalJobs = len(allResults)
        jobsCompleted = 0
        while len(allResults) > 0:
            for i in range(len(allResults)):
                if allResults[i].ready():
                    jobsCompleted += 1
                    print("{} / {}".format(jobsCompleted, totalJobs))
                    allResults[i].get()
                    allResults.pop(i)
                    break

        self.parallel_pool.close()
        print('finish all envs!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_agents", default=[4, 8, 16, 32, 64, 128, 256], help="number of agents in the env")
    parser.add_argument("-s", "--env_size", default=[20, 40, 80, 160], help="env size")
    parser.add_argument("-d", "--obs_dense", default=[0.3 + 0.15 * i for i in range(4)],
                        help="obstacle density of the env")
    parser.add_argument("-w", "--wall_component", default=[1, 6, 12, 20], help="average length of each wall")
    parser.add_argument("-t", "--num_tests", default=20, help="number of tests per env setting")
    parser.add_argument("-e", "--env_dir", default='./new_testing_envs', help="dir where you want to save the envs")
    parser.add_argument("-p", "--printInfo", default=True, help="if you want to print generated env info")
    parser.add_argument("-a", "--addInfo", default=False, help="switch between addInfo mode and map generation mode")
    parser.add_argument("--pureMaps", default=False, help="only generate state map and goals map in map generation")
    args = parser.parse_args()

    if args.addInfo:
        adder = MazeTestInfoAdder(args.env_dir, printInfo=args.printInfo)
        adder.run_mazeMap_infoAdder()
    else:
        generator = MazeTestGenerator(args.env_dir, printInfo=args.printInfo, pureMaps=args.pureMaps)
        # generator.create_map(1, 10, 1, 1, 1, 1)
        generator.run_mazeMap_creator(args.num_agents, args.env_size, args.obs_dense,
                                      args.wall_component, args.num_tests)

        # generator.run_mazeMap_creator([4, 5, 6, 7], [10, 20], [0.1, 0.2],
        #                               [2, 3, 4], 1)
