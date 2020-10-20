import warnings
import multiprocessing
import os
import time
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=Warning)
from pathos.multiprocessing import ProcessPool as Pool
import argparse
import subprocess


# - I don't think these are necessary - Emerson
# from Map_Generator import *
# from Observer_Builder import DummyObserver
# from Env_Builder import *

def get_map_names(env_path, result_path, resume_testing):
    # assert self.worker.test_type == self.test_type

    print('loading testing env...')

    valid_map_names = []
    for root, dirs, files in os.walk(env_path, topdown=False):
        for name in files:
            if name.split('.')[-1] != 'npy':
                continue
            if resume_testing:
                env_name = name[:name.rfind('.')]
                if os.path.exists("{}_continuous{}.txt".format(result_path + env_name, args.planner)):
                    continue

            valid_map_names.append(name)

    print('There are ' + str(len(valid_map_names)) + ' remaining tests detected')
    return valid_map_names


def run_tests(args, env_path, result_path):
    num_core = args.num_worker
    parallel_pool = Pool(num_core)

    valid_map_names = get_map_names(env_path, result_path, args.resume_testing)

    print('start testing with ' + str(num_core) + ' processes...')
    allResults = []
    for name in valid_map_names:
        result = parallel_pool.apipe(run_1_test_wrapper, args, name)
        allResults.append(result)

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

    parallel_pool.close()


def run_1_test_wrapper(args, name):
    """
    Calls TestingEnv.py in a subprocess.
    This approach avoids any multiprocessing issues with tensorflow
    """

    s = "python3.6 TestingEnv.py -r {resume_testing} -g {GIF_prob} " \
        + "-p {planner} -n {mapName}"

    s = s.format(resume_testing=args.resume_testing, GIF_prob=args.GIF_prob,
                 planner=args.planner, mapName=name)
    if args.printInfo:
        subprocess.run(s, stderr=subprocess.STDOUT, shell=True)
    else:
        try:
            devNull = open('/dev/null', 'w')
            subprocess.run(s, stderr=devNull, shell=True)
        except Exception:
            Warning('cannot mute info, reset printInfo to True')
            subprocess.run(s, stderr=subprocess.STDOUT, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./continuous_testing_result/")
    parser.add_argument("--env_path", default='./primal2_testing_envs50/')
    parser.add_argument("--num_worker", default=10, type=int)
    parser.add_argument("--printInfo", default=True, type=bool)
    parser.add_argument("-r", "--resume_testing", default=True, help="resume testing")
    parser.add_argument("-g", "--GIF_prob", default=0., help="prob to write GIF")
    parser.add_argument("-p", "--planner", default='RL', help="choose between mstar and RL")
    args = parser.parse_args()

    run_tests(args, args.env_path, args.result_path)
