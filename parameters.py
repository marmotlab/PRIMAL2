import numpy as np
# Learning parameters

gamma                   = .95  # discount rate for advantage estimation and reward discounting
LR_Q                    = 2.e-5  # 8.e-5 / NUM_THREADS # default: 1e-5
ADAPT_LR                = True
ADAPT_COEFF             = 5.e-5  # the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
EXPERIENCE_BUFFER_SIZE  = 256
max_episode_length      = 256
IL_MAX_EP_LENGTH        = 64
episode_count           = 0

# observer parameters
OBS_SIZE                = 11   # the size of the FOV grid to apply to each agent
NUM_FUTURE_STEPS        = 0

# environment parameters
ENVIRONMENT_SIZE        = (10, 70)  # the total size of the environment (length of one side) , Starting Point of Curriculum Only
WALL_COMPONENTS         = (3, 21)    # Starting Params of Curriculum = TRUE
OBSTACLE_DENSITY        = (0.2, 0.7)  # range of densities   Starting Params of Curriculum = TRUE

DIAG_MVMT               = False  # Diagonal movements allowed?
a_size                  = 5 + int(DIAG_MVMT) * 4
NUM_META_AGENTS         = 9
NUM_IL_META_AGENTS      = 4

NUM_THREADS             = 8 # int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS             = 1  # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)

# training parameters
SUMMARY_WINDOW          = 10
load_model              = False
RESET_TRAINER           = False
training_version        = 'astar3_continuous_0.5IL_ray2'
model_path              = 'model_' + training_version
gifs_path               = 'gifs_' + training_version
train_path              = 'train_' + training_version
OUTPUT_GIFS             = False  # Only for RL gifs
GIFS_FREQUENCY_RL       = 512
OUTPUT_IL_GIFS          = False
IL_GIF_PROB             = 0.


# Imitation options
PRIMING_LENGTH          = 0   # number of episodes at the beginning to train only on demonstrations
MSTAR_CALL_FREQUENCY    = 1

# observation variables
NUM_CHANNEL             = 4 + NUM_FUTURE_STEPS

# others
EPISODE_START           = episode_count
TRAINING                = True
EPISODE_SAMPLES         = EXPERIENCE_BUFFER_SIZE  # 64
GLOBAL_NET_SCOPE        = 'global'
swarm_reward            = [0] * NUM_META_AGENTS
swarm_targets           = [0] * NUM_META_AGENTS

# Shared arrays for tensorboard
episode_rewards         = [[] for _ in range(NUM_META_AGENTS)] 
episode_finishes        = [[] for _ in range(NUM_META_AGENTS)]
episode_lengths         = [[] for _ in range(NUM_META_AGENTS)]
episode_mean_values     = [[] for _ in range(NUM_META_AGENTS)]
episode_invalid_ops     = [[] for _ in range(NUM_META_AGENTS)]
episode_stop_ops        = [[] for _ in range(NUM_META_AGENTS)]
episode_wrong_blocking  = [[] for _ in range(NUM_META_AGENTS)]
rollouts                = [None for _ in range(NUM_META_AGENTS)]
GIF_frames              = []

# Joint variables 
joint_actions           = [{} for _ in range(NUM_META_AGENTS)]
joint_env               = [None for _ in range(NUM_META_AGENTS)]
joint_observations      =[{} for _ in range(NUM_META_AGENTS)]
joint_rewards           = [{} for _ in range(NUM_META_AGENTS)]
joint_done              = [{} for _ in range(NUM_META_AGENTS)]


env_params              = [[ [WALL_COMPONENTS[0], WALL_COMPONENTS[1]] , [OBSTACLE_DENSITY[0],OBSTACLE_DENSITY[1]]]  for _ in range(NUM_META_AGENTS)]




class JOB_OPTIONS:
    getExperience = 1
    getGradient = 2


class COMPUTE_OPTIONS:
    multiThreaded = 1
    synchronous = 2
    

JOB_TYPE = JOB_OPTIONS.getGradient
COMPUTE_TYPE = COMPUTE_OPTIONS.multiThreaded
