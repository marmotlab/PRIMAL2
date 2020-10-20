from __future__ import division

import os
import threading
import warnings
import scipy.signal as signal
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow.python.client import device_lib
from Map_Generator import maze_generator
import GroupLock
from ACNet import ACNet
from Env_Builder import *
from PRIMAL2Env import PRIMAL2Env
from PRIMAL2Observer import PRIMAL2Observer
from PRIMALObserver import PRIMALObserver
dev_list = device_lib.list_local_devices()


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# ## Worker Agent

# In[ ]:
class Worker:
    def __init__(self, game, metaAgentID, workerID, a_size, groupLock):
        self.workerID = workerID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_" + str(workerID)
        self.agentID = ((workerID - 1) % num_workers) + 1
        self.groupLock = groupLock

        self.nextGIF = episode_count  # For GIFs output
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = ACNet(self.name, a_size, trainer, True, NUM_CHANNEL, OBS_SIZE, GLOBAL_NET_SCOPE)
        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)

    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if not hasattr(self, "lock_bool"):
            self.lock_bool = False
        self.groupLock.release(int(self.lock_bool), self.name)
        self.groupLock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool

    def train(self, rollout, sess, gamma, bootstrap_value, rnn_state0, imitation=False):
        global episode_count
        if imitation:
            rollout = np.array(rollout)
            # we calculate the loss differently for imitation
            # if imitation=True the rollout is assumed to have different dimensions:
            # [o[0],o[1],optimal_actions]
            rnn_state = self.local_AC.state_init
            feed_dict = {global_step                  : episode_count,
                         self.local_AC.inputs         : np.stack(rollout[:, 0]),
                         self.local_AC.goal_pos       : np.stack(rollout[:, 1]),
                         self.local_AC.optimal_actions: np.stack(rollout[:, 2]),
                         self.local_AC.state_in[0]    : rnn_state0[0],
                         self.local_AC.state_in[1]    : rnn_state0[1],
                         self.local_AC.train_imitation : (rollout[:, 3])
                         }
            _, i_l, _ = sess.run([self.local_AC.policy, self.local_AC.imitation_loss,
                                  self.local_AC.apply_imitation_grads],
                                 feed_dict=feed_dict)
            return i_l
        else:
            rollout = np.array(rollout)
            observations = rollout[:, 0]
            goals = rollout[:, -3]
            actions = rollout[:, 1]
            rewards = rollout[:, 2]
            values = rollout[:, 4]
            valids = rollout[:, 5]
            train_value = rollout[:, -2]
            train_policy = rollout[:,-1] 

            # Here we take the rewards and values from the rollout, and use them to
            # generate the advantage and discounted returns. (With bootstrapping)
            # The advantage function uses "Generalized Advantage Estimation"
            self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
            discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
            self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
            advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
            advantages = discount(advantages, gamma)

            num_samples = min(EPISODE_SAMPLES, len(advantages))
            sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

            rnn_state = self.local_AC.state_init
            feed_dict = {
                global_step              : episode_count,
                self.local_AC.target_v   : np.stack(discounted_rewards),
                self.local_AC.inputs     : np.stack(observations),
                self.local_AC.goal_pos   : np.stack(goals),
                self.local_AC.actions    : actions,
                self.local_AC.train_valid: np.stack(valids),
                self.local_AC.advantages : advantages,
                self.local_AC.train_value: train_value,
                self.local_AC.state_in[0]: rnn_state0[0],
                self.local_AC.state_in[1]: rnn_state0[1],
                self.local_AC.train_policy: train_policy,
                self.local_AC.train_valids : np.vstack(train_policy) 
            }

            v_l, p_l, valid_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                            self.local_AC.policy_loss,
                                                            self.local_AC.valid_loss,
                                                            self.local_AC.entropy,
                                                            self.local_AC.grad_norms,
                                                            self.local_AC.var_norms,
                                                            self.local_AC.apply_grads],
                                                           feed_dict=feed_dict)
            return v_l, p_l, valid_l, e_l, g_n, v_n

    def shouldRun(self, coord, episode_count=None):
        if TRAINING:
            return not coord.should_stop()

    def parse_path(self,episode_count):
        """needed function to take the path generated from M* and create the
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode":
                list of length num_agents with each sublist a list of tuples
                (observation[0],observation[1],optimal_action,reward)"""   

        global ACTION_SKIPPING, GIF_frames, SAVE_IL_GIF , IL_GIF_PROB 
        saveGIF= False  
        if np.random.rand() < IL_GIF_PROB  : 
            saveGIF= True     
        if saveGIF and SAVE_IL_GIF:    
          GIF_frames = [self.env._render()]       
        result  = [[] for i in range(num_workers)]
        actions = {} 
        o       = {}
        finished = {}
        train_imitation = {} 
        count_finished  = 0 
        pos_buffer = [] 
        goal_buffer  = [] 
        all_obs = self.env._observe()
        for agentID in range(1, num_workers + 1):
            o[agentID] = all_obs[agentID]
            train_imitation[agentID] = 1 
            finished[agentID] = 0 
        step_count = 0 
        while step_count <= max_episode_length and count_finished<num_workers :
            path = self.env.expert_until_first_goal()
            if path is None:  # solution not exists
                if step_count !=0 :
                    return result, 0 
                print('Failed intially')     
                return None, 0      
            none_on_goal = True
            path_step = 1  
            while none_on_goal and step_count <= max_episode_length and count_finished<num_workers :
                positions= []
                goals=[]  
                for i in range(num_workers):
                    agent_id = i+1
                    if finished[agent_id] :
                        actions[agent_id] = 0
                    else :     
                        next_pos = path[path_step][i]
                        diff = tuple_minus(next_pos, self.env.world.getPos(agent_id))  
                        try :
                            actions[agent_id] = dir2action(diff)
                        except :
                            print(pos_buffer) 
                            print(goal_buffer) 
                            actions[agent_id] = dir2action(diff)                                 
                all_obs, _ = self.env.step_all(actions)
                for i in range(num_workers) :
                    agent_id = i+1
                    positions.append(self.env.world.getPos(agent_id)) 
                    goals.append(self.env.world.getGoal(agent_id))                    
                    result[i].append([o[agent_id][0], o[agent_id][1], actions[agent_id],train_imitation[agent_id]])
                    if self.env.world.agents[agent_id].status >= 1 and finished[agent_id]!=1:
                        # none_on_goal = False
                        finished[agent_id] = 1 
                        count_finished +=1 
                pos_buffer.append(positions)   
                goal_buffer.append(goals)   
                if saveGIF and SAVE_IL_GIF:   
                    GIF_frames.append(self.env._render())         
                o = all_obs
                step_count += 1
                path_step += 1  
        if saveGIF and SAVE_IL_GIF :
            make_gif(np.array(GIF_frames),
                                     '{}/episodeIL_{}.gif'.format(gifs_path,episode_count))       
        return result,count_finished
      
    def work(self, max_episode_length, gamma, sess, coord, saver):

        global GIF_frames,CHANGE_FREQUENCY,GIF_FREQUENCY, PURE_RL_FUNCTIONALITY, IL_agents_done, episode_count, swarm_reward, swarm_targets, episode_rewards, episode_finishes, episode_lengths, episode_mean_values,episode_stop_ops, episode_invalid_ops, episode_wrong_blocking,env_params  # , episode_invalid_goals
        last_reset = 0 # Local Variable Used For Tracking Curriculum Updates
        
        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)
                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = targets_done =episode_stop_count= 0
            
                # Initial state from the environment
                if self.agentID == 1:
                    global demon_probs, IL_DECAY_RATE , Prob_Demonstration
                    Prob_Demonstration = DEMONSTRATION_PROB * np.exp(episode_count*IL_DECAY_RATE)
                    self.env._reset()
                    joint_observations[self.metaAgentID] = self.env._observe()
                    demon_probs[self.metaAgentID] = np.random.rand()  # for IL possibility

                self.synchronize()  # synchronize starting time of the threads

                # Get Information For Each Agent 
                validActions = self.env.listValidActions(self.agentID,joint_observations[self.metaAgentID][self.agentID])
                s = joint_observations[self.metaAgentID][self.agentID]

                rnn_state = self.local_AC.state_init
                rnn_state0 = rnn_state

                self.synchronize()  # synchronize starting time of the threads
                swarm_reward[self.metaAgentID] = 0
                swarm_targets[self.metaAgentID] = 0
# ===============================start training =======================================================================
                # IL
                if episode_count < PRIMING_LENGTH or (demon_probs[self.metaAgentID] < Prob_Demonstration):
                    global rollouts
                    rollouts[self.metaAgentID] = None
                    if self.agentID == 1:
                        rollouts[self.metaAgentID],agents_done = self.parse_path(episode_count)
                        IL_agents_done.append(agents_done)
                    self.synchronize()
                    if rollouts[self.metaAgentID] is not None:
                        i_l = self.train(rollouts[self.metaAgentID][self.agentID - 1], sess, gamma, None,
                                         rnn_state0,
                                         imitation=True)
                        if self.agentID == 1:
                            episode_count += 1
                            print('Episode Number:', episode_count, ' IL',
                                  ' Environment Number:', self.metaAgentID, 'Agents Done:', agents_done)
                            if int(episode_count) % 50 == 0:
                                print('Saving Model', end='\n')
                                saver.save(sess, model_path + '/model-' + str(int(episode_count)) + '.cptk')
                                print('Saved Model', end='\n')
                            summary = tf.Summary()
                            summary.value.add(tag='Losses/Imitation loss', simple_value=i_l)
                            if int(episode_count) > 50 :
                                IL_summary = IL_agents_done[-SUMMARY_WINDOW:] 
                                IL_summary = np.nanmean(IL_summary)
                                summary.value.add(tag='Perf/Targets Done by IL', simple_value=IL_summary)
                            global_summary.add_summary(summary, int(episode_count))
                            global_summary.flush()
                        continue
                    continue
                # RL
                else:
                    # prepare to save GIF
                    saveGIF = False
                    if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                        saveGIF = True
                        self.nextGIF = episode_count + GIF_FREQUENCY
                        GIF_episode = int(episode_count)
                        GIF_frames = [self.env._render()]

                    # start RL
                    self.env.finished = False
                    agent_done = False 
                    while not self.env.finished:  # Give me something!
                        if not agent_done : 
                            a_dist, v, rnn_state = sess.run([self.local_AC.policy,
                                                            self.local_AC.value,
                                                            self.local_AC.state_out],
                                                            feed_dict={self.local_AC.inputs     : [s[0]],  # state
                                                                    self.local_AC.goal_pos   : [s[1]],  # goal vector
                                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                                    self.local_AC.state_in[1]: rnn_state[1]})
                            train_policy = train_val = 1                      
                       
                        if not agent_done :
                            if not (np.argmax(a_dist.flatten()) in validActions):
                                episode_inv_count += 1
                                train_val = 0 
                            train_valid = np.zeros(a_size)
                            train_valid[validActions] = 1

                            valid_dist = np.array([a_dist[0, validActions]])
                            valid_dist /= np.sum(valid_dist)

                            a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                            joint_actions[self.metaAgentID][self.agentID] = a
                            if a == 0 :
                                episode_stop_count += 1

                        # Make A Single Agent Gather All Information

                        self.synchronize()

                        if self.agentID == 1:
                            all_obs, all_rewards = self.env.step_all(joint_actions[self.metaAgentID])
                            for i in range(1, num_workers+1):
                                joint_observations[self.metaAgentID][i] = all_obs[i]
                                joint_rewards[self.metaAgentID][i]      = all_rewards[i]
                                joint_done[self.metaAgentID][i]         = (self.env.world.agents[i].status >=1)
                            if saveGIF and self.workerID == 1:
                                GIF_frames.append(self.env._render())

                        self.synchronize()  # synchronize threads

                        # Get observation,reward, valid actions for each agent 
                        s1           = joint_observations[self.metaAgentID][self.agentID]
                        r            = copy.deepcopy(joint_rewards[self.metaAgentID][self.agentID]) 
                        if not agent_done :
                            validActions = self.env.listValidActions(self.agentID, s1)

                        self.synchronize() 
                        # Append to Appropriate buffers 
                        if not agent_done :
                            episode_buffer.append([s[0], a, joint_rewards[self.metaAgentID][self.agentID] , s1, v[0, 0], train_valid, s[1], train_val,train_policy])
                            episode_values.append(v[0, 0])
                        episode_reward += r
                        episode_step_count += 1

                        # Update State
                        s = s1
                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if (not agent_done) and (len(episode_buffer)>1) and ((len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0) or joint_done[self.metaAgentID][self.agentID] or episode_step_count==max_episode_length):
                            # Since we don't know what the true final return is,
                            # we "bootstrap" from our current value estimation.
                            if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                                train_buffer = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                            else:
                                train_buffer = episode_buffer[:]    
                            if joint_done[self.metaAgentID][self.agentID]:
                                s1Value        = 0       # Terminal state
                                episode_buffer = [] 
                                targets_done   += 1
                            else:
                                s1Value = sess.run(self.local_AC.value,
                                                   feed_dict={self.local_AC.inputs     : np.array([s[0]]),
                                                              self.local_AC.goal_pos   : [s[1]],
                                                              self.local_AC.state_in[0]: rnn_state[0],
                                                              self.local_AC.state_in[1]: rnn_state[1]})[0, 0]

                            v_l, p_l, valid_l, e_l, g_n, v_n = self.train(train_buffer, sess, gamma,
                                                                          s1Value, rnn_state0)

                            rnn_state0 = rnn_state

                        self.synchronize()

                        # finish condition: reach max-len or all agents are done under one-shot mode
                        if joint_done[self.metaAgentID][self.agentID] :
                            agent_done = True 
                        if episode_step_count >= max_episode_length or (
                                IS_ONESHOT and all([self.env.world.getDone(agentID) for agentID in range(1, num_agents + 1)]) is True):

                            break

                    # Add to appropriate buffers at the end of episode 
                    episode_lengths[self.metaAgentID].append(episode_step_count)
                    episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                    episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                    episode_stop_ops[self.metaAgentID].append(episode_stop_count)
                    swarm_reward[self.metaAgentID] += episode_reward
                    swarm_targets[self.metaAgentID] += targets_done

                    self.synchronize()  # synchronize threads

                    if self.agentID == 1:
                        episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])
                        episode_finishes[self.metaAgentID].append(swarm_targets[self.metaAgentID])
                        episode_count += 1
                        print('Episode Number:', episode_count,'Steps Taken:', episode_step_count, 'Targets Done:', swarm_targets[self.metaAgentID], ' Environment Number:', self.metaAgentID)

                        if episode_count % SUMMARY_WINDOW == 0:
                            if int(episode_count) % 100 == 0:
                                print('Saving Model', end='\n')
                                saver.save(sess, model_path + '/model-' + str(int(episode_count)) + '.cptk')
                                print('Saved Model', end='\n')
                            SL = SUMMARY_WINDOW * num_workers
                            SW = SUMMARY_WINDOW
                            mean_reward   = np.nanmean(episode_rewards[self.metaAgentID][-SW:])
                            mean_finishes = np.nanmean(episode_finishes[self.metaAgentID][-SW:])
                            mean_length   = np.nanmean(episode_lengths[self.metaAgentID][-SL:])
                            mean_value    = np.nanmean(episode_mean_values[self.metaAgentID][-SL:])
                            mean_invalid  = np.nanmean(episode_invalid_ops[self.metaAgentID][-SL:])
                            mean_stop     = np.nanmean(episode_stop_ops[self.metaAgentID][-SL:])
                            current_learning_rate = sess.run(lr, feed_dict={global_step: episode_count})

                            summary = tf.Summary()
                            summary.value.add(tag='Perf/Learning Rate', simple_value=current_learning_rate)
                            summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                            summary.value.add(tag='Perf/Targets Done', simple_value=mean_finishes)
                            summary.value.add(tag='Perf/Length', simple_value=mean_length)
                            summary.value.add(tag='Perf/Valid Rate', simple_value=(mean_length - mean_invalid) / mean_length)
                            summary.value.add(tag='Perf/Stop Rate', simple_value=(mean_stop) / mean_length)

                            summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                            summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                            summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                            summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                            summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                            global_summary.add_summary(summary, int(episode_count))

                            global_summary.flush()

                        if saveGIF and self.workerID == 1:
                            make_gif(np.array(GIF_frames),
                                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode, episode_step_count,
                                                                           swarm_reward[self.metaAgentID]))
                    self.synchronize()


# Learning parameters
gamma                   = .95  # discount rate for advantage estimation and reward discounting
LR_Q                    = 2.e-5  # 8.e-5 / NUM_THREADS # default: 1e-5
ADAPT_LR                = True
ADAPT_COEFF             = 5.e-5  # the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
EXPERIENCE_BUFFER_SIZE  = 256 
max_episode_length      = 256
episode_count           = 0

# observer parameters
OBS_SIZE                = 11   # the size of the FOV grid to apply to each agent
NUM_FUTURE_STEPS        = 3

# environment parameters
ENVIRONMENT_SIZE        = (10, 60)  # the total size of the environment (length of one side)
WALL_COMPONENTS         = (3, 21)
OBSTACLE_DENSITY        = (0.2, 0.7)  # range of densities
CHANGE_FREQUENCY        = 5000       # Frequency of Changing environment params  
DIAG_MVMT               = False  # Diagonal movements allowed?
a_size                  = 5 + int(DIAG_MVMT) * 4
NUM_META_AGENTS         = 1
NUM_THREADS             = 4 # int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS             = 1  # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)

# training parameters
SUMMARY_WINDOW          = 10
load_model              = False
RESET_TRAINER           = False
training_version        = 'VanillaOneShot' 
model_path              = 'model_' + training_version
gifs_path               = 'gifs_' + training_version
train_path              = 'train_' + training_version
OUTPUT_GIFS             = False  
GIF_FREQUENCY           = 512
SAVE_IL_GIF             = False   
IL_GIF_PROB             = 0  
IS_ONESHOT              = True

# Imitation options
PRIMING_LENGTH          = 0    # number of episodes at the beginning to train only on demonstrations
DEMONSTRATION_PROB      = 0.5   # probability of training on a demonstration per episode
IL_DECAY_RATE           = 0

# observation variables
NUM_CHANNEL             = 8 + NUM_FUTURE_STEPS

# others
EPISODE_START           = episode_count
TRAINING                = True
EPISODE_SAMPLES         = EXPERIENCE_BUFFER_SIZE  # 64
GLOBAL_NET_SCOPE        = 'global'
swarm_reward            = [0] * NUM_META_AGENTS
swarm_targets           = [0] * NUM_META_AGENTS

# Shared arrays for tensorboard

IL_agents_done         = []
episode_rewards         = [[] for _ in range(NUM_META_AGENTS)] 
episode_finishes        = [[] for _ in range(NUM_META_AGENTS)]
episode_lengths         = [[] for _ in range(NUM_META_AGENTS)]
episode_mean_values     = [[] for _ in range(NUM_META_AGENTS)]
episode_invalid_ops     = [[] for _ in range(NUM_META_AGENTS)]
episode_stop_ops        = [[] for _ in range(NUM_META_AGENTS)]
episode_wrong_blocking  = [[] for _ in range(NUM_META_AGENTS)]
rollouts                = [None for _ in range(NUM_META_AGENTS)]
demon_probs             = [np.random.rand() for _ in range(NUM_META_AGENTS)]
GIF_frames              = []

# Joint variables 
joint_actions           = [{} for _ in range(NUM_META_AGENTS)]
joint_env               = [None for _ in range(NUM_META_AGENTS)]
joint_observations      =[{} for _ in range(NUM_META_AGENTS)]
joint_rewards           = [{} for _ in range(NUM_META_AGENTS)]
joint_done              = [{} for _ in range(NUM_META_AGENTS)]

env_params              = [[ [WALL_COMPONENTS[0], WALL_COMPONENTS[1]] , [OBSTACLE_DENSITY[0],OBSTACLE_DENSITY[1]]]  for _ in range(NUM_META_AGENTS)]

tf.reset_default_graph()
print("Hello World")
for path in [train_path, model_path, gifs_path]:
    if not os.path.exists(path):
        os.makedirs(path)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE, a_size, None, False, NUM_CHANNEL, OBS_SIZE, GLOBAL_NET_SCOPE)

    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        # we need the +1 so that lr at step 0 is defined
        lr = tf.divide(tf.constant(LR_Q), tf.sqrt(tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), global_step))))
    else:
        lr = tf.constant(LR_Q)
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

    num_workers = NUM_THREADS  # Set workers # = # of available CPU threads
    gameEnvs, workers, groupLocks = [], [], []
    n = 1  # counter of total number of agents (for naming)
    for ma in range(NUM_META_AGENTS):
        num_agents = NUM_THREADS
        gameEnv = PRIMAL2Env(num_agents=num_agents,
                              observer=PRIMAL2Observer(observation_size=OBS_SIZE),
                              map_generator=maze_generator(
                                   env_size=ENVIRONMENT_SIZE,
                                   wall_components=WALL_COMPONENTS,
                                   obstacle_density=OBSTACLE_DENSITY),
                              IsDiagonal=DIAG_MVMT,
                              isOneShot=IS_ONESHOT)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_" + str(i) for i in range(n, n + num_workers)]
        groupLock = GroupLock.GroupLock([workerNames, workerNames])
        groupLocks.append(groupLock)

        # Create worker classes
        workersTmp = []
        for i in range(ma * num_workers + 1, (ma + 1) * num_workers + 1):
            workersTmp.append(Worker(gameEnv, ma, n, a_size, groupLock))
            n += 1
        workers.append(workersTmp)

    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')

            ckpt = tf.train.get_checkpoint_state(model_path)
            p = ckpt.model_checkpoint_path
            p = p[p.find('-') + 1:]
            p = p[:p.find('.')]
            episode_count = int(p)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("episode_count set to ", episode_count)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for ma in range(NUM_META_AGENTS):
            for worker in workers[ma]:
                groupLocks[ma].acquire(0, worker.name)  # synchronize starting time of the threads
                worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=worker_work)
                t.start()
                worker_threads.append(t)
        coord.join(worker_threads)


# In[ ]:
