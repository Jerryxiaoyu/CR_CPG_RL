import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
from torch.autograd import Variable
import math
import time

from CPG_core.controllers.CPG_controller_quadruped_sin import CPG_network
from CPG_core.PID_controller import PID_controller

obs_low = 6
obs_high = 19
def CPG_transfer(RL_output, CPG_controller, obs, t):
 
    
    #CPG_controller.update(RL_output)
    # adjust CPG_neutron parm using RL_output
    output_list = CPG_controller.output(state=None)
    target_joint_angles = np.array(output_list[1:])
    cur_angles = obs[obs_low:obs_high]
    action = PID_controller(cur_angles, target_joint_angles)
    return  action
 
def obs2state(obs):
     # root x y z alpha  beta gamma
    return obs[:6]

def collect_samples(pid, queue, env, policy, custom_reward, mean_action,
                    tensor, render, running_state, update_rs, min_batch_size, logger, position_vector,log_flag=False):
    torch.randn(pid, )
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    
    while num_steps < min_batch_size:
        obs = env.reset()
        #TODO 设置全局变量 position  vector
        #CPG_controller = CPG_network(position_vector)
        CPG_controller = CPG_network(position_vector)
        
        obs1, obs2, rewards, dones, actions,   = [], [], [], [], [],
        # TODO 观测量到NN输入 的转换函数
        state = obs2state(obs)

        obs2.append(state.reshape((1,-1))) # for storage
        obs1.append(obs.reshape((1, -1)))  # for storage
        if running_state is not None:
            state = running_state(state, update=update_rs)
        reward_episode = 0
        reward_period =0
        
        for t in range(10000):
            state_var = Variable(tensor(state).unsqueeze(0), volatile=True)
            if t%1 == 0:
                if mean_action:
                    action = policy(state_var)[0].data[0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()
                
                rl_action = int(action) if policy.is_disc_action else action.astype(np.float64)

            # if t%100 == 0:
            #     print('rl = ', rl_action)
            #rl_action = np.zeros(13)
            #rl_action = np.array([1,0])
            rl_action = np.clip(rl_action,0,1)
            action = CPG_transfer(rl_action, CPG_controller,obs, t)
            
            next_state, reward, done, _ = env.step(action)

            obs = next_state
            
            # transfer
            obs1.append(next_state.reshape((1, -1)))  # for storage
            next_state = obs2state(next_state)
            obs2.append(next_state.reshape((1, -1)))  # for storage
            
            actions.append(rl_action.reshape((1, -1)))
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state, update=update_rs)
            
            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)
            
            rewards.append(reward)  # for storage
            dones.append(done)  # for storage
            mask = 0 if done else 1
            
            memory.push(state, rl_action, mask, next_state, reward)
            
            if render:
                env.render()
            if done:
                break
            
            state = next_state
            
        
        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
        
        # log sample data ,just for debug
        if log_flag:
            rewards = np.array(rewards, dtype=np.float64)
            dones = np.array(dones, dtype=np.float64)
            tmp = np.vstack((rewards, dones))  # states_x, states_y,
            tmp1 = np.transpose(tmp)
            actions = np.concatenate(actions)
            
            obs1 = np.concatenate(obs1[:-1])
            obs2 = np.concatenate(obs2[:-1])
            data = np.concatenate((obs1, obs2, actions,   tmp1), axis=1)
            
            trajectory = {}
            for j in range(data.shape[0]):
                for i in range(data.shape[1]):
                    trajectory[i] = data[j][i]
                logger.log(trajectory)
                logger.write()
    
    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward
    
    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...
    
    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        
        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        
        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])
    
    return log


class Agent:
    
    def __init__(self, env_factory, policy, custom_reward=None, mean_action=False, render=False,
                 tensor_type=torch.DoubleTensor, running_state=None, num_threads=1, logger=None, position_vector=None):
        self.env_factory = env_factory
        self.policy = policy
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.tensor = tensor_type
        self.num_threads = num_threads
        self.env_list = []
        self.position_vector =position_vector
        
        if logger is not None:  # storage sampling data for debug
            self.logger = logger
            self.log_flag = True
        else:
            self.logger = None
            self.log_flag = False
        
        for i in range(num_threads):
            self.env_list.append(self.env_factory(i))
    
    def collect_samples(self, min_batch_size):
        t_start = time.time()
        if use_gpu:
            self.policy.cpu()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []
        
        for i in range(self.num_threads - 1):
            worker_args = (i + 1, queue, self.env_list[i + 1], self.policy, self.custom_reward, self.mean_action,
                           self.tensor, False, self.running_state, False, thread_batch_size, self.logger,self.position_vector, self.log_flag)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()
        
        memory, log = collect_samples(0, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action,
                                      self.tensor, self.render, self.running_state, True, thread_batch_size,
                                      self.logger, self.position_vector,log_flag=self.log_flag)
        
        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        if use_gpu:
            self.policy.cuda()
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
