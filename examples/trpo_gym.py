import argparse
import gym
import os
import sys
import pickle
import time
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from torch.autograd import Variable
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent
from datetime import datetime
import matplotlib.pyplot as plt


from my_envs.mujoco import *

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="AntTrajEnv-v0", metavar='G',  #HalfCheetah-v2
					help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',   default=None,
					help='path of pre-trained model')
parser.add_argument('--render',  type=ast.literal_eval, default=False,
					help='render the environment')
parser.add_argument('--log-std', type=float, default=0.01, metavar='G',
					help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
					help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
					help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
					help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
					help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
					help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
					help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=5000, metavar='N',
					help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=100, metavar='N',
					help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=10, metavar='N',
					help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--monitor', type=ast.literal_eval, default=False,
					help="save gym mointor files (default: False, means don't save)")
parser.add_argument('--based-model', type=ast.literal_eval, default=False,
					help="Init params using model data (default: False, means don't init)")
parser.add_argument('--info', type=str, default='-generate-expert-Mb',
					help="Info)")
parser.add_argument('--store_data', type=ast.literal_eval, default=False,   ## note : most of time, False expect for storing the whole data!!
					help="store state action reward from sampling)")
args = parser.parse_args()

print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
	torch.cuda.manual_seed_all(args.seed)


"""create log-files"""
txt_note = 'Exp-TRPO'+args.info
log_name = datetime.now().strftime("%b-%d_%H:%M:%S") +'-'+txt_note
logdir = configure_log_dir(logname =args.env_name, name=log_name)

"""save args prameters"""
with open(logdir+'/info.txt', 'wt') as f:
	print('Hello World!\n', file=f)
	print(args, file=f)

"""create log.csv"""
logger = LoggerCsv(logdir, csvname='log_loss')
if args.store_data:
	logger_data = LoggerCsv(logdir, csvname='log_data')
else:
	logger_data = None

def env_factory(thread_id):
	env = gym.make(args.env_name)
	env.seed(args.seed + thread_id)

	"""Gym Monitor"""
	if args.monitor is True:
		monitor_path = os.path.join(log_dir(), args.env_name, log_name, 'monitor')
		env = gym.wrappers.Monitor(env, monitor_path, force=True,mode='training')
	return env

env_dummy = env_factory(0)
state_dim = env_dummy.observation_space.shape[0]
is_disc_action = len(env_dummy.action_space.shape) == 0
ActionTensor = LongTensor if is_disc_action else DoubleTensor

running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)


"""define actor and critic"""
if args.model_path is None:
	if is_disc_action:
		policy_net = DiscretePolicy(state_dim, env_dummy.action_space.n)
	else:
		policy_net = Policy(state_dim, env_dummy.action_space.shape[0], hidden_size=(500,500), activation='relu',log_std=args.log_std)
	value_net = Value(state_dim)
else:
	policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
	print('loaded pre_trained model!')

if args.based_model is True:
	policy_net.load_state_dict(torch.load(assets_dir()+'/MB_model/net_params2_bestA01-2.pkl')) #work
	print('loaded net params from model-training.')


if use_gpu:
	policy_net = policy_net.cuda()
	value_net = value_net.cuda()
del env_dummy


"""create agent"""
agent = Agent(env_factory, policy_net,running_state=running_state, render=args.render, num_threads=args.num_threads, logger= logger_data)


def update_params(batch):
	states = torch.from_numpy(np.stack(batch.state))
	actions = torch.from_numpy(np.stack(batch.action))
	rewards = torch.from_numpy(np.stack(batch.reward))
	masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
	if use_gpu:
		states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
	values = value_net(Variable(states, volatile=True)).data

	"""get advantage estimation from the trajectories"""
	advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

	"""perform TRPO update"""
	trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)

def plot_figure():
	import pandas as pd
	obj = pd.read_csv(logdir+"/log_loss.csv",header=0)

	plt.figure()
	plt.plot(obj['Iteration'], obj['AverageReward'],  'g', label='Reward')

	plt.grid(True)
	plt.xlabel('itr')
	plt.ylabel('reward')
	plt.legend(loc="upper right")
	#plt.show()
	plt.savefig( logdir+'/reward.jpg')

def main_loop():
	for i_iter in range(args.max_iter_num):
		"""generate multiple trajectories that reach the minimum batch_size"""
		batch, log = agent.collect_samples(args.min_batch_size)
		t0 = time.time()
		update_params(batch)
		t1 = time.time()

		if i_iter % args.log_interval == 0:
			print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
				i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

		#output data to log.csv
		logger.log({'Iteration': i_iter,
					'num_steps':log['num_steps'],
					'num_episodes':log['num_episodes'],
					'AverageReward': log['avg_reward'],
					'MinimumReward': log['min_reward'],
					'MaximumReward': log['max_reward'],
					'sample_time': log['sample_time'],
					})
		logger.write()

		if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
			if use_gpu:
				policy_net.cpu(), value_net.cpu()

			path = os.path.join(logdir, 'model')
			if not (os.path.exists(path)):
				os.makedirs(path)
			pickle.dump((policy_net, value_net, running_state),
						open(os.path.join(path, '{}_trpo.p'.format(args.env_name)), 'wb'))

			if use_gpu:
				policy_net.cuda(), value_net.cuda()

	#close log_file
	logger.close()
	logger_data.close()



print('Start time:\n')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
t0=datetime.now()

main_loop()

print('End time:\n')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
t1=datetime.now()
print("Toatal time is  ",(t1-t0).seconds/60,'min')
plot_figure()

