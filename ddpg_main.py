import argparse
import time
import os
import logging
from test.baselines import logger, bench
from test.baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import test.baselines.ddpg.training as training
#import test.baselines.ddpg.training_hr as training
from test.baselines.ddpg.models import Actor, Critic
from test.baselines.ddpg.memory import Memory
from test.baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI

def run(env_id, seed, noise_type, layer_norm, evaluation, out_layer, action_dim=2,  **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        #eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        os.mkdir(os.path.join(logger.get_dir(), 'model'))
        eval_env = gym.wrappers.Monitor(eval_env, os.path.join(logger.get_dir(), 'model'), force=True )
        
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = action_dim   #env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=(nb_actions,), observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm, out_layer = out_layer)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory,action_dim=nb_actions, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

from my_envs.mujoco import *

## mpiexec -n 8 python3 test/baselines/ddpg/main.py
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='CellRobotRL2Env-v0') #CellRobotRLEnv-v0   HalfCheetah-v2 CellRobotRLEnv CellRobotRLBigdog2Env-v0
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=100)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=10)#20
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=200)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=1000)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='normal_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--action-dim', type=int, default=2)
    parser.add_argument('--group-dir', type=str, default=None)
    parser.add_argument('--out-layer', type=str, default='tanh')  #sigmoid
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    print(args)
    
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args
    

import os.path as osp
import json
import time
import datetime
if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        if args['group_dir'] is not None:
            dir = osp.join(args['group_dir'], args['env_id'],
                           datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
        else:
            dir = osp.join('log_files',args['env_id'],
                           datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
        
        logger.configure(dir)

    logger.info(args)
    # Run actual script.
    run(**args)
