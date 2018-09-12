#!/usr/bin/env python3
import argparse
from test.baselines.common.cmd_util import mujoco_arg_parser
from test.baselines import bench, logger

from my_envs.mujoco import *

import tensorflow as tf
import os
import time
import datetime
dir = os.path.join(os.getcwd(),'log-files',
                   datetime.datetime.now().strftime("ppo-%Y-%m-%d-%H-%M-%S-%f"))

def train(env_id, num_timesteps, seed, nsteps=2048, nminbatches =1024, noptepochs = 10,ncpu=8):
    from test.baselines.common import set_global_seeds
    from test.baselines.common.vec_env.vec_normalize import VecNormalize
    from test.baselines.ppo2 import ppo2
    from test.baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    
    from test.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = ncpu
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.per_process_gpu_memory_fraction = 1 / 2.
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=1024,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=1e-3,
        cliprange=0.2,
        total_timesteps=num_timesteps, action_dim= 2, save_interval= 10)

    Saver = tf.train.Saver(max_to_keep=10)
    Saver.save(sess, os.path.join(dir,  'trained_variables.ckpt'), write_meta_graph=False)





def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
   
    parser.add_argument('--env', help='environment ID', type=str, default='CellRobotRLEnv-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--nsteps', type=int, default=2048)
    parser.add_argument('--nminibatches', type=int, default=int(1024))
    parser.add_argument('--noptepochs', type=int, default=int(10))
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()
    print(args)
    logger.configure(dir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, nsteps = args.nsteps,nminbatches=args.nminibatches,
          noptepochs=args.noptepochs, ncpu = args.ncpu)


if __name__ == '__main__':
    main()
