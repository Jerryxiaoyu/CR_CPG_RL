#!/usr/bin/env python3

from test.baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from test.baselines.common import tf_util as U
from test.baselines import logger

import tensorflow as tf
import os
import time
import datetime

dir = os.path.join(os.getcwd(),'log-files',
                   datetime.datetime.now().strftime("ppo-%Y-%m-%d-%H-%M-%S-%f"))

def train(env_id, num_timesteps, seed):
    from test.baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    
    
    
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=5000,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=4048,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()
    writer = tf.summary.FileWriter('dir/log', sess.graph)  # write to file
    merge_op = tf.summary.merge_all()  # operation to merge all summary

    Saver = tf.train.Saver(max_to_keep=10)
    Saver.save(sess, os.path.join(dir,  'trained_variables.ckpt'), write_meta_graph=False)
    

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure(dir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
