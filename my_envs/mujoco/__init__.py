from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv

from gym.envs.registration import registry, register, make, spec
from my_envs.mujoco.half_cheetah_DisableEnv import HalfCheetahEnvRandDisable
from my_envs.mujoco.half_cheetah_VaryingEnv import HalfCheetahVaryingEnv
from my_envs.mujoco.ant_DisableEnv import AntEnvRandDisable
from my_envs.mujoco.arm_reacher import ArmReacherEnv
from my_envs.mujoco.swimmer2 import SwimmerEnv
from my_envs.mujoco.ant2 import AntEnv
from my_envs.mujoco.half_cheetah2 import HalfCheetahEnv
from my_envs.mujoco.hopper2 import HopperEnv
from my_envs.mujoco.ant_traj import AntTrajEnv
from my_envs.mujoco.cellrobot import CellRobotEnv
from my_envs.mujoco.motor_test import MotorTestEnv
from my_envs.mujoco.cellrobot_arm import CellRobotArmEnv
from my_envs.mujoco.UR5 import UR5Env
from my_envs.mujoco.cellrobot_snake import CellRobotSnakeEnv
from my_envs.mujoco.cellrobot_butterfly import CellRobotButterflyEnv
from my_envs.mujoco.cellrobot_rl import CellRobotRLEnv
from my_envs.mujoco.cellrobot_rl_hr import CellRobotRLHrEnv
from my_envs.mujoco.cellrobot_Bigdog2_rl import CellRobotRLBigDog2Env
from my_envs.mujoco.cellrobot_rl2 import CellRobotRL2Env


register(
    id='HalfCheetahEnvDisableEnv-v0',
    entry_point='my_envs.mujoco:HalfCheetahEnvRandDisable',
    max_episode_steps=2000,
    reward_threshold=4800.0,
)

register(
    id='HalfCheetahVaryingEnv-v0',
    entry_point='my_envs.mujoco:HalfCheetahVaryingEnv',
    max_episode_steps=2000,
    reward_threshold=4800.0,
)

register(
    id='AntDisableEnv-v0',
    entry_point='my_envs.mujoco:AntEnvRandDisable',
    max_episode_steps=2000,
    reward_threshold=6000.0,
)

register(
    id='ArmReacherEnv-v0',
    entry_point='my_envs.mujoco:ArmReacherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Swimmer2-v2',
    entry_point='my_envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='HalfCheetah2-v2',
    entry_point='my_envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper2-v2',
    entry_point='my_envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id='Ant2-v2',
    entry_point='my_envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='AntTrajEnv-v0',
    entry_point='my_envs.mujoco:AntTrajEnv',
    max_episode_steps=2000,
    reward_threshold=6000.0,
)

register(
    id='CellrobotEnv-v0',
    entry_point='my_envs.mujoco:CellRobotEnv ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='MotorTestEnv-v0',
    entry_point='my_envs.mujoco:MotorTestEnv ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='CellrobotArmEnv-v0',
    entry_point='my_envs.mujoco:CellRobotArmEnv ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


register(
    id='UR5Env-v0',
    entry_point='my_envs.mujoco:UR5Env ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='CellrobotSnakeEnv-v0',
    entry_point='my_envs.mujoco:CellRobotSnakeEnv ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='CellrobotButterflyEnv-v0',
    entry_point='my_envs.mujoco:CellRobotButterflyEnv ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='CellRobotRLEnv-v0',
    entry_point='my_envs.mujoco:CellRobotRLEnv ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='CellRobotRLHrEnv-v0',
    entry_point='my_envs.mujoco:CellRobotRLHrEnv ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='CellRobotRLBigdog2Env-v0',
    entry_point='my_envs.mujoco:CellRobotRLBigDog2Env ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='CellRobotRL2Env-v0',
    entry_point='my_envs.mujoco:CellRobotRL2Env ',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)