from test.baselines.deepq import models  # noqa
from test.baselines.deepq.build_graph import build_act, build_train  # noqa
from test.baselines.deepq.simple import learn, load  # noqa
from test.baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from test.baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)