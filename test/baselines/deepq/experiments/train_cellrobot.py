import gym

from test.baselines import deepq

from my_envs.mujoco import *

def main():
    env = gym.make("CellRobotRLEnv-v0")  #CellRobotRLEnv-v0  MountainCar-v0
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64,64,32], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        param_noise=True
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
