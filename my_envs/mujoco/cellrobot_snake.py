import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
state_M =np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

class CellRobotSnakeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'cellrobot/cellrobot_Snake_float.xml', 1)  #cellrobot_test_gen  CR_quadruped_v1_A001  'cellrobot/cellrobot_test_gen.xml' Atlas_v5/atlas_v5.xml
        utils.EzPickle.__init__(self)
        
    def step(self, a):
     
        
        self.do_simulation(a, self.frame_skip)
        forward_reward =1
        ctrl_cost =1
        contact_cost =1
        survive_reward =1
        reward = 1

        state = self.state_vector()
        # notdone = np.isfinite(state).all()  \
        #           and state[2] >= 0.1 and state[2] <= 0.6
        # done = not notdone
        done = False
        ob = self._get_obs()
        
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        # return  np.concatenate([
        #     state_M.dot( self.sim.data.qpos.reshape((-1, 1))).flat,
        #     state_M.dot(self.sim.data.qvel.reshape((-1, 1))).flat
        # ] )

        return np.concatenate([
            self.get_body_com("torso").flat,
            self.sim.data.qpos.flat[3:6],
            state_M.dot(self.sim.data.qpos[7:].reshape((-1, 1))).flat,
            state_M.dot(self.sim.data.qvel[6:].reshape((-1, 1))).flat
        ])
 

    def reset_model(self,reset_args=None):
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
