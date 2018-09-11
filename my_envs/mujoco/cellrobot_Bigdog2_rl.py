import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from math import pi,sin,cos

from CPG_core.PID_controller import PID_controller

from CPG_core.math.transformation import euler_from_quaternion,quaternion_inverse ,quaternion_multiply

# choose your CPG network
# from CPG_core.controllers.CPG_controller_quadruped_sin import CPG_network
from CPG_core.controllers.CPG_controller_quadruped_sin import CPG_network

state_M =np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])

position_vector = [0.9005710154022419, 0.19157649858525766, 0.20363844865472536, -0.2618038524762938, -0.04764016477204058, -0.4923544636213292, -0.30514082693887024, 0.7692727139092137, 0.7172509186944478, -0.6176943450166859, -0.43476218435592706, 0.7667223977603919, 0.29081693103406536, 0.09086369237435465, 0.0, 0.0, -0.0171052262902362, 0.0, 0.0, 0.0, 0.0, 0.0004205454597565903, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6989070655586036, 1.231416257452789, 1.188419262405775, -1.0974581723778125, -1.023151598620554, -0.40304458466288917, 0.5513169936393982, 0.646385738643396, 1.3694066886743392, 0.7519699447089043, 0.06997050535309216, -1.5500743998481212, 0.8190474090403703]

#position_vector =[0.9858483561978868, -0.26776244703081553, 0.06341770508024377, 0.1803821478742611, -0.737421299071324, 0.36522978966198716, 0.5044235400002197, 0.40991580439035835, -0.8443021471919203, 0.04228059212402413, 0.7554420347469687, -0.5639823285279588, -0.2670696994557711, 0.15494509911424004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0018091287679612926, 0.0, 0.0, 0.0025530353527089796, 0.0, 0.0, 0.0, 0.0, -0.5828629636313896, -1.0350978973515126, 0.2869339954154033, -0.14215879560893563, -1.0662735162976038, -0.3594286719244508, -1.3610669510250777, 1.1259577909453355, 0.7737809075621813, -0.3512915481737431, 0.00528021283053004, 0.6727569875008399, -0.10960030747213767,]

class CellRobotRLBigDog2Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal_theta =  -pi/4.0
        self.quat_init = [0.49499825, -0.49997497, 0.50500175, 0.49997499]
        self.t =0
        self.CPG_controller = CPG_network(position_vector)
        
        mujoco_env.MujocoEnv.__init__(self, 'cellrobot/cellrobot_BigDog2_float.xml',10)  # cellrobot_test_gen  CR_quadruped_v1_A001  'cellrobot/cellrobot_test_gen.xml' Atlas_v5/atlas_v5.xml
        utils.EzPickle.__init__(self)
        
        
        
    def step(self, a):
        
        action = self.CPG_transfer(a, self.CPG_controller )
         
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        theta = self.goal_theta
        comvel_xy_before = np.array([xposbefore, yposbefore])
        u = np.array([cos(theta), sin(theta)])
        proj_parbefore = comvel_xy_before.dot(np.transpose(u))
        
        self.do_simulation(action, self.frame_skip)
        
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        
        comvel_xy_after = np.array([xposafter, yposafter])
        # u = np.array([cos(theta), sin(theta)])
        proj_parafter = comvel_xy_after.dot(np.transpose(u))
        proj_ver = abs(u[0] * comvel_xy_after[1] - u[1] * comvel_xy_after[0])
        forward_reward = 1 * (proj_parafter - proj_parbefore)/ self.dt  - 5 * proj_ver  #/ self.dt
        #forward_reward = 1 *  proj_parafter   - 5 * proj_ver  # / self.dt
        
        ctrl_cost = 0
        contact_cost = 0
        survive_reward = 0
        reward = forward_reward+survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.1 and state[2] <= 0.6
        done = not notdone
        # done = False
        ob = self._get_obs()
        self.t += 1
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)
    
    def _get_obs(self):
        quat = self.sim.data.qpos.flat[3:7]
        #print('quat = ', quat)
        quat_tranfor = quaternion_multiply(quat, quaternion_inverse(self.quat_init))
        angle = euler_from_quaternion(quat_tranfor, 'rxyz') # radian
        # angles = []
        # for i in range(3):
        #     angles.append(angle[i] / pi * 180.0)
        # print(angles)  # angle
        
        return np.concatenate([
            self.get_body_com("torso").flat,
            #self.sim.data.qpos.flat[:3],  # 3:7 表示角度
            #self.sim.data.qpos.flat[:7],  # 3:7 表示角度
            np.array(angle),
            np.array( [ angle[2] -self.goal_theta])  # self.goal_theta
        ])
    
    def reset_model(self, reset_args=None):
        qpos = self.init_qpos  # + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qpos[:7] = qpos[:7] +self.np_random.uniform(size=7, low=-.05, high=.05)
        qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        
        if reset_args is not None:
            self.goal_theta = reset_args
            print('goal theta = ', reset_args)
        self.CPG_controller = CPG_network(position_vector)

        self.t =0
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def CPG_transfer(self,RL_output, CPG_controller ):
        #print(RL_output)
        CPG_controller.update(RL_output)
        # if self.t % 100 == 0:
        #     #CPG_controller.update(RL_output)
        #     print(RL_output)
        ###adjust CPG_neutron parm using RL_output
        output_list = CPG_controller.output(state=None)
        target_joint_angles = np.array(output_list[1:])# CPG 第一个输出为placemarke
        cur_angles = np.concatenate([state_M.dot(self.sim.data.qpos[7:].reshape((-1, 1))).flat])
        action = PID_controller(cur_angles, target_joint_angles)
        return action