import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from math import pi,sin,cos

from CPG_core.PID_controller import PID_controller

from CPG_core.math.transformation import euler_from_quaternion,quaternion_inverse ,quaternion_multiply

# choose your CPG network
# from CPG_core.controllers.CPG_controller_quadruped_sin import CPG_network
from CPG_core.controllers.CPG_controller_bigdog2_sin import CPG_network

state_M = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.]])

position_vector = [0.6429382282419057, -1.504515744295227, -1.2122146768896522, -1.5430866210080398, -0.40339625754811115, 1.797866485292635, 1.8465815905382332, 1.3156099280263092, 1.1815248128471576, 0.4557141890654699, -1.3136558716839821, -1.383742714528093, 1.0666337767012442, -0.46609084892228037, 0.5266071808950659, 0.008279403737123438, 0.0, 0.0, 0.0, 0.011237911878610663, -0.0146826210838802, 0.0, 0.0, 0.0, 0.0, -0.017814994116624273, 0.0, 0.018121282385488727, 0.0, -0.3242713509141666, -0.5905465210728494, -1.5530731911249718, 0.7060008434892526, 0.6718690361326529, 0.30814153016454116, -0.2900699626568739, -1.4214811438222459, -0.8181964756164031, -0.9037143779342285, -0.6716727364566586, -1.0711308729593805, -1.464835073477411, 0.2443659340371438]


CPG_node_num = 14
#position_vector =[0.9858483561978868, -0.26776244703081553, 0.06341770508024377, 0.1803821478742611, -0.737421299071324, 0.36522978966198716, 0.5044235400002197, 0.40991580439035835, -0.8443021471919203, 0.04228059212402413, 0.7554420347469687, -0.5639823285279588, -0.2670696994557711, 0.15494509911424004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0018091287679612926, 0.0, 0.0, 0.0025530353527089796, 0.0, 0.0, 0.0, 0.0, -0.5828629636313896, -1.0350978973515126, 0.2869339954154033, -0.14215879560893563, -1.0662735162976038, -0.3594286719244508, -1.3610669510250777, 1.1259577909453355, 0.7737809075621813, -0.3512915481737431, 0.00528021283053004, 0.6727569875008399, -0.10960030747213767,]

class CellRobotRLBigDog2Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal_theta =  0/4.0
        self.quat_init = [ 3.82689422e-01,-3.71637028e-06,-1.52615385e-06, 9.23877052e-01]
        self.t =0
        self.CPG_controller = CPG_network(CPG_node_num,position_vector)
        
        mujoco_env.MujocoEnv.__init__(self, 'cellrobot/cellrobot_BigDog2_float.xml',10)  # cellrobot_test_gen  CR_quadruped_v1_A001  'cellrobot/cellrobot_test_gen.xml' Atlas_v5/atlas_v5.xml
        utils.EzPickle.__init__(self)
        
        
        
    def step(self, a):
        
        action = self.CPG_transfer(a )
         
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
        self.CPG_controller = CPG_network(CPG_node_num,position_vector)

        self.t =0
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def CPG_transfer(self,RL_output ):
        #print(RL_output)
        self.CPG_controller.update(RL_output)
        # if self.t % 100 == 0:
        #     #CPG_controller.update(RL_output)
        #     print(RL_output)
        ###adjust CPG_neutron parm using RL_output
        output_list = self.CPG_controller.output(state=None)
        # ignore some cell
        target_joint_angles = np.array( [output_list[1], 0, 0, 0, output_list[2],
                               output_list[3],output_list[4],output_list[5],output_list[6],
                               output_list[7],output_list[8],output_list[9],output_list[10],
                               output_list[11],output_list[12],output_list[13],output_list[14],])

       
        
        cur_angles = np.concatenate([state_M.dot(self.sim.data.qpos[7:].reshape((-1, 1))).flat])
        action = PID_controller(cur_angles, target_joint_angles)
        return action