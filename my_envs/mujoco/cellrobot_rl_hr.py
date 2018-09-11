import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env
from math import pi,sin,cos,atan

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

l= 0.2
traj = np.array([
       [0.00000000e+00*l, 6.12323400e-16*l, 0.200000000e+00],
       [3.01536896e-01*l, 1.71010072e+00*l, 0.200000000e+00],
       [1.16977778e+00*l, 3.21393805e+00*l, 0.200000000e+00],
       [2.50000000e+00*l, 4.33012702e+00*l, 0.200000000e+00],
       [4.13175911e+00*l, 4.92403877e+00*l, 0.200000000e+00],
       [5.86824089e+00*l, 4.92403877e+00*l, 0.200000000e+00],
       [7.50000000e+00*l, 4.33012702e+00*l, 0.200000000e+00],
       [8.83022222e+00*l, 3.21393805e+00*l, 0.200000000e+00],
       [9.69846310e+00*l, 1.71010072e+00*l, 0.200000000e+00],
       [1.00000000e+01*l, 0.00000000e+00*l, 0.200000000e+00],
       # [ 1.00000000e+01, -6.12323400e-16,  1.00000000e+00],
       # [ 1.03015369e+01, -1.71010072e+00,  1.00000000e+00],
       # [ 1.11697778e+01, -3.21393805e+00,  1.00000000e+00],
       # [ 1.25000000e+01, -4.33012702e+00,  1.00000000e+00],
       # [ 1.41317591e+01, -4.92403877e+00,  1.00000000e+00],
       # [ 1.58682409e+01, -4.92403877e+00,  1.00000000e+00],
       # [ 1.75000000e+01, -4.33012702e+00,  1.00000000e+00],
       # [ 1.88302222e+01, -3.21393805e+00,  1.00000000e+00],
       # [ 1.96984631e+01, -1.71010072e+00,  1.00000000e+00],
       # [ 2.00000000e+01, -0.00000000e+00,  1.00000000e+00]
])

position_vector = [0.9005710154022419, 0.19157649858525766, 0.20363844865472536, -0.2618038524762938, -0.04764016477204058, -0.4923544636213292, -0.30514082693887024, 0.7692727139092137, 0.7172509186944478, -0.6176943450166859, -0.43476218435592706, 0.7667223977603919, 0.29081693103406536, 0.09086369237435465, 0.0, 0.0, -0.0171052262902362, 0.0, 0.0, 0.0, 0.0, 0.0004205454597565903, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6989070655586036, 1.231416257452789, 1.188419262405775, -1.0974581723778125, -1.023151598620554, -0.40304458466288917, 0.5513169936393982, 0.646385738643396, 1.3694066886743392, 0.7519699447089043, 0.06997050535309216, -1.5500743998481212, 0.8190474090403703]

def dis_point2line(point, line_points):
    a = np.array([point[0] - line_points[0][0], point[1] - line_points[0][1]])
    b = np.array([line_points[1][0] - line_points[0][0], line_points[1][1] - line_points[0][1]])
    tmp = a.dot(b) / b.dot(b)
    c = b.dot(tmp)
    dis = np.sqrt((a - c).dot(a - c))
    return dis, np.sqrt(c.dot(c)), c + line_points[0][:2]

class CellRobotRLHrEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.t = None
        self.N_point = traj.shape[0]
        self.Lines_list = []
        for i in range(self.N_point - 1):
            self.Lines_list.append((traj[i], traj[i + 1]))
            
        self.goal_theta =  pi/4.0
        self.quat_init = [0.49499825, -0.49997497, 0.50500175, 0.49997499]
        
        self.CPG_controller = CPG_network(position_vector)
        
        mujoco_env.MujocoEnv.__init__(self, 'cellrobot/cellrobot_Quadruped_float_traj.xml',10)  # cellrobot_test_gen  CR_quadruped_v1_A001  'cellrobot/cellrobot_test_gen.xml' Atlas_v5/atlas_v5.xml
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
        
        # done = False
        

        if self.t is None:
            self.t = 0
            self.cur_line = 0
        else:
            self.t += 1

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.1 and state[2] <= 0.6
        done = not notdone
        
        if self._Is_curline_done((xposafter, yposafter)):
            if self.cur_line == self.N_point - 1:
                done = True
            else:
                self.cur_line += 1

        
        proj_verbefore, proj_parbefore, _ = dis_point2line((xposbefore, yposbefore), self.Lines_list[self.cur_line])
        proj_verafter, proj_parafter, _ = dis_point2line((xposafter, yposafter), self.Lines_list[self.cur_line])

        parall = 1 * (proj_parafter - proj_parbefore)/ self.dt
        vert = 5 * proj_verafter
        forward_reward = parall - vert
        reward = forward_reward + survive_reward

        ob = self._get_obs()
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
        
        object_x = self.get_body_com("torso")[1]
        object_y = self.get_body_com("torso")[0]

        target_y = traj[self.cur_line + 1][1]
        target_x = traj[self.cur_line + 1][0]
        
        self.goal_theta = atan((target_y-object_y)/(target_x-object_x))
        #print(self.goal_theta/3.1415*180.0)
        return np.concatenate([
            self.get_body_com("torso").flat,
            self.sim.data.qpos.flat[:7],  # 3:7 表示角度
            np.array(angle),
            np.array([angle[2] - self.goal_theta])
        ])
    
    def reset_model(self, reset_args=None):
        qpos = self.init_qpos  # + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        #qpos[:7] = qpos[:7] +self.np_random.uniform(size=7, low=-.1, high=.1)
        qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        
        if reset_args is not None:
            self.goal_theta = reset_args
            print('goal theta = ', reset_args)
        self.CPG_controller = CPG_network(position_vector)

        #self.t =0
        self._init_goal()
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

    def _init_goal(self):
        # alpha = np.linspace(-np.pi, 0, self.N_point)
        # a = 5
        #
        # # 1. circle traj.
        # # x_point = (a * np.cos(alpha) + a).reshape((-1, 1))
        # # y_point = (a * np.sin(alpha)).reshape((-1, 1))
        #
        # # 2. sin traj.
        # x_point = ( 2*alpha ).reshape((-1, 1))
        # y_point = (a * np.sin(alpha)).reshape((-1, 1))
        #
        # z_point = np.ones(self.N_point).reshape((-1, 1))
        #
        # point = np.concatenate((x_point, y_point, z_point), 1)
        #
        # if self.model.site_pos.shape[0] != self.N_point:
        #     assert print('The shape of trajactory does not match site !')
    
        for i in range(self.N_point):
            self.model.site_pos[i] = traj[i]

    def _Is_curline_done(self, xy_pose):
        (x, y) = xy_pose
        safe_distance = 0.1
        
        if self.cur_line == 0:
            dis_goal_vec_0 = np.array(
                [x - self.Lines_list[self.cur_line][0][0], y - self.Lines_list[self.cur_line][0][1]])
            if np.linalg.norm(dis_goal_vec_0) < safe_distance:
                self.model.site_rgba[self.cur_line] = np.array([0, 1, 0, 1])  # green
    
        dis_goal_vec = np.array([x - self.Lines_list[self.cur_line][1][0], y - self.Lines_list[self.cur_line][1][1]])
    
        if np.linalg.norm(dis_goal_vec) < safe_distance:
            self.model.site_rgba[self.cur_line + 1] = np.array([0, 1, 0, 1])  # green
            return True
        else:
            return False