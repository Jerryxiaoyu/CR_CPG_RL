import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env

# amp =5 circle
traj = np.array([
       [0.00000000e+00, 6.12323400e-16, 1.00000000e+00],
       [3.01536896e-01, 1.71010072e+00, 1.00000000e+00],
       [1.16977778e+00, 3.21393805e+00, 1.00000000e+00],
       [2.50000000e+00, 4.33012702e+00, 1.00000000e+00],
       [4.13175911e+00, 4.92403877e+00, 1.00000000e+00],
       [5.86824089e+00, 4.92403877e+00, 1.00000000e+00],
       [7.50000000e+00, 4.33012702e+00, 1.00000000e+00],
       [8.83022222e+00, 3.21393805e+00, 1.00000000e+00],
       [9.69846310e+00, 1.71010072e+00, 1.00000000e+00],
       [1.00000000e+01, 0.00000000e+00, 1.00000000e+00],
       [ 1.00000000e+01, -6.12323400e-16,  1.00000000e+00],
       [ 1.03015369e+01, -1.71010072e+00,  1.00000000e+00],
       [ 1.11697778e+01, -3.21393805e+00,  1.00000000e+00],
       [ 1.25000000e+01, -4.33012702e+00,  1.00000000e+00],
       [ 1.41317591e+01, -4.92403877e+00,  1.00000000e+00],
       [ 1.58682409e+01, -4.92403877e+00,  1.00000000e+00],
       [ 1.75000000e+01, -4.33012702e+00,  1.00000000e+00],
       [ 1.88302222e+01, -3.21393805e+00,  1.00000000e+00],
       [ 1.96984631e+01, -1.71010072e+00,  1.00000000e+00],
       [ 2.00000000e+01, -0.00000000e+00,  1.00000000e+00]
])

## amp =3 circle
# traj = np.array(
#     [[ 0.00000000e+00,  3.67394040e-16,  1.00000000e+00],
#        [ 1.80922138e-01,  1.02606043e+00,  1.00000000e+00],
#        [ 7.01866671e-01,  1.92836283e+00,  1.00000000e+00],
#        [ 1.50000000e+00,  2.59807621e+00,  1.00000000e+00],
#        [ 2.47905547e+00,  2.95442326e+00,  1.00000000e+00],
#        [ 3.52094453e+00,  2.95442326e+00,  1.00000000e+00],
#        [ 4.50000000e+00,  2.59807621e+00,  1.00000000e+00],
#        [ 5.29813333e+00,  1.92836283e+00,  1.00000000e+00],
#        [ 5.81907786e+00,  1.02606043e+00,  1.00000000e+00],
#        [ 6.00000000e+00,  0.00000000e+00,  1.00000000e+00],
#        [ 6.00000000e+00, -3.67394040e-16,  1.00000000e+00],
#        [ 6.18092214e+00, -1.02606043e+00,  1.00000000e+00],
#        [ 6.70186667e+00, -1.92836283e+00,  1.00000000e+00],
#        [ 7.50000000e+00, -2.59807621e+00,  1.00000000e+00],
#        [ 8.47905547e+00, -2.95442326e+00,  1.00000000e+00],
#        [ 9.52094453e+00, -2.95442326e+00,  1.00000000e+00],
#        [ 1.05000000e+01, -2.59807621e+00,  1.00000000e+00],
#        [ 1.12981333e+01, -1.92836283e+00,  1.00000000e+00],
#        [ 1.18190779e+01, -1.02606043e+00,  1.00000000e+00],
#        [ 1.20000000e+01, -0.00000000e+00,  1.00000000e+00]])


def dis_point2line(point, line_points):
    a = np.array([point[0] - line_points[0][0], point[1] - line_points[0][1]])
    b = np.array([line_points[1][0] - line_points[0][0], line_points[1][1] - line_points[0][1]])
    tmp = a.dot(b) / b.dot(b)
    c = b.dot(tmp)
    dis = np.sqrt((a - c).dot(a - c))
    return dis, np.sqrt(c.dot(c)), c + line_points[0][:2]

class AntTrajEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
       
        self.t = None
        self.N_point = traj.shape[0]
        self.Lines_list=[]
        for i in range(self.N_point-1):
            self.Lines_list.append(( traj[i], traj[i+1] ))
        
        mujoco_env.MujocoEnv.__init__(self, 'ant_traj.xml', 5)
        utils.EzPickle.__init__(self)
        
    def step(self, a):
        
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        
        
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        done = not notdone
        
        
        if self.t is None:
            self.t = 0
            self.cur_line = 0
        else:
            self.t +=1


        if self._Is_curline_done((xposafter,yposafter)):
            if self.cur_line == self.N_point-1:
                done = True
            else:
                self.cur_line += 1

        alpha =1
        beta =1
        proj_verbefore, proj_parbefore, _ = dis_point2line((xposbefore, yposbefore), self.Lines_list[self.cur_line])
        proj_verafter, proj_parafter, _ = dis_point2line((xposafter,yposafter), self.Lines_list[self.cur_line])

        parall = beta * (proj_parafter - proj_parbefore)
        vert = alpha * proj_verafter
        forward_reward =  parall -  vert

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        ob = self._get_obs()
        #print(' reward :{} parall :{} vert:{} forward_reward:{} contact_cost:{}  ctrl_cost:{} survive_reward{}'.format(reward,parall ,
        #                                                                                                                       vert,forward_reward,contact_cost,ctrl_cost,survive_reward))
        
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:],
            #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            (self.get_body_com("torso") - np.array(self.Lines_list[self.cur_line][1]) ).flat[:2],
        ])

    def reset_model(self,reset_args=None):
        
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
 
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        
        self._init_goal()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
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
        
        if self.model.site_pos.shape[0] != self.N_point:
            assert print('The shape of trajactory does not match site !')
        
        for i in range(self.N_point):
            self.model.site_pos[i] =  traj[i]
    def _Is_curline_done(self, xy_pose):
        ( x, y ) = xy_pose
        
        if self.cur_line == 0:
            dis_goal_vec_0 = np.array([x - self.Lines_list[self.cur_line][0][0] , y - self.Lines_list[self.cur_line][0][1] ])
            if np.linalg.norm(dis_goal_vec_0) < 0.25:
                self.model.site_rgba[self.cur_line  ] = np.array([0, 1, 0, 1])  # green
        
        dis_goal_vec = np.array([x - self.Lines_list[self.cur_line][1][0] , y - self.Lines_list[self.cur_line][1][1] ])
        
        if np.linalg.norm(dis_goal_vec) < 0.25:
            self.model.site_rgba[self.cur_line+1] = np.array([0, 1, 0, 1])  #green
            return True
        else:
            return False
        
   

    