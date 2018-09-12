from  instrument import VariantGenerator, variant, IO
import os
from datetime import datetime
import shutil
import glob
class VG(VariantGenerator):
    
    @variant
    def env_name(self):
        return [ 'CellRobotRLEnv-v0',   ]  # 'Cellrobot2Env-v0','CellrobotSnakeEnv-v0' , 'CellrobotSnake2Env-v0','CellrobotButterflyEnv-v0'
    @variant
    def log_std(self):
        return [-3,0 ]
    
    @variant
    def learning_rate(self):
        return [ 3e-4,   ]

    @variant
    def num_threads(self):
        return [1]
    
    @variant
    def min_batch_size(self):
        return [ 2048,20480 ]

    @variant
    def max_iter_num(self):
        return [100 ]
 

    @variant
    def action_dim(self):
        return [ 2  ]
 
    
exp_id = 1
EXP_NAME ='PPO_rl'
group_note ="************ABOUT THIS EXPERIMENT****************\n" \
            "测试所有环境是否可用!" \
            "测试 不同fitness 对cellrobot的影响"
            
variants = VG().variants()
num=0
for v in variants:
    num +=1
    print('exp{}: '.format(num), v)

# save gourp parms
exp_group_dir = datetime.now().strftime("%b_%d")+EXP_NAME+'_Exp{}'.format(exp_id)
group_dir = os.path.join('log-files', exp_group_dir)
os.makedirs(group_dir)

filenames = glob.glob('*.py')  # put copy of all python files in log_dir
for filename in filenames:  # for reference
    shutil.copy(filename, group_dir)

variants = VG().variants()
num = 0
param_dict = {}
for v in variants:
    num += 1
    print('exp{}: '.format(num), v)
    parm = v
    parm = dict(parm, **v)
    param_d = {'exp{}'.format(num): parm}
    param_dict.update(param_d)

IO('log-files/' + exp_group_dir + '/exp_id{}_param.pkl'.format(exp_id)).to_pickle(param_dict)
print(' Parameters is saved : exp_id{}_param.pkl'.format(exp_id))
# save args prameters
with open(group_dir + '/readme.txt', 'wt') as f:
    print("Welcome to Jerry's lab\n", file=f)
    print(group_note, file=f)
    
num_exp =0

seed =1

for v in variants:
    num_exp += 1
    print(v)
    # load parm
    env_name = v['env_name']
    log_std = v['log_std']
    learning_rate = v['learning_rate']
    num_threads = v['num_threads']
    min_batch_size = v['min_batch_size']
    max_iter_num = v['max_iter_num']

    action_dim = v['action_dim']
    

    os.system("python3  ../examples/ppo_gym_rl.py " +
              " --seed " + str(seed) +
              " --env-id " + str(env_name) +
              " --nsteps " + str(nsteps) +
              " --nminibatches " + str(nminibatches) +
              " --noptepochs " + str(noptepochs) +
              " --num-timesteps " + str(num-timesteps) +
              
 
              )
     