from  instrument import VariantGenerator, variant, IO
import os
from datetime import datetime
import shutil
import glob
class VG(VariantGenerator):
    
    @variant
    def env_name(self):
        return [ 'CellRobotRLEnv-v0',    ]  # CellRobotRLHrEnv-v0  'CellRobotRLEnv-v0'
    @variant
    def batch_size(self):
        return [ 512,64 ]
    
    @variant
    def actor_lr(self):
        return [ 1e-4 ]

    @variant
    def critic_lr(self):
        return [1e-3]
    
    @variant
    def nb_epochs(self):
        return [ 50 ]

    @variant
    def nb_epoch_cycles(self):
        return [10 ]
 

    @variant
    def nb_train_steps(self):
        return [ 500, 1000]

    @variant
    def action_dim(self):
        return [2]

    @variant
    def noise_type(self):
        return ['normal_0.2', 'adaptive-param_0.2']


exp_id = 1
EXP_NAME ='DDPG_rl'
group_note ="************ABOUT THIS EXPERIMENT****************\n" \
            "测试所有环境是否可用!" \
            "测试 不同fitness 对cellrobot的影响"
n_cpu =8
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

#mpiexec -n 8 python3 test/baselines/ddpg/main.py

for v in variants:
    num_exp += 1
    print(v)
    # load parm
    env_name = v['env_name']
    batch_size = v['batch_size']
    actor_lr = v['actor_lr']
    critic_lr = v['critic_lr']
    nb_epochs = v['nb_epochs']
    nb_epoch_cycles = v['nb_epoch_cycles']

    nb_train_steps = v['nb_train_steps']
    action_dim = v['action_dim']
    noise_type= v['noise_type']

    os.system("nohup mpiexec -n 5"
              "  python3  ddpg_main.py " +
              " --seed " + str(seed) +
              " --env-id " + str(env_name) +
              " --batch-size " + str(batch_size) +
              " --actor-lr " + str(actor_lr) +
              " --critic-lr " + str(critic_lr) +
              " --nb-epochs " + str(nb_epochs) +
              " --nb-epoch-cycles " + str(nb_epoch_cycles) +
              " --nb-train-steps " + str(nb_train_steps)+
              " --action-dim " + str(action_dim)+
              " --noise-type " + str(noise_type)+
              " --group_dir " + str(group_dir)
                +
              " >log_files/ddpg.log </dev/null 2>&1 &"
              )
     