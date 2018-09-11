from CPG_core.CPG_Sin_osillator import sin_oscillator


from CPG_core.CPG_Sin_osillator import CPG_Sinneutron
class CPG_network(object):
    def __init__(self, CPG_node_num, position_vector):
        kf = position_vector[0]
        
        self.CPG_node_num = CPG_node_num   # 不包括placemarker
        
        if len(position_vector) != self.CPG_node_num *3+1:
            assert "Position vector out of range!"
            
        GAIN,BIAS,PHASE = [],[],[]
        
        for i in range(self.CPG_node_num):
            GAIN.append(position_vector[i+1])
            BIAS.append(position_vector[self.CPG_node_num+i+1])
            PHASE.append(position_vector[2 * self.CPG_node_num+i+1])
         
        
        
        self.parm_list = {
            0:  [0.0, 0.0, 0.0,    1.0, 0.0, 0],
        }
        
        for i in range(self.CPG_node_num):
            parm ={i+1:[0.0, 0.0, 0.0, GAIN[i], BIAS[i], PHASE[i]]}
            self.parm_list.update(parm)
        
        #print(parm_list)
        
        
        self.kf = position_vector[0]
        self.num_CPG = len(self.parm_list)
        self.CPG_list =[]
        #self.w_ms_list = [None, 1, 1,1,1, 1, 1, 1,  1, 1,1,1, 1, 1, 1,  ]
        self.w_ms_list = [None, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, ]
        self.master_list = [None, 0,0,1,3,3,1,6,6,2,9,9, 2,12,12   ]
        
        
        for i in range(self.num_CPG):
            if i == 0:
                self.CPG_list.append(CPG_Sinneutron(0, master_nuron = None, param=self.parm_list[0] ,kf= self.kf, w_ms = 0))
            else:
                self.CPG_list.append(CPG_Sinneutron(i, master_nuron=self.CPG_list[self.master_list[i]],
                                                 param=self.parm_list[i], kf=self.kf, w_ms= self.w_ms_list[i]))
    
    def output(self, state):
        output_list = []
        for cpg_n in self.CPG_list:
            cpg_n.next_output(f1=0, f2=0)
            output_list.append(cpg_n.parm['o'])
        
        return output_list

    def update(self, fi_l):
        self.kesi = 5
        if len(fi_l) == 2:
            
            # f_left = 1 - (0.5 - fi_l[0]) * self.kesi
            # f_right = 1 - (0.5 - fi_l[1]) * self.kesi
            # self.CPG_list[2].parm['f12'] = self.parm_list[2][5] * f_left
            # self.CPG_list[6].parm['f12'] = self.parm_list[6][5] * f_left
            # self.CPG_list[7].parm['f12'] = self.parm_list[7][5] * f_left
            #
            # self.CPG_list[3].parm['f12'] = self.parm_list[3][5] * f_right
            # self.CPG_list[8].parm['f12'] = self.parm_list[8][5] * f_right
            # self.CPG_list[9].parm['f12'] = self.parm_list[9][5] * f_right
            #
            # self.CPG_list[4].parm['f12'] = self.parm_list[4][5] * f_right
            # self.CPG_list[10].parm['f12'] = self.parm_list[10][5] * f_right
            # self.CPG_list[11].parm['f12'] = self.parm_list[11][5] * f_right
            #
            # self.CPG_list[5].parm['f12'] = self.parm_list[5][5] * f_left
            # self.CPG_list[12].parm['f12'] = self.parm_list[12][5] * f_left
            # self.CPG_list[13].parm['f12'] = self.parm_list[13][5] * f_left
    
            gain_left = 1 - (0.5 - fi_l[0]) * self.kesi
            gain_right = 1 - (0.5 - fi_l[1]) * self.kesi
    
            self.CPG_list[3].parm['R1'] = self.parm_list[3][3] * gain_left
            self.CPG_list[4].parm['R1'] = self.parm_list[4][3] * gain_left
            self.CPG_list[5].parm['R1'] = self.parm_list[5][3] * gain_left
    
            self.CPG_list[6].parm['R1'] = self.parm_list[6][3] * gain_left
            self.CPG_list[7].parm['R1'] = self.parm_list[7][3] * gain_left
            self.CPG_list[8].parm['R1'] = self.parm_list[8][3] * gain_left
    
            self.CPG_list[9].parm['R1'] = self.parm_list[9][3] * gain_right
            self.CPG_list[10].parm['R1'] = self.parm_list[10][3] * gain_right
            self.CPG_list[11].parm['R1'] = self.parm_list[11][3] * gain_right
    
            self.CPG_list[12].parm['R1'] = self.parm_list[12][3] * gain_right
            self.CPG_list[13].parm['R1'] = self.parm_list[13][3] * gain_right
            self.CPG_list[14].parm['R1'] = self.parm_list[14][3] * gain_right

        else:
            assert 'RL output error'
# import numpy as np
# position_vector = np.zeros(40)
# position_vector[0]=1
# for i in range(1,14):
#     position_vector[i] = 1
# CPG_network(position_vector)

class CPG_network5(object):
    def __init__(self, CPG_node_num, position_vector):
        kf = position_vector[0]
        
        self.CPG_node_num = CPG_node_num  # 不包括placemarker
        
        if len(position_vector) != self.CPG_node_num * 4 + 1:
            assert "Position vector out of range!"

        GAIN, BIAS, PHASE, WEIGHT = [], [], [], []

        for i in range(self.CPG_node_num):
            GAIN.append(position_vector[i + 1])
            BIAS.append(position_vector[self.CPG_node_num + i + 1])
            PHASE.append(position_vector[2 * self.CPG_node_num + i + 1])
            WEIGHT.append(position_vector[3 * self.CPG_node_num + i + 1])
        
        parm_list = {
            0: [0.0, 0.0, 0.0, 1.0, 0.0, 0],
        }
        
        for i in range(self.CPG_node_num):
            parm = {i + 1: [0.0, 0.0, 0.0, GAIN[i], BIAS[i], PHASE[i]]}
            parm_list.update(parm)

        # print(parm_list)
        
        self.kf = position_vector[0]
        self.num_CPG = len(parm_list)
        self.CPG_list = []
        self.w_ms_list = [None,WEIGHT[0], WEIGHT[1], WEIGHT[2], WEIGHT[3], WEIGHT[4], WEIGHT[5], WEIGHT[6],
                          WEIGHT[7], WEIGHT[8], WEIGHT[9], WEIGHT[10], WEIGHT[11], WEIGHT[12], WEIGHT[13]  ]
        self.master_list = [None, 0,0,1,3,3,1,6,6,2,9,9, 2,12,12]
        
        for i in range(self.num_CPG):
            if i == 0:
                self.CPG_list.append(CPG_Sinneutron(0, master_nuron=None, param=parm_list[0], kf=self.kf, w_ms=0))
            else:
                self.CPG_list.append(CPG_Sinneutron(i, master_nuron=self.CPG_list[self.master_list[i]],
                                                    param=parm_list[i], kf=self.kf, w_ms=self.w_ms_list[i]))
    
    def output(self, state):
        output_list = []
        for cpg_n in self.CPG_list:
            cpg_n.next_output(f1=0, f2=0)
            output_list.append(cpg_n.parm['o'])
        
        return output_list
