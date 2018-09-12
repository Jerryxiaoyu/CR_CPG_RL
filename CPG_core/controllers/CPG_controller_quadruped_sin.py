from CPG_core.CPG_Sin_osillator import sin_oscillator


from CPG_core.CPG_Sin_osillator import CPG_Sinneutron
class CPG_network(object):
    def __init__(self, position_vector):
        kf = position_vector[0]
        GAIN0 = position_vector[1]
        GAIN1 = position_vector[2]
        GAIN2 = position_vector[3]
        GAIN3 = position_vector[4]
        GAIN4 = position_vector[5]
        GAIN5 = position_vector[6]
        GAIN6 = position_vector[7]
        GAIN7 = position_vector[8]
        GAIN8 = position_vector[9]
        GAIN9 = position_vector[10]
        GAIN10 = position_vector[11]
        GAIN11 = position_vector[12]
        GAIN12 = position_vector[13]
    
        BIAS0 = position_vector[14]
        BIAS1 = position_vector[15]
        BIAS2 = position_vector[16]
        BIAS3 = position_vector[17]
        BIAS4 = position_vector[18]
        BIAS5 = position_vector[19]
        BIAS6 = position_vector[20]
        BIAS7 = position_vector[21]
        BIAS8 = position_vector[22]
        BIAS9 = position_vector[23]
        BIAS10 = position_vector[24]
        BIAS11 = position_vector[25]
        BIAS12 = position_vector[26]

        PHASE0 = position_vector[27]
        PHASE1 = position_vector[28]
        PHASE2 = position_vector[29]
        PHASE3 = position_vector[30]
        PHASE4 = position_vector[31]
        PHASE5 = position_vector[32]
        PHASE6 = position_vector[33]
        PHASE7 = position_vector[34]
        PHASE8 = position_vector[35]
        PHASE9 = position_vector[36]
        PHASE10 = position_vector[37]
        PHASE11 = position_vector[38]
        PHASE12 = position_vector[39]

        self.parm_list = {
            0:  [0.0, 0.0, 0.0,    1.0, 0.0, 0],
            1: [0.0, 0.0, 0.0,    GAIN0, BIAS0, PHASE0],
            2: [0.0, 0.0, 0.0,    GAIN1, BIAS1, PHASE1],
            3: [0.0, 0.0, 0.0,    GAIN2, BIAS2, PHASE2],
            4: [0.0, 0.0, 0.0,     GAIN3, BIAS3, PHASE3],
            5: [0.0, 0.0, 0.0,     GAIN4, BIAS4, PHASE4],
            6: [0.0, 0.0, 0.0,  GAIN5, BIAS5, PHASE5],
            7: [0.0, 0.0, 0.0,    GAIN6, BIAS6, PHASE6],
            8: [0.0, 0.0, 0.0,   GAIN7, BIAS7, PHASE7],
            9: [0.0, 0.0, 0.0,  GAIN8, BIAS8, PHASE8],
            10: [0.0, 0.0, 0.0,   GAIN9, BIAS9, PHASE9],
            11: [0.0, 0.0, 0.0,  GAIN10, BIAS10, PHASE10],
            12: [0.0, 0.0, 0.0,    GAIN11, BIAS11, PHASE11],
            13: [0.0, 0.0, 0.0,   GAIN12, BIAS12, PHASE12],
        }
        
        self.kf = position_vector[0]
        self.num_CPG = len(self.parm_list)
        self.CPG_list =[]
        
        self.w_ms_list = [None, 1, 1, -1, - 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  #yuanlai hao de
        #self.w_ms_list =[None, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1]
        
        self.master_list = [None,  0,1,1,1,1,2,2,3,3,4,4,5,5]
        self.kesi = 5.0
        
        for i in range(self.num_CPG):
            if i == 0:
                self.CPG_list.append(CPG_Sinneutron(0, master_nuron = None, param=self.parm_list[0] ,kf= self.kf, w_ms = 0))
            else:
                self.CPG_list.append(CPG_Sinneutron(i, master_nuron=self.CPG_list[self.master_list[i]],
                                                 param=self.parm_list[i], kf=self.kf, w_ms= self.w_ms_list[i]))
                
        #print('CPG init')
    
    def output(self, state):
        output_list = []
        for cpg_n in self.CPG_list:
            cpg_n.next_output(f1=0, f2=0)
            output_list.append(cpg_n.parm['o'])
        
        return output_list

    def update(self, fi_l):
    
        if len(fi_l) ==2:
            
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
            
            

            self.CPG_list[2].parm['R1'] = self.parm_list[2][3] * gain_left
            self.CPG_list[6].parm['R1'] = self.parm_list[6][3] * gain_left
            self.CPG_list[7].parm['R1'] = self.parm_list[7][3] * gain_left

            self.CPG_list[3].parm['R1'] = self.parm_list[3][3] * gain_left
            self.CPG_list[8].parm['R1'] = self.parm_list[8][3] * gain_left
            self.CPG_list[9].parm['R1'] = self.parm_list[9][3] * gain_left

            self.CPG_list[4].parm['R1'] = self.parm_list[4][3] * gain_right
            self.CPG_list[10].parm['R1'] = self.parm_list[10][3] * gain_right
            self.CPG_list[11].parm['R1'] = self.parm_list[11][3] * gain_right

            self.CPG_list[5].parm['R1'] = self.parm_list[5][3] * gain_right
            self.CPG_list[12].parm['R1'] = self.parm_list[12][3] * gain_right
            self.CPG_list[13].parm['R1'] = self.parm_list[13][3] * gain_right
            
        elif len(fi_l) ==4:

            f_left = 1 - (0.5 - fi_l[0]) * 2
            f_right = 1 - (0.5 - fi_l[1] ) * 2
            gain_left= 1 - (0.5 - fi_l[2]) * self.kesi
            gain_right = 1 - (0.5 -fi_l[3] ) * self.kesi

            self.CPG_list[2].parm['f12'] = self.parm_list[2][5] * f_left
            self.CPG_list[6].parm['f12'] = self.parm_list[6][5] * f_left
            self.CPG_list[7].parm['f12'] = self.parm_list[7][5] * f_left

            self.CPG_list[3].parm['f12'] = self.parm_list[3][5] * f_left
            self.CPG_list[8].parm['f12'] = self.parm_list[8][5] * f_left
            self.CPG_list[9].parm['f12'] = self.parm_list[9][5] * f_left

            self.CPG_list[4].parm['f12'] = self.parm_list[4][5] * f_right
            self.CPG_list[10].parm['f12'] = self.parm_list[10][5] * f_right
            self.CPG_list[11].parm['f12'] = self.parm_list[11][5] * f_right

            self.CPG_list[5].parm['f12'] = self.parm_list[5][5] * f_right
            self.CPG_list[12].parm['f12'] = self.parm_list[12][5] * f_right
            self.CPG_list[13].parm['f12'] = self.parm_list[13][5] * f_right
#           #gain
            self.CPG_list[2].parm['R1'] = self.parm_list[2][3] * gain_left
            self.CPG_list[6].parm['R1'] = self.parm_list[6][3] * gain_left
            self.CPG_list[7].parm['R1'] = self.parm_list[7][3] * gain_left

            self.CPG_list[3].parm['R1'] = self.parm_list[3][3] * gain_left
            self.CPG_list[8].parm['R1'] = self.parm_list[8][3] * gain_left
            self.CPG_list[9].parm['R1'] = self.parm_list[9][3] * gain_left

            self.CPG_list[4].parm['R1'] = self.parm_list[4][3] * gain_right
            self.CPG_list[10].parm['R1'] = self.parm_list[10][3] * gain_right
            self.CPG_list[11].parm['R1'] = self.parm_list[11][3] * gain_right

            self.CPG_list[5].parm['R1'] = self.parm_list[5][3] * gain_right
            self.CPG_list[12].parm['R1'] = self.parm_list[12][3] * gain_right
            self.CPG_list[13].parm['R1'] = self.parm_list[13][3] * gain_right
            
    
        # if len(fi_l) == 4:
        #     print('add gain parm')
        #     fl_gain_left = 1 - (1.0 - self.kesi) * fi_l[2]
        #     fl_gain_right = 1 - (1.0 - self.kesi) * fi_l[3]
        #
        #     self.CPG_list[2].parm['R1'] = self.parm_list[2][3] * fl_gain_left
        #     self.CPG_list[6].parm['R1'] = self.parm_list[6][3] * fl_gain_left
        #     self.CPG_list[7].parm['R1'] = self.parm_list[7][3] * fl_gain_left
        #
        #     self.CPG_list[3].parm['R1'] = self.parm_list[3][3] * fl_gain_right
        #     self.CPG_list[8].parm['R1'] = self.parm_list[8][3] * fl_gain_right
        #     self.CPG_list[9].parm['R1'] = self.parm_list[9][3] * fl_gain_right
        #
        #     self.CPG_list[4].parm['R1'] = self.parm_list[4][3] * fl_gain_right
        #     self.CPG_list[10].parm['R1'] = self.parm_list[10][3] * fl_gain_right
        #     self.CPG_list[11].parm['R1'] = self.parm_list[11][3] * fl_gain_right
        #
        #     self.CPG_list[5].parm['R1'] = self.parm_list[5][3] * fl_gain_left
        #     self.CPG_list[12].parm['R1'] = self.parm_list[12][3] * fl_gain_left
        #     self.CPG_list[13].parm['R1'] = self.parm_list[13][3] * fl_gain_left


class CPG_network5(object):
    def __init__(self, CPG_node_num,position_vector):
        
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
        
        self.kf = position_vector[0]
        self.num_CPG = len(parm_list)
        self.CPG_list = []
        self.w_ms_list = [None, WEIGHT[0], WEIGHT[1], WEIGHT[2], WEIGHT[3], WEIGHT[4], WEIGHT[5], WEIGHT[6], WEIGHT[7], WEIGHT[8],WEIGHT[9], WEIGHT[10], WEIGHT[11], WEIGHT[12],]
        self.master_list = [None, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        
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
        
       