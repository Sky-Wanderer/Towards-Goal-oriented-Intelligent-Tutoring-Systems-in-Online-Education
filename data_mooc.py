
import argparse
from utils import *
import json
import torch
from agent.model import Net

def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()
    
def get_exer_params(net):
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    net.eval()
    exer_params_dict = {}
    with open('data/MOOCCubeX/exer_params.txt', 'w', encoding='utf8') as o_f:
        for exer_id in range(exer_n):
            # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
            k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
            o_f.write(str(k_difficulty.tolist()[0])+'\n')
            o_f.write(str(e_discrimination.tolist()[0])+'\n')

def get_status(net):
    '''
    An example of getting student's knowledge status
    :return:
    '''
    net.eval()
    with open('data/MOOCCubeX/student_stat.txt', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
            output_file.write(str(status) + '\n')


exer_n = 3432
knowledge_n = 642
student_n = 5904

kg_file = '/Users/ff/Desktop/aaai/Neural_Cognitive_Diagnosis-NeuralCD/mooc_data/kg.pkl'
kg = pickle.load(open(kg_file, 'rb'))
user_info=kg.G['user']
exer_info=kg.G['exercise']
concept=kg.G['concept']

net = Net(student_n, exer_n, knowledge_n)
device = torch.device('cpu')
print('testing model...')
load_snapshot(net, 'agent/model/model_epoch' + str(18))
net = net.to(device)

get_status(net)
get_exer_params(net)



 