
import argparse
from utils import *
import json
import torch

kg_file = '/Users/ff/Desktop/aaai/Neural_Cognitive_Diagnosis-NeuralCD/mooc_data/kg.pkl'
kg = pickle.load(open(kg_file, 'rb'))
user_info=kg.G['user']
exer_info=kg.G['exercise']
log_data=[]
concept=kg.G['concept']
 
# for u_id,u_info in user_info.items():
#     user_data={}
#     exercise_data=[]
#     user_data['user_id']=u_id+1
#     length=len(u_info['practice_correct'])+len(u_info['practice_wrong'])
#     if length<15:
#         continue
#     else:
#         user_data['log_num']=length
#     for exer_id in u_info['practice_correct']:
#         exe={}
#         exe['exer_id']=exer_id+1
#         exe['score']=1.0
#         exe['knowledge_code']=[i+1 for i in exer_info[exer_id]['belong_to']]
#         exercise_data.append(exe)
#     for exer_id in u_info['practice_wrong']:
#         exe={}
#         exe['exer_id']=exer_id+1
#         exe['score']=0.0
#         exe['knowledge_code']=[i+1 for i in exer_info[exer_id]['belong_to']]
#         exercise_data.append(exe)
#     user_data['logs']=exercise_data
#     log_data.append(user_data)
    
# with open(os.path.join('log.json'), 'w', encoding='utf-8') as f:
#             json.dump(log_data, f, indent=2, ensure_ascii=False)
logs=[]
for exer_id in concept[11]['belong_to']:
    new_ex={}
    new_ex['exer_id']=exer_id+1
    new_ex['score']=1.0
    new_ex['knowledge_code']=[i+1 for i in exer_info[exer_id]['belong_to']]
    logs.append(new_ex)

with open(os.path.join('11.json'), 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

print(1)



# 创建与训练模型结构相同的模型
checkpoint = torch.load('tmp/mooc/FM-model-embeds/transe.ckpt')

print(1)