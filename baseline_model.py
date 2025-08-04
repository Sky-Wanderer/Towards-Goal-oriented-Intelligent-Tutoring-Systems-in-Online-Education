
import math
import random
import numpy as np
import os
import sys
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
# sys.path.append('..')
from agent.model import Net

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import utils

#TODO select env
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from RL_model import Agent, ReplayMemoryPER
from gcn import GraphEncoder
import time
import warnings

warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_STAR: BinaryRecommendEnv,
    MOOC:BinaryRecommendEnv
    }
FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_STAR: 'feature',
    YELP: 'large_feature',
    YELP_STAR: 'feature',
    MOOC:'concept'
}
class Agent(object):
    def __init__(self,data_name,embed,type,user_dict,kg):
        self.type=type
        self.device=torch.device('cpu')
        self.user_dict=user_dict
        embeds = load_embed(data_name,embed,epoch=0)
        self.kg=kg
        self.user_length=len(kg.G['user'])
        self.item_length=len(kg.G['exercise'])
        self.feature_length=len(kg.G['concept'])
        if embeds:
            self.ui_embeds =embeds['ent_embeddings.weight'][:self.user_length+self.item_length].numpy()
            self.feature_emb = embeds['ent_embeddings.weight'][self.user_length+self.item_length:].numpy()
        self.cand_items=[]
        self.net = Net(self.user_length, self.item_length, self.feature_length)
        self.load_snapshot(self.net, 'agent/model/model_epoch' + str(11))
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.1)
        
    def load_snapshot(self, model, filename):
        f = open(filename, 'rb')
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close() 
    
    def init_user(self,uid,target_item):
        self.user_id=int(uid)
        self.target_item=int(target_item)
        
    def select_action(self):
        if self.type=='greedy':
            action=self.cand_items.pop(0)
        elif self.type=='KNN':
            action=self.cand_items.pop(0)
        else:
            feature_items=tuple(item for i in target_item for item in self.kg.G['concept'][i]['belong_to'])
            cand_items = set(feature_items)  #  itersection
            cand_items = list(cand_items)
        reward=0
        return action
    def update_cand_items(self,user,target_item):
        if self.type=='greedy':
             cand_items = list(self.kg.G['concept'][target_item]['belong_to'])
        elif self.type=='KNN':
            user_emb=self.ui_embeds[int(user)]
            similarities = []
            cand_items=[]
            for exercise in self.kg.G['exercise'].keys():
                cand_items.append(exercise)
                exer_emb=self.ui_embeds[self.user_length+int(exercise)]
                similarity = cosine_similarity([user_emb], [exer_emb])[0][0]
                similarities.append(similarity)
            sorted_items = [value for _, value in sorted(zip(similarities, cand_items), reverse=True)]
            cand_items=sorted_items
        else:
            feature_items=tuple(item for i in target_item for item in self.kg.G['concept'][i]['belong_to'])
            cand_items = set(feature_items)  #  itersection
            cand_items = list(cand_items)
        self.cand_items =cand_items
        
        
    
    def update_agent(self,action):
        device=self.device
        loss_function = nn.NLLLoss()
        input_knowledge_embs = [0.] * self.feature_length
        for knowledge_code in self.kg.G['exercise'][action]['belong_to']:
                input_knowledge_embs[knowledge_code] = 1.0
        
        input_stu_ids, input_exer_ids, input_knowledge_embs, label= torch.LongTensor([self.user_id]), torch.LongTensor([action]), torch.Tensor(input_knowledge_embs),torch.LongTensor([1.0])
        input_stu_ids, input_exer_ids, input_knowledge_embs, label = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), label.to(device)
        
        before_concept_level = self.net.get_knowledge_status(torch.LongTensor([self.user_id])).tolist()[0]
        before_target=before_concept_level[self.target_item]
        out_put = self.net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        epsilon = 1e-7
        out_put=out_put-epsilon
        output_0 = torch.ones(out_put.size()).to(device) - out_put
        output1 = torch.cat((output_0, out_put), 1)
        if out_put.item()<0.7:
            return before_target,before_target,out_put.item()
        # grad_penalty = 0
        loss = loss_function(torch.log(output1), label)
        loss.backward()
        self.optimizer.step()
        self.net.apply_clipper()
            
        current_concept_level = self.net.get_knowledge_status(torch.LongTensor([self.user_id])).tolist()[0]
                
        current_target=current_concept_level[self.target_item]
                    
        return before_target,current_target,out_put.item()   
        
def __test_tuple_generate__(ui_dict):
        ui_list = []
        for user_str, items in ui_dict.items():
            user_id = int(user_str)
            ui_list.append([user_id, items])
        ui_array = np.array(ui_list)
        np.random.shuffle(ui_array)
        return ui_array
    
def __load_rl_data__mooc(data_name):
    with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/test_dict.json'), encoding='utf-8') as f:
            print('test_data: load RL test data')
            mydict = json.load(f)
    return mydict
        
def evaluate(args, kg, dataset, filename):
    SR5, SR10, SR15, AvgT, Rank = 0, 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    tt = time.time()
    start = tt
    kg_file='./tmp/mooc/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    ui_dict=__load_rl_data__mooc('MOOC')
    ui_array=__test_tuple_generate__(ui_dict)
    set_random_seed(args.seed)
    SR10, SR20, SR50, AvgT, Difficulty = 0, 0, 0, 0, 0
    turn_result = []
    result = []
    user_size = ui_array.shape[0]
    agent=Agent('MOOC','transe','KNN',ui_dict,kg)
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(args.load_rl_epoch) + filename

    for user_id,targetconcept in ui_dict.items():  #user_size
        print(user_id,targetconcept)
        # TODO uncommend this line to print the dialog process 
        agent.update_cand_items(user_id,targetconcept)
        improvement=0
        agent.init_user(user_id,targetconcept)
        for t in count():  # user  dialog
            done=0
            action=agent.select_action()
            before_target,current_target,out_put=agent.update_agent(action)
            if current_target>0.75:
                done=1
            if t==40:
                done=1
            if done:
                enablePrint()
                if current_target > 0.75:  # recommend successfully
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                    if t < 10:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 20:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                    Rank += (1/math.log(t+3,2) + (1/math.log(t+2,2)-1/math.log(t+3,2))/math.log(done+1,2))
                else:
                    Rank += 0
                AvgT += t+1
                break
        
        SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT / args.observe_num, Rank / args.observe_num]
        SR_TURN = [i/args.observe_num for i in SR_turn_15]
            
        result.append(SR)
        turn_result.append(SR_TURN)
        SR5, SR10, SR15, AvgT, Rank = 0, 0, 0, 0, 0
        SR_turn_15 = [0] * args.max_turn
        tt = time.time()
        enablePrint()

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    Rank_mean = np.mean(np.array([item[4] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=10, SR=SR_all, spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=10, SR=SR_all, spend_time=time.time() - start,
                  mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}'.format(SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean))
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        #f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=MOOC, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    parser.add_argument('--entropy_method', type=str, default='weight_entropy', help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--cand_len_size', type=int, default=20, help='binary state size for the length of candidate items')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--observe_num', type=int, default=500, help='the number of epochs to save RL model and metric')
    parser.add_argument('--load_rl_epoch', type=int, default=19, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--max_steps', type=int, default=100, help='max training steps')
    parser.add_argument('--eval_num', type=int, default=10, help='the number of epochs to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
    parser.add_argument('--fix_emb', type=bool, default=True, help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='transe', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name)
    filename = 'train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    evaluate(args, kg, dataset, filename)

if __name__ == '__main__':
    main()