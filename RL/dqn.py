
import json
import numpy as np
import os
import random
from utils import *
from torch import nn
import math
from tkinter import _flatten
from collections import Counter

from agent.model import Net
import torch.optim as optim

class BinaryRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, embed,data_type, seed=1, max_turn=15, cand_num=10, cand_item_num=10, attr_num=20, mode='train', ask_num=1, entropy_way='weight entropy', fm_epoch=0):
        self.data_name = data_name
        self.data_type=data_type
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn    #MAX_TURN
        self.attr_state_num = attr_num
        self.kg = kg
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'concept').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'exercise').value_len
        
        self.mastery=0.9
        self.patience=4.0
        print('mastery and patience')
        print(self.mastery,self.patience)
        
        # action parameters
        self.ask_num = ask_num
        self.rec_num = 10
        self.random_sample_feature = False
        self.random_sample_item = False
        if cand_num == 0:
            self.cand_num = 10
            self.random_sample_feature = True
        else:
            self.cand_num = cand_num
        if cand_item_num == 0:
            self.cand_item_num = 10
            self.random_sample_item = True
        else:
            self.cand_item_num = cand_item_num
            
        self.pre_list={}
        self.exer_score={}
        #  entropy  or weight entropy
        self.ent_way = entropy_way
        
        # # user's profile
        self.user_acc_item = []  # user accepted item which asked by agent
        self.user_rej_item = []  # user rejected item which asked by agent
        self.user_master_item = [] # user mastered item which asked by agent
        self.cand_items = []   # candidate items
        self.all_exercises=[]
        self.item_feature_pair = {}
        self.reachable_concept=[]

        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        # self.cur_node_set = []     # maybe a node or a node set  /   normally save feature node
        # state veactor
        self.user_embed = None
        self.conver_his = []    #conversation_history
        self.attr_ent = []  # attribute entropy
        self.cand_item_score={}
        
        self.ask_open=True

        self.ui_dict = self.__load_rl_data__mooc(data_name, data_type,mode=mode, difficulty='easy')  # np.array [ u i weight]
        self.user_set=list(self.ui_dict.keys())
        self.user_weight_dict = dict()
        self.user_items_dict = dict()
        self.device = torch.device('cpu')
        
        #init seed & init user_dict
        set_random_seed(self.seed) # set random seed
        if mode == 'train':
            self.__user_dict_init__() # init self.user_weight_dict  and  self.user_items_dict
        elif mode == 'test':
            self.ui_array = None    # u-i array [ [userID1, itemID1], ...,[userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0
        # embeds = {
        #     'ui_emb': ui_emb,
        #     'feature_emb': feature_emb
        # }
        # load fm epoch
        embeds = load_embed(data_name, embed, data_type)
        if embeds:
            self.ui_embeds =embeds['ent_embeddings.weight'][:self.user_length+self.item_length].numpy()
            self.feature_emb = embeds['ent_embeddings.weight'][self.user_length+self.item_length:].numpy()
        else:
            self.ui_embeds = nn.Embedding(self.user_length+self.item_length, 64).weight.data.numpy()
            self.feature_emb = nn.Embedding(self.feature_length, 64).weight.data.numpy()
        # self.feature_length = self.feature_emb.shape[0]-1

        self.action_space = []

        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.2,
            'until_T': -0.3,      # MAX_Turn
            'cand_none': -0.1,
        }
        
        
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_suc': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   # This dict is used to calculate entropy
        print('reward_dict',self.reward_dict)
        print('parameter',self.cand_item_num,self.max_turn)
    
    def setmastery__patiency(self,mastery,patiency):
        self.demand=float(mastery)
        self.patience=float(patiency)
        print('reset!',self.demand,self.patience)

    def __load_rl_data__(self, data_name, mode):
        data_name='LAST_FM'
        if mode == 'train':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'), encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
        return mydict
    
    def __load_rl_data__mooc(self, data_name,data_type, mode, difficulty):
        if mode == 'train':
           with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data'+TYPE_DIR[data_type]+'/train_dict.json'), encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
                
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name],'UI_Interaction_data'+TYPE_DIR[data_type]+'/test_dict.json'), encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
        return mydict


    def __user_dict_init__(self):   
        print('user_dict init successfully!(ingnored)')

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str, items in self.ui_dict.items():
            user_id = int(user_str)
            ui_list.append([user_id, items])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)

    def reset(self, embed=None):
        self.net = Net(self.user_length, self.item_length, self.feature_length)
        self.load_snapshot(self.net, 'agent/model'+TYPE_DIR[self.data_type]+'/model_epoch' + str(11))
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.02)
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]
        #init  user_id  item_id  cur_step   cur_node_set
        self.cur_conver_step = 0   #reset cur_conversation step
        # self.cur_node_set = []
        if self.mode == 'train':
            # users = list(i for i in range(self.user_length))
            # tuple_=np.random.choice(len(self.ui_dict))
            # tuple_=self.ui_dict[str(tuple_)]
            # self.user_id = int(tuple_[0]) # select user  according to user weights
            # self.target_item = tuple_[1]
            self.user_id = int(np.random.choice(self.user_set)) # select user  according to user weights
            self.target_item = self.ui_dict[str(self.user_id)]
        elif self.mode == 'test':
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item = self.ui_array[self.test_num, 1]
            self.test_num += 1
        
        self.difficulty_record=0

        # init user's profile
        print('-----------reset state vector------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        self.user_acc_item = []  # user accepted item which asked by agent
        self.user_rej_item = []  # user rejected item which asked by agent
        self.user_master_item = []  # user mastered item which asked by agent
        self.all_exercises=[]
        self.cand_items=[]
        # self.action_record=[]
        self.demand=0.5
        self.pre_list={}
        self.exer_score=self.update_exer_score()
        # init state vector
        self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.attr_ent = [0] * self.attr_state_num  # attribute entropy
        self.reachable_concept=[]
        self.cand_item_score={}
        self.ask_open=True
        self.mastery_step=0
        
        # initialize dialog by randomly asked a question from ui interaction
        self._update_cand_concepts()
        
        # self.user_acc_item.extend(list(self.candidate_concept)) #update user acc_fea
        
        self._updata_reachable_feature()
        # self._update_cand_items(self.candidate_concept, acc_rej=True)
        
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1


        return self._get_state(),self._get_cand(), self._get_action_space()

    def _get_state(self):
        self_cand_items = self.all_exercises      
        user = [self.user_id]
        target_idx=self.target_item+self.user_length+self.item_length
        cur_node = [self.target_item + self.user_length + self.item_length,self.user_id] 
        cand_items = [x + self.user_length for x in self_cand_items]
        reachable_concept = [x + self.user_length + self.item_length for x in self.reachable_concept]
        reachable_concept.remove(target_idx)
        neighbors = cur_node + cand_items + reachable_concept
        idx = dict(enumerate(neighbors))
        idx = {v: k for k, v in idx.items()}
        i = []
        v = []
        adj = torch.FloatTensor(1)
        neighbors = torch.LongTensor(neighbors)
        state = {'cur_node': cur_node,
                 'neighbors': neighbors,
                 'adj': adj}
        return state
    
    def _get_cand(self):
        target_feature = self._map_to_all_id([self.target_item],'feature')
        self.cand_items_thisround=self.cand_items
        if self.random_sample_item:
            cand_item =  self._map_to_all_id(random.sample(self.cand_items_thisround, min(len(self.cand_items_thisround),self.cand_item_num)),'item')
        else:
             cand_item =  self._map_to_all_id(random.sample(self.cand_items_thisround, min(len(self.cand_items_thisround),self.cand_item_num)),'item')
        if self.ask_open:
            cand = target_feature+cand_item
        else:
            cand = cand_item
        return cand
    
    def _get_action_space(self):
        if self.ask_open:
            action_space = [self._map_to_all_id([self.target_item],'feature'),self._map_to_all_id(self.cand_items_thisround,'item')]
        else:
            action_space = [self._map_to_all_id(self.cand_items_thisround,'item')]
        return action_space

    
    
    def step(self, action, sorted_actions, embed=None):  
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        
        cand_items = self.cand_items
        a=self.test_master(self.net,self.user_id,self.all_exercises)
        
        if self.mastery_step==0 and a>self.mastery:
            self.mastery_step=self.cur_conver_step
            print(self.mastery_step,'master already!')
        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')        
            done = 1
        elif cand_items==[]:
            reward = self.reward_dict['cand_none']
            print('--> no cand items!')
            done = 1
        elif action >= self.user_length + self.item_length:   #ask feature
            asked_feature = self._map_to_old_id(action)
            print('test concept')
            reward, done = self._ask_update(asked_feature)  #update user's profile:  user_acc_feature & user_rej_feature
            # self._update_cand_items(asked_feature, acc_rej)   #update cand_items
            #========================================
            if reward > 0:
                print('-->Ask successfully!')
            else:
                print('-->Ask fail !')
        else:
            #===================== rec update=========
            action=self._map_to_old_id(action)
            reward, done = self._recommend_update(action)  
        
        # print('cand_item num: {}'.format(len(self.pre_list[self.current_prede])))
        self.cur_conver_step += 1
        return self._get_state(), self._get_cand(), self._get_action_space(), reward, done

   
    def _updata_reachable_feature(self):
        next_reachable_feature = []
        reachable_item_feature_pair={}
        cand_items=self.all_exercises
        for cand in cand_items:
                fea_belong_items = list(self.kg.G['exercise'][cand]['belong_to']) # A-I
                next_reachable_feature.extend(fea_belong_items)
                reachable_item_feature_pair[cand] = list(set(fea_belong_items))
        self.reachable_concept= list(set(next_reachable_feature) )
        self.item_feature_pair=reachable_item_feature_pair


    def _update_cand_concepts(self):
        concept_exercise=self.kg.G['concept'][self.target_item]['belong_to']
        self.cand_items=list(concept_exercise)
        self.all_exercises=concept_exercise
        self.candidate_concept=list(self.kg.G['concept'][self.target_item]['predecessor'])
        

    def _ask_update(self, target_item):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0
        # TODO datafram!     groundTruth == target_item features
        current_lev=self.test_master(self.net,self.user_id,self.all_exercises)
        if current_lev > self.mastery:
            # dif_day=self.cur_conver_step-self.mastery_step
            # reward = self.reward_dict['rec_suc']*(20-dif_day)
            reward = self.reward_dict['rec_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_suc'] #update state vector: conver_his
            done=1
              #update conver_his
        else:
            # if self.mastery_step:
            #     dif_day=self.cur_conver_step-self.mastery_step
            #     reward = self.reward_dict['ask_fail']*dif_day
            # else:
            #     reward = self.reward_dict['ask_fail']
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his
            if self.mode=='test':
                self.difficulty_record+=1
            self.ask_open=False
            
        if self.difficulty_record>=self.patience:
            done = 1
            reward = self.reward_dict['cand_none']
            print('out of patience')
            return reward, done
            
        return reward, done
    
    def _item_score(self):
        cand_item_score = {}
        for item_id in self.cand_items:
            item_embeds=self.ui_embeds[self.user_length:]
            item_embed = item_embeds[item_id]
            score = 0
            score += np.inner(np.array(self.user_embed), item_embed)
            prefer_embed = item_embeds[self.user_acc_item, :]  #np.array (x*64)
            unprefer_item = list(set(self.user_rej_item))
            unprefer_embed = item_embeds[unprefer_item, :]  #np.array (x*64)
            master_embed=item_embeds[self.user_master_item, :] 
            for i in range(len(self.user_acc_item)):
                score += np.inner(prefer_embed[i], item_embed)
            for i in range(len(unprefer_item)):
                score -= self.sigmoid([np.inner(unprefer_embed[i], item_embed)])[0]
            for i in range(len(self.user_master_item)):
                score -= self.sigmoid([np.inner(master_embed[i], item_embed)])[0]
                # score -= np.inner(unprefer_embed[i], item_embed)
            cand_item_score[item_id]=score
        self.cand_item_score=cand_item_score
    
    def select_cand_concepts(self):
        self._item_score()
        candidate_prede={}
        for prede_id,exer in self.pre_list.items():
            if len(exer)!=0:
                exer_score=[self.cand_item_score[i]for i in exer]
                candidate_prede[prede_id]=sum(self.sigmoid(exer_score))/len(exer)
            else:
                candidate_prede[prede_id]=0
        sorted_keys = sorted(self.pre_list.keys(), key=lambda x: candidate_prede[x], reverse=True)

        sorted_pre_list = {key: self.pre_list[key] for key in sorted_keys}
        
        return [item for values in sorted_pre_list.values() for item in values]

    
    def _recommend_update(self, recom_items):
        print('-->action: recommend items')
        feedback=self.update_net(recom_items)
        self.difficulty_record+=(1-feedback)
        done=0
        # a=self.test_master(self.net,self.user_id,self.pre_list[self.current_prede])
        if self.difficulty_record>=self.patience:
            done = 1
            reward = self.reward_dict['cand_none']
            print('out of patience')
            return reward, done
        elif feedback<=self.demand:
            print('recommend fail: too small')
            reward = self.reward_dict['ask_fail']
            self.user_rej_item.append(recom_items)
            # self.load_snapshot(self.net, 'agent/model/model_epoch' + str(20))
        elif feedback==1:
           
            self.user_master_item.append(recom_items)
            #if only has the last concept exercise, it should be acessed by asking not recommending
            reward = self.reward_dict['ask_fail']
            self.cand_items.remove(recom_items)
            print('recommend fail: proficiency ')
                 
        else:
            reward = self.reward_dict['ask_suc'] 
            self.cand_items.remove(recom_items)
            if self.user_rej_item:
                # self.restore_cand_items()
                self.user_rej_item=[]
            self.ask_open=True
            print('recommend successfully:')
            print(recom_items,self.cur_conver_step)
            
        return reward, done

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def _map_to_all_id(self, x_list, old_type):
        if old_type == 'item':
            return [x + self.user_length for x in x_list]
        elif old_type == 'feature':
            return [x + self.user_length + self.item_length for x in x_list]
        else:
            return x_list

    def _map_to_old_id(self, x):
        if x >= self.user_length + self.item_length:
            x -= (self.user_length + self.item_length)
        elif x >= self.user_length:
            x -= self.user_length
        return x


    def get_status(self,net):
        '''
        An example of getting student's knowledge status
        :return:
        '''
        net.eval()
        with open('data/MOOCCubeX/student_stat.txt', 'w', encoding='utf8') as output_file:
            for stu_id in range(self.user_length):
                # get knowledge status of student with stu_id (index)
                status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
                output_file.write(str(status) + '\n')
    def load_snapshot(self, model,filename):
        f = open(filename, 'rb')
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close()
    
    def test_master(self,net,user,exercises):
        output=[]
        device=self.device
        input_stu_ids, input_exer_ids, input_knowledge_embs = [], [], []
        input_exer_ids=list(exercises)
        input_stu_ids=[user]*len(exercises)
        for e in exercises:
                knowledge_emb = [0.] * self.feature_length
                for knowledge_code in self.kg.G['exercise'][e]['belong_to']:
                    knowledge_emb[knowledge_code] = 1.0
                input_knowledge_embs.append(knowledge_emb)
        input_stu_ids, input_exer_ids, input_knowledge_embs=torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.LongTensor(input_knowledge_embs)
        input_stu_ids, input_exer_ids, input_knowledge_embs = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        return sum(output)/len(output)
    
    def test_ex(self,net,user,exercises):
        output=[]
        device=self.device
        input_stu_ids, input_exer_ids, input_knowledge_embs = [], [], []
        input_exer_ids=list(exercises)
        input_stu_ids=[user]*len(exercises)
        for e in exercises:
                knowledge_emb = [0.] * self.feature_length
                for knowledge_code in self.kg.G['exercise'][e]['belong_to']:
                    knowledge_emb[knowledge_code] = 1.0
                input_knowledge_embs.append(knowledge_emb)
        input_stu_ids, input_exer_ids, input_knowledge_embs=torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.LongTensor(input_knowledge_embs)
        input_stu_ids, input_exer_ids, input_knowledge_embs = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        return output
    
    def update_exer_score(self):
        kg=self.kg
        exercise_score={}
        user_exercise_correct=kg.G['user'][self.user_id]['practice_correct']
        user_exercise_wrong=kg.G['user'][self.user_id]['practice_wrong']
        for i in user_exercise_correct:
            if self.target_item in kg.G['exercise'][i]['belong_to']:
                exercise_score[i]=1
        for i in user_exercise_wrong:
            if self.target_item in kg.G['exercise'][i]['belong_to']:
                exercise_score[i]=-1
        return exercise_score
    
    
    def update_net(self,action):
        print('testing model...')
        device=self.device
        print('training model...')

        loss_function = nn.NLLLoss()
        input_knowledge_embs = [0.] * self.feature_length
        for knowledge_code in self.kg.G['exercise'][action]['belong_to']:
                input_knowledge_embs[knowledge_code] = 1.0
        
        input_stu_ids, input_exer_ids, input_knowledge_embs, label= torch.LongTensor([self.user_id]), torch.LongTensor([action]), torch.Tensor(input_knowledge_embs),torch.LongTensor([1.0])
        input_stu_ids, input_exer_ids, input_knowledge_embs, label = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), label.to(device)
        
        # before_concept_level = self.net.get_knowledge_status(torch.LongTensor([self.user_id])).tolist()[0]
        # before_target=before_concept_level[self.target_item]
        out_put = self.net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        # epsilon = 1e-7
        # out_put=out_put-epsilon
        if out_put.item()<self.demand or out_put==1:
            return out_put.item()
        # grad_penalty = 0
        output_0 = torch.ones(out_put.size()).to(device) - out_put
        output1 = torch.cat((output_0, out_put), 1)
        loss = loss_function(torch.log(output1), label)
        loss.backward()
        self.optimizer.step()
        self.net.apply_clipper()
            
        # current_concept_level = self.net.get_knowledge_status(torch.LongTensor([self.user_id])).tolist()[0]
                
        # current_target=current_concept_level[self.target_item]
                    
        return out_put.item()
    
    def restore_cand_items(self):
        user_rej_item=list(set(self.user_rej_item))
        for i in user_rej_item:
            self.cand_items.append(i)
            
    

                
        