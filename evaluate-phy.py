
import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from sum_tree import SumTree

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


def pai_evaluate(args, kg, dataset, filename,epoch,difficulty):
    test_env = EnvDict[args.data_name](kg, dataset, args.data_name, args.embed,args.data_type, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num, mode='test', ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    memory = ReplayMemoryPER(args.memory_size) #10000
    embed = torch.FloatTensor(np.concatenate((test_env.ui_embeds, test_env.feature_emb, np.zeros((1,test_env.ui_embeds.shape[1]))), axis=0))
    gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg, embeddings=embed, \
        fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden).to(args.device)
    

    agent = Agent(device=args.device, memory=memory, state_size=args.hidden, action_size=embed.size(1), \
        hidden_size=args.hidden, gcn_net=gcn_net, learning_rate=args.learning_rate, l2_norm=args.l2_norm, PADDING_ID=embed.size(0)-1)
    print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
    agent.load_model(data_name=args.data_name, filename=filename, epoch_user=epoch,data_type=args.data_type,mastery=args.mastery,difficulty='0.5')

    tt = time.time()
    start = tt

    SR5,SR10, SR15, SR20, AvgT, patience,delay_step = 0,0, 0, 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(args.load_rl_epoch) + filename

    for user_num in tqdm(range(user_size)):  #user_size
        # TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================test tuple:{}===================='.format(user_num))

        state, cand, action_space = test_env.reset()  # Reset environment and record the starting state
        is_last_turn = False
        test_env.setmastery__patiency(difficulty,4)
        
        
        for t in count():  # user  dialog
            
            action, sorted_actions = agent.select_action(state, cand, action_space, is_test=True, is_last_turn=is_last_turn)
            next_state, next_cand, action_space, reward, done = test_env.step(action.item(), sorted_actions)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if t == 19:
                done=1
            if done:
                next_state = None
            state = next_state
            cand = next_cand
            if done:
                enablePrint()
                print('look',test_env.difficulty_record,test_env.cur_conver_step)
                if reward.item() >0.01:  # recommend successfully
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                    if t < 4:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                        SR20 += 1
                    elif t < 9:
                        SR10 += 1
                        SR15 += 1
                        SR20 += 1
                    elif t < 14:
                        SR15 += 1
                        SR20 += 1
                    else:
                        SR20 += 1
                    delay_step+=test_env.cur_conver_step-test_env.mastery_step
                    patience+= test_env.difficulty_record
                    AvgT += t+1
                else:
                    patience+= test_env.difficulty_record
                    AvgT += 20
                break
        
        if (user_num+1) % args.observe_num == 0 and user_num > 0:
            SR = [SR5/args.observe_num,SR10/args.observe_num, SR15/args.observe_num, SR20/args.observe_num, AvgT / args.observe_num, patience / args.observe_num,delay_step/ SR20 if SR20 else 0]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                    float(user_num) * 100 / user_size))
            print('SR5:{},SR10:{}, SR15:{}, SR20:{}, AvgT:{}, patience:{} '
                'Total epoch_uesr:{}'.format(SR5 / args.observe_num,SR10 / args.observe_num, SR15 / args.observe_num, SR20 / args.observe_num,
                                                AvgT / args.observe_num, patience / args.observe_num, user_num + 1))
           
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5,SR10, SR15, SR20, AvgT, patience,delay_step = 0,0, 0, 0, 0, 0,0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
        enablePrint()

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    SR20_mean = np.mean(np.array([item[3] for item in result]))
    AvgT_mean = np.mean(np.array([item[4] for item in result]))
    patience_mean = np.mean(np.array([item[5] for item in result]))
    DELAY_MEAN = np.mean(np.array([item[6] for item in result]))
    SR_all = [SR5_mean,SR10_mean, SR15_mean, SR20_mean, AvgT_mean, patience_mean,DELAY_MEAN]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                mode='test')
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    print('SR5:{},SR10:{}, SR15:{}, SR20:{}, AvgT:{}, patience:{}, delay:{}'.format(SR5_mean,SR10_mean, SR15_mean, SR20_mean, AvgT_mean, patience_mean,DELAY_MEAN))
    print(difficulty,'the demand is!')
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' +args.data_type+'/'+args.mastery+args.difficulty+ test_filename + '.txt'
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
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
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
    parser.add_argument('--observe_num', type=int, default=5, help='the number of epochs to save RL model and metric')
    parser.add_argument('--load_rl_epoch', type=int, default=10, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--max_steps', type=int, default=100, help='max training steps')
    parser.add_argument('--eval_num', type=int, default=10, help='the number of epochs to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=30, help='candidate item sampling number')
    parser.add_argument('--fix_emb', type=bool, default=True, help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='transe', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='mean', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')
    
    parser.add_argument('--data_type', type=str, default='MATH', help='the type of data')
    parser.add_argument('--mastery', type=str, default='math-best', help='the type of data')
    parser.add_argument('--difficulty', type=str, default='0.5', help='the type of data')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name,args.data_type)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name,args.data_type)
    filename = 'train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    for epoch,difficulty in [(2,'0.5')]:
        pai_evaluate(args, kg, dataset, filename,epoch,difficulty)

if __name__ == '__main__':
    main()