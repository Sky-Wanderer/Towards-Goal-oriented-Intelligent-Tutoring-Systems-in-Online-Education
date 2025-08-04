import torch
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader_mooc import ValTestDataLoader
from model import Net
import re


import torch.nn as nn
import torch.optim as optim

# can be changed according to config.txt
exer_n = 3432
knowledge_n = 642
student_n = 5904


def test(epoch):
    data_loader = ValTestDataLoader('test')
    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, 'model/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += out_put.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch, accuracy, rmse, auc))


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def get_status():
    '''
    An example of getting student's knowledge status
    :return:
    '''
    net = Net(student_n, exer_n, knowledge_n)
    load_snapshot(net, 'model/model_epoch13')       # load model
    net.eval()
    with open('result/student_stat.txt', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
            output_file.write(str(status) + '\n')


def get_exer_params():
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    net = Net(student_n, exer_n, knowledge_n)
    load_snapshot(net, 'model/model_epoch12')    # load model
    net.eval()
    exer_params_dict = {}
    for exer_id in range(exer_n):
        # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
        k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
        exer_params_dict[exer_id + 1] = (k_difficulty.tolist()[0], e_discrimination.tolist()[0])
    with open('result/exer_params.txt', 'w', encoding='utf8') as o_f:
        o_f.write(str(exer_params_dict))

def test_master(uid):
        data_loader = ValTestDataLoader('test')
        logs = data_loader.data[7]['logs']
        user_id = data_loader.data[7]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * data_loader.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)
if __name__ == '__main__':

    # global student_n, exer_n, knowledge_n
    
    exer_n = 3432
    knowledge_n = 642
    student_n = 5904

    # test(int(23))
    # get_status()
    # # get_exer_params()
    # a=[]
    # with open('result/student_stat.txt','r') as f:
    #     for line in f:
    #         # Remove brackets and split the line by commas
    #         line_values = line.strip()[1:-1].split(', ')

    #         # Convert string values to floats and add to the list
    #         line_values = [float(value) for value in line_values]
    #         a.append(line_values)
    # top_indices_per_row = []
    # num_top_elements = 60

    # for row in a:
    #     # Create a list of (index, value) tuples for each row
    #     indexed_values = list(enumerate(row))
        
    #     # Sort the list of tuples by value in descending order
    #     sorted_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)
        
    #     # Get the indices of the top elements
    #     top_indices = [index for index, _ in sorted_values[:num_top_elements]]
    #     top_indices_per_row.append(top_indices)

    # print(1)
    # for i in top_indices_per_row[16]:
    #     print(i,a[16][i])
    
    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    load_snapshot(net, 'model/model_epoch' + str(23))
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')

    loss_function = nn.NLLLoss()
    net.eval()

    # correct_count, exer_count = 0, 0
    # pred_all, label_all = [], []
    # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = test_master(16)
    # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
    #         device), input_knowledge_embs.to(device), labels.to(device)
    # out_put = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
    # output_0 = torch.ones(out_put.size()).to(device) - out_put
    # output1 = torch.cat((output_0, out_put), 1)

    # # grad_penalty = 0
    # loss = loss_function(torch.log(output1), labels)
    # loss.backward()
    # optimizer.step()
    # net.apply_clipper()
    
    # correct_count, exer_count = 0, 0
    # pred_all, label_all = [], []
    # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = test_master(16)
    # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
    #         device), input_knowledge_embs.to(device), labels.to(device)
    # out_put = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
    # output_0 = torch.ones(out_put.size()).to(device) - out_put
    # output1 = torch.cat((output_0, out_put), 1)

    # # grad_penalty = 0
    # loss = loss_function(torch.log(output1), labels)
    # loss.backward()
    # optimizer.step()
    # net.apply_clipper()
    
    # correct_count, exer_count = 0, 0
    # pred_all, label_all = [], []
    # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = test_master(16)
    # input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
    #         device), input_knowledge_embs.to(device), labels.to(device)
    # out_put = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
    # output_0 = torch.ones(out_put.size()).to(device) - out_put
    # output1 = torch.cat((output_0, out_put), 1)

    # # grad_penalty = 0
    # loss = loss_function(torch.log(output1), labels)
    # loss.backward()
    # optimizer.step()
    # net.apply_clipper()
    for i in range(15):
        pred_all, label_all = [], []
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = test_master(16)
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output_0 = torch.ones(out_put.size()).to(device) - out_put
        output1 = torch.cat((output_0, out_put), 1)

        # grad_penalty = 0
        loss = loss_function(torch.log(output1), labels)
        loss.backward()
        optimizer.step()
        net.apply_clipper()
        
    with open('result/student_stat.txt', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
            output_file.write(str(status) + '\n')

    a=[]
    with open('result/student_stat.txt','r') as f:
        for line in f:
            # Remove brackets and split the line by commas
            line_values = line.strip()[1:-1].split(', ')

            # Convert string values to floats and add to the list
            line_values = [float(value) for value in line_values]
            a.append(line_values)
    top_indices_per_row = []
    num_top_elements = 60

    for row in a:
        # Create a list of (index, value) tuples for each row
        indexed_values = list(enumerate(row))
        
        # Sort the list of tuples by value in descending order
        sorted_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)
        
        # Get the indices of the top elements
        top_indices = [index for index, _ in sorted_values[:num_top_elements]]
        top_indices_per_row.append(top_indices)

    print(1)
    for i in top_indices_per_row[16]:
        print(i,a[16][i])
    
    






