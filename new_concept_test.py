import torch
from agent.model import Net
import pickle
import torch.nn as nn
import torch.optim as optim

def load_snapshot(model, filename):
        f = open(filename, 'rb')
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close()
def save_snapshot(model, filename):
        f = open(filename, 'wb')
        torch.save(model.state_dict(), f)
        f.close()
def load_kg(dataset):
    kg_file = dataset
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def test_master(uid,exercises):
        user_id = uid
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in exercises:
            input_stu_ids.append(user_id)
            input_exer_ids.append(log)
            knowledge_emb = [0.] *knowledge_n
            for knowledge_code in kg.G['exercise'][log]['belong_to']:
                knowledge_emb[knowledge_code] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            ys.append(1.0)
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

def get_exer_params(exer_n):
    '''
    An example of getting exercise's parameters (knowledge difficulty and exercise discrimination)
    :return:
    '''
    exer_params_dict = {}
    b={}
    for exer_id in exer_n:
        # get knowledge difficulty and exercise discrimination of exercise with exer_id (index)
        k_difficulty, e_discrimination = net.get_exer_params(torch.LongTensor([exer_id]))
        exer_params_dict[exer_id] = (k_difficulty.tolist()[0][148])
        b[exer_id]=(e_discrimination.tolist()[0])
    return exer_params_dict,b


def compute_proficiency(output):
        sum=0
        for i in output:
                sum+=i[0]
        return sum/len(output)

def generate_ui_data(uid):
        mydict={}
        a=[]
        file_path ='data/MOOCCubeX/student_stat.txt'
        with open(file_path, "r") as file:
                lines = file.readlines()
                if len(lines) >= uid+1:
                        data = lines[uid]
        num_top_elements = 30
        indexed_values = list(enumerate(data))
            
            # Sort the list of tuples by value in descending order
        sorted_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)

            # Get the indices of the top elements
        top_indices = [index for index, _ in sorted_values[:num_top_elements]]
        return top_indices
exer_n = 3432
knowledge_n = 642
student_n = 5904
net = Net(student_n, exer_n, knowledge_n)


load_snapshot(net, 'agent/model/model_epoch19')

kg = load_kg('./tmp/mooc/kg.pkl')
device = torch.device('cpu')
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.NLLLoss()
user=2212
target_concept=637
concept_exercise=kg.G['concept'][target_concept]['belong_to']
exercise2=kg.G['user'][user]['practice_correct']+kg.G['user'][user]['practice_wrong']
exercise=list(set(concept_exercise)&set(exercise2))
input_stu_ids, input_exer_ids, input_knowledge_embs, labels = test_master(user,concept_exercise)
input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)

# out_put = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs).tolist()

# print(compute_proficiency(out_put))
# print(1)
a=[]
m=exercise
l,b=get_exer_params(m)
for i in m:
        
        input_knowledge_embs = [0.] * knowledge_n
        for knowledge_code in kg.G['exercise'][i]['belong_to']:
                input_knowledge_embs[knowledge_code] = 1.0
                
        input_stu_ids, input_exer_ids, input_knowledge_embs, label= torch.LongTensor([user]), torch.LongTensor([i]), torch.Tensor(input_knowledge_embs),torch.LongTensor([1.0])
        
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
        before_concept_level = net.get_knowledge_status(torch.LongTensor([user])).tolist()[0]
        before_target=before_concept_level[target_concept]
        
        out_put = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        epsilon = 1e-7
        out_put=out_put-epsilon
        output_0 = torch.ones(out_put.size()).to(device) - out_put
        output1 = torch.cat((output_0, out_put), 1)
        

        # grad_penalty = 0
        loss = loss_function(torch.log(output1), label)
        loss.backward()
        optimizer.step()
        net.apply_clipper()
            
        current_concept_level = net.get_knowledge_status(torch.LongTensor([user])).tolist()[0]
                
        current_target=current_concept_level[target_concept]
        
        different_indices = []

        for index, (item1, item2) in enumerate(zip(before_concept_level, current_concept_level)):
                if item1 != item2:
                        different_indices.append(index)

        exer_concept=kg.G['exercise'][i]['belong_to']
        a.append(current_target-before_target)
        
h={}
for i in m:
        h[i]=concept_exercise=len(kg.G['exercise'][i]['practice_wrong'])/(len(kg.G['exercise'][i]['practice_wrong'])+len(kg.G['exercise'][i]['practice_correct']))
        
print(1)

0.5084508657455444

0.5084508657455444
        
        


