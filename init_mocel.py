import torch
from agent.model import Net
def load_snapshot(model, filename):
        f = open(filename, 'rb')
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close()
def save_snapshot(model, filename):
        f = open(filename, 'wb')
        torch.save(model.state_dict(), f)
        f.close()
exer_n = 3432
knowledge_n = 642
student_n = 5904
net = Net(student_n, exer_n, knowledge_n)
load_snapshot(net, 'agent/model/model_epoch20_back')
save_snapshot(net,'agent/model/model_epoch' + str(20))  