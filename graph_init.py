
import argparse

from Graph_generate.lastfm_data_process import LastFmDataset
from Graph_generate.lastfm_graph import LastFmGraph
from Graph_generate.lastfm_star_data_process import LastFmStarDataset
from Graph_generate.mooc_data_process import MoocDataset
from Graph_generate.mooc_graph import MoocGraph
from Graph_generate.yelp_data_process import YelpDataset
from Graph_generate.yelp_graph import YelpGraph
from utils import *
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=MOOC, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR, MOOC],
                     help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    # parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR, MOOC],
    #                      help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    args = parser.parse_args()
    DatasetDict = {
        LAST_FM: LastFmDataset,
        LAST_FM_STAR: LastFmStarDataset,
        YELP: YelpDataset,
        YELP_STAR: YelpDataset,
        
        MOOC:MoocDataset
    }
    GraphDict = {
        LAST_FM: LastFmGraph,
        LAST_FM_STAR: LastFmGraph,
        YELP: YelpGraph,
        YELP_STAR: YelpGraph,
        
        MOOC:MoocGraph
    }

    # Create 'data_name' instance for data_name.
    print('Load', args.data_name, 'from file...')
    print(TMP_DIR[args.data_name])
    if not os.path.isdir(TMP_DIR[args.data_name]):
        os.makedirs(TMP_DIR[args.data_name])
    dataset = DatasetDict[args.data_name](DATA_DIR[args.data_name])
    save_dataset(args.data_name, dataset)
    print('Save', args.data_name, 'dataset successfully!')

    # Generate graph instance for 'data_name'
    print('Create', args.data_name, 'graph from data_name...')
    dataset = load_dataset(args.data_name)
    kg = GraphDict[args.data_name](dataset)
    save_kg(args.data_name, kg)
    graph=construct_mooc(kg)
    print('Save', args.data_name, 'graph successfully!')
    # 创建包含前任节点的字典
    predecessors_within_n_hop_dict = {}
    n_hop = 2  # 修改为您想要的跳数
    for concept_id,concept_data in kg.G['concept'].items():
            predecessors = find_predecessors_within_n_hop(concept_id, n_hop, kg.G['concept'])
            predecessors_within_n_hop_dict[concept_id] = list(predecessors)
    generate_ui_data(kg)
    print(predecessors_within_n_hop_dict)
    

def construct(kg):
    users = kg.G['user'].keys()
    items = kg.G['item'].keys()
    features = kg.G['feature'].keys()
    num_node = len(users) + len(items) + len(features)
    graph = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            if i < len(users) and j < len(users)+len(items):
                graph[i][j] = 1
                graph[j][i] = 1
            elif i >= len(users) and i < len(users)+len(items):
                if j-len(users)-len(items) in kg.G['item'][i-len(users)]['belong_to']:
                    graph[i][j] = 1
                    graph[j][i] = 1
            else:
                pass
    ##print(graph)
    return graph

def construct_mooc(kg):
    PRACTICE = 'practice'
    SUCCESSOR = 'successor'
    relation_name = [PRACTICE, SUCCESSOR]
    users = kg.G['user'].keys()
    exercise = kg.G['exercise'].keys()
    concept = kg.G['concept'].keys()
    user_length=len(users)
    exercise_length=len(exercise)
    concept_start= user_length + exercise_length
    num_node = user_length + exercise_length + len(concept)
    graph = np.zeros((num_node, num_node))
    for relation in relation_name:
        if relation in ['practice']:
            for user, exercises in kg.G['user'].items():
                for exercise_id in exercises['practice_correct']:
                    graph[user][exercise_id+user_length] = 1
                    graph[exercise_id+user_length][user] = 1
                for exercise_id in exercises['practice_wrong']:
                    graph[user][exercise_id+user_length] = -1
                    graph[exercise_id+user_length][user] = -1
        if relation in ['successor']:
            for concept,exercises in kg.G['concept'].items():
                for exercise_id in exercises['belong_to']:
                    graph[concept+concept_start][exercise_id+user_length] = 1
                    graph[exercise_id+user_length][concept+concept_start] = 1
                for concept_id in exercises['successor']:
                    graph[concept+concept_start][concept_id+concept_start] = 1
                    graph[concept_id+concept_start][concept+concept_start] = 1
    ##print(graph)
    return graph


 # 创建前任节点的函数
def find_predecessors_within_n_hop(concept_id, n, data):
    predecessors = set()
    for _ in range(n):
        new_predecessors = set()
        for id,concept_data in data.items():
            if concept_id in concept_data['successor']:
                new_predecessors.add(id)
        predecessors.update(new_predecessors)
        concept_id = list(new_predecessors)  # Move to the next hop
    return predecessors

def find_smallest_greater_than_0_6(numbers):
    smallest = None
    
    for num in numbers:
        if num > 0.6:
            if smallest is None or num < smallest:
                smallest = num
    
    return smallest
def sort_concepts_by_score(user_data):
            concept_ids, scores = user_data
            sorted_concepts = [concept_id for _, concept_id in sorted(zip(scores, concept_ids), reverse=True)]
            return sorted_concepts

def generate_ui_data(kg):
        mydict={}
        a=[]
        #find the concept belong to the user:
        with open('data/MOOCCubeX/student_stat.txt','r') as f:
            for line in f:
                # Remove brackets and split the line by commas
                line_values = line.strip()[1:-1].split(', ')
                # Convert string values to floats and add to the list
                line_values = [float(value) for value in line_values]
                a.append(line_values)
        user_concept={}
        for i in range(5904):
            concept_lev=[]
            exercise=kg.G['user'][i]['practice_correct']+kg.G['user'][i]['practice_wrong']
            concept=set()
            for e in exercise:
                concept.update(kg.G['exercise'][e]['belong_to'])
            for c in concept:
                concept_lev.append(a[i][c])
            user_concept[i]=[concept, concept_lev]            
        print(user_concept[2212])
        # 对每个用户的 concept 进行重新排序,后续可根据需要选择用户掌握度为中为高的concept
        sorted_user_data_list = []
        for user_data in user_concept.values():
            sorted_concepts = sort_concepts_by_score(user_data)
            sorted_user_data_list.append(sorted_concepts)

        for index in range(5904):
             mydict[index]=sorted_user_data_list[index][-1]
        with open('data/MOOCCubeX/UI_Interaction_data/train_dict.json', 'w') as f:
            json.dump(mydict,f)



if __name__ == '__main__':
    main()

