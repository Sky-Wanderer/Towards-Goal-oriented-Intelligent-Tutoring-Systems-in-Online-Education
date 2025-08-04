
class MoocGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_knowledge(dataset)
        # self._construct_triple()
        
        self._clean()

    def _load_entities(self, dataset):
        print('load entities...')
        num_nodes = 0
        data_relations, _, _ = dataset.get_relation()  # entity_relations, relation_name, link_entity_type
        entity_list = list(data_relations.keys())
        for entity in entity_list:
            self.G[entity] = {}
            entity_size = getattr(dataset, entity).value_len
            for eid in range(entity_size):
                entity_rela_list = data_relations[entity].keys()
                self.G[entity][eid] = {r: [] for r in entity_rela_list}
            num_nodes += entity_size
            print('load entity:{:s}  : Total {:d} nodes.'.format(entity, entity_size))
        print('ALL total {:d} nodes.'.format(num_nodes))
        print('===============END==============')

    def _load_knowledge(self, dataset):
        _, data_relations_name, link_entity_type = dataset.get_relation()  # entity_relations, relation_name, link_entity_type
        for relation in data_relations_name:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for he_id, te_ids in enumerate(data):  # head_entity_id , tail_entity_ids
                if len(te_ids) <= 0:
                    continue
                e_head_type = link_entity_type[relation][0]
                e_tail_type = link_entity_type[relation][1]
                for te_id in set(te_ids):
                    self._add_edge(e_head_type, he_id, relation, e_tail_type, te_id)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))
        print('===============END==============')
        
    # def _construct_triple(self):
    #     relation_id={
    #         "PRACTICE_C": 1,
    #         "PRACTICE_W": 2,
    #         "SUCCESSOR":  3,
    #         "BELONG_TO":  4
    #     }
    #     triple_relation={
    #         "PRACTICE_C": ['user', 'exercise'],
    #         "PRACTICE_W":  ['USER', 'EXERCISE'],
    #         "SUCCESSOR":  ['CONCEPT', 'CONCEPT'],
    #         "BELONG_TO":  ['CONCEPT', 'EXERCISE']
    #     }
    #     entity2id=[]
    #     relation2id=[]
    #     train2id=[]
        
    #     users = self.G['user'].keys()
    #     exercise = self.G['exercise'].keys()
    #     concept = self.G['concept'].keys()
    #     user_length=len(users)
    #     exercise_length=len(exercise)
    #     concept_start= user_length + exercise_length
    #     num_entity = user_length + exercise_length + len(concept)
    #     for relation in relation_id:
    #         if relation in ['practice']:
    #             for user, exercises in self.G['user'].items():
    #                 for exercise_id in exercises['practice_correct']:
    #                     graph[user][exercise_id+user_length] = 1
    #                     graph[exercise_id+user_length][user] = 1
    #                 for exercise_id in exercises['practice_wrong']:
    #                     graph[user][exercise_id+user_length] = -1
    #                     graph[exercise_id+user_length][user] = -1
            
        
    # def write_list_of_lists_to_file(data_list, file_path):
    #     # Get the length of the list of lists
    #     list_length = len(data_list)

    #     with open(file_path, 'w') as f:
    #         # Write the length of the list on the first line
    #         f.write(str(list_length) + '\n')

    #         # Write each sublist as a separate line
    #         for sublist in data_list:
    #             # Convert each element in the sublist to a string and join them with tabs
    #             line_data = '\t'.join(str(item) for item in sublist)
    #             f.write(line_data + '\n')
    

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        if relation in 'predecessor' or relation in 'successor':
            self.G[etype1][eid1][relation].append(eid2)
        else:
            self.G[etype1][eid1][relation].append(eid2)           
            self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data
                    



                    
 
        
