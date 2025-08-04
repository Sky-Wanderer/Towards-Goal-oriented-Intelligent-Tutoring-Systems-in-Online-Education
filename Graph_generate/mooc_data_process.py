import os
import json
from easydict import EasyDict as edict
import random


class MoocDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir + '/Graph_generate_data'
        self.map_info=data_dir+'/id_map_info'
        self.practice_data=data_dir+'/practice_data'
        
        if not os.path.exists(os.path.join(self.map_info, 'entity_dict.json')):
            self.merge_data()
            
        self.load_entities()
        self.load_relations()
        
    def get_relation(self):
        #Entities
        USER = 'user'
        EXERCISE = 'exercise'
        CONCEPT = 'concept'

        #Relations
        PRACTICE_C = 'practice_correct'
        PRACTICE_W = 'practice_wrong'
        SUCCESSOR = 'successor'
        PREDECESSOR = 'predecessor'
        BELONG_TO = 'belong_to'
        relation_name = [PRACTICE_C, PRACTICE_W, SUCCESSOR, PREDECESSOR, BELONG_TO]

        fm_relation = {
            USER: {
                PRACTICE_C: EXERCISE,
                PRACTICE_W: EXERCISE 
            },
            EXERCISE: {
                BELONG_TO: CONCEPT,
                PRACTICE_C: USER,
                PRACTICE_W: USER
            },
            CONCEPT: {
                BELONG_TO: EXERCISE,
                SUCCESSOR: CONCEPT,
                PREDECESSOR:CONCEPT
                
            }
        }
        fm_relation_link_entity_type = {
            PRACTICE_C: [USER, EXERCISE],
            PRACTICE_W:  [USER, EXERCISE],
            SUCCESSOR:  [CONCEPT, CONCEPT],
            BELONG_TO:  [CONCEPT, EXERCISE],
            PREDECESSOR:  [CONCEPT, CONCEPT]
        }
        return fm_relation, relation_name, fm_relation_link_entity_type
    
    def load_entities(self):
        
        
        with open(os.path.join(self.map_info,'entity_dict.json'), encoding='utf-8') as f:
                    data = json.load(f)
        
        for entity_name in ['user','exercise','concept']:
            entity_id=list(data[entity_name].values())
            setattr(self, entity_name,edict(id=entity_id, value_len=max(entity_id)+1))
            print('Load', entity_name, 'of size', len(entity_id))
            print(entity_name, 'of max id is', max(entity_id))

    def load_relations(self):
        """
        relation: head entity---> tail entity
        --
        """
        LastFm_relations = edict(
            practice_correct=('user_exercise.json', self.user, self.exercise), #(filename, head_entity, tail_entity)
            practice_wrong=('user_exercise.json', self.user, self.exercise),
            successor=('concept_exercise.json', self.concept, self.concept),
            predecessor=('concept_exercise.json', self.concept, self.concept),
            belong_to=('concept_exercise.json', self.concept, self.exercise)
        )
        for name in LastFm_relations:
            #  Save tail_entity
            relation = edict(
                data=[],
            )
            knowledge = [list([]) for i in range(LastFm_relations[name][1].value_len)]
            # load relation files
            with open(os.path.join(self.data_dir,LastFm_relations[name][0]), encoding='utf-8') as f:
                mydict = json.load(f)
                if name in ['practice_correct']:
                    for key, value in mydict.items():
                        head_id = int(key)
                        tail_ids=[]
                        tail_ids = [item['exercise_id'] for item in value if item['is_correct'] == 1]
                        knowledge[head_id] = tail_ids
                if name in ['practice_wrong']:
                    for key, value in mydict.items():
                        head_id = int(key)
                        tail_ids=[]
                        tail_ids = [item['exercise_id'] for item in value if item['is_correct'] == 0]
                        knowledge[head_id] = tail_ids
                elif name in ['belong_to']:
                    for key in mydict.keys():
                        head_str = key
                        head_id = int(key)
                        tail_ids = mydict[head_str][0]['exercise']
                        knowledge[head_id] = tail_ids
                elif name in ['successor']:
                    for key in mydict.keys():
                        head_str = key
                        head_id = int(key)
                        tail_ids = mydict[head_str][0]['successor']                      
                        knowledge[head_id] = tail_ids
                elif name in ['predecessor']:
                    for key in mydict.keys():
                        head_str = key
                        head_id = int(key)
                        tail_ids = mydict[head_str][0]['predecessor']                        
                        knowledge[head_id] = tail_ids
            relation.data = knowledge
            setattr(self, name, relation)
            tuple_num = 0
            for i in knowledge:
                tuple_num += len(i)
            print('Load', name, 'of size', tuple_num)



    def merge_data(self):
        user_data = {}  # Dictionary to store merged user data

        entity_files = edict(
            user='user_dict.json',
            concept='concept_dict.json'
        )
        entity_classes = edict(
            cs='cs_',
            math='math_',
            psy="psy_"
        )
        user_practice={}
        user_dict = {} 
        user_counter = 0  
        exercise_dict = {} 
        exercise_counter = 0  
        concept_dict={}
        concept_data={}
        concept_counter=0

        for entity_name in entity_files:
            
            for entity_class in entity_classes: 
                with open(os.path.join(self.data_dir,entity_classes[entity_class]+entity_files[entity_name]), encoding='utf-8') as f:
                    data = json.load(f)
                if entity_name in ['user']:
                    # Loop through the data in each file
                    for user_id, exercises in data.items():
                        if user_id not in user_dict:
                            user_dict[user_id] = user_counter
                            user_counter+=1
                            user_practice[user_dict[user_id]] = []

                        for exercise_info in exercises:
                            exercise_id = exercise_info['exercise']
                            # time = exercise_info['time']
                            if exercise_id not in exercise_dict:
                                exercise_dict[exercise_id] = exercise_counter
                                exercise_counter += 1
                            user_practice[user_dict[user_id]].append({
                            'exercise_id': exercise_dict[exercise_id],
                            'is_correct': exercise_info['is_correct'],
                        })
                        
                else:
                    for concept_id, exercises in data.items():
                        if concept_id not in concept_dict:
                            concept_dict[concept_id] = concept_counter
                            concept_counter+=1
                            concept_data[concept_dict[concept_id]] = []
                    for concept_id, exercises in data.items():
                            concept_data[concept_dict[concept_id]].append({
                                'exercise':[exercise_dict[id_] for id_ in exercises['exercise']],
                                'successor':[concept_dict[id_] for id_ in exercises['successor']],
                                'predecessor':[concept_dict[id_] for id_ in exercises['predecessor']]
                            })
                        
                        

                
               
        print('number of user:',user_counter,'\nnumber of exercises:', exercise_counter,'\nnumber of concept:', concept_counter)

                        # Add the exercise to the user's data with the new exercise ID

        # Write the merged user data to a new file
        entity_dict={}
        entity_dict['user']=user_dict
        entity_dict['exercise']=exercise_dict
        entity_dict['concept']=concept_dict
        with open(os.path.join(self.map_info,"entity_dict.json"), 'w', encoding='utf-8') as f:
               json.dump(entity_dict, f, indent=2, ensure_ascii=False)
        user_practice=self.remove_duplicate_exercises(user_practice)
        train_data, test_data, valid_data=self.split_user_data(user_practice)
        
        with open(os.path.join(self.data_dir,'user_exercise.json'), 'w', encoding='utf-8') as f:
            json.dump(user_practice, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.data_dir,'concept_exercise.json'), 'w', encoding='utf-8') as f:
            json.dump(concept_data, f, indent=2, ensure_ascii=False)
            
        with open(os.path.join(self.practice_data,'practice_train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.practice_data,'practice_test.json'), 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.practice_data,'practice_valid.json'), 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, indent=2, ensure_ascii=False)
        


    def remove_duplicate_exercises(self,user_practice):
        for user_id, user_info in user_practice.items():
            unique_exercises = []  # List to store unique exercises for this user
            seen_exercise_ids = set()  # Set to track seen exercise IDs for this user

            for exercise_info in user_info:
                exercise_id = exercise_info['exercise_id']
                
                # Check if the exercise ID has been seen for this user before
                if exercise_id not in seen_exercise_ids:
                    seen_exercise_ids.add(exercise_id)
                    unique_exercises.append(exercise_info)
            
            # Update the user's information with the unique exercises
            user_practice[user_id] = unique_exercises
            return user_practice

    def split_user_data(self,user_practice, train_ratio=0.6, test_ratio=0.3, valid_ratio=0.1):
        train_data, test_data, valid_data = {}, {}, {}
        for user_id, user_info in user_practice.items():
            # Shuffle the exercises for each user to ensure random distribution
            # random.shuffle(user_info)
            
            total_exercises = len(user_info)
            
            # Calculate the number of exercises for each set based on the ratios
            train_size = int(train_ratio * total_exercises)
            test_size = int(test_ratio * total_exercises)
            
            # Split the exercises into train, test, and valid sets
             # Split the exercises into train, test, and valid sets
            train_data[user_id] = user_info[:train_size]
            test_data[user_id] = user_info[train_size:train_size + test_size]
            valid_data[user_id] = user_info[train_size + test_size:]
            
        return train_data, test_data, valid_data