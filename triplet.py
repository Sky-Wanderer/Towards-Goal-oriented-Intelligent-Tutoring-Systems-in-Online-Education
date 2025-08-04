
import argparse
from utils import *
import json

kg_file = '/Users/ff/Desktop/aaai/Neural_Cognitive_Diagnosis-NeuralCD/mooc_data/kg.pkl'
kg = pickle.load(open(kg_file, 'rb'))
user_info=kg.G['user']
exer_info=kg.G['exercise']
concept_info=kg.G['concept']


def generate_train2id(train_data,concept_data, entity2id, relation2id):
    with open('train2id.txt', 'w') as f:
        total_triples = 0
        
        for user_id, user_info in train_data.items():
            practice_correct = user_info.get('practice_correct', ())
            practice_wrong= user_info.get('practice_wrong', ())
            for exercise_id in practice_correct:
                f.write(f"{entity2id['User'+str(user_id)]} {entity2id['Exercise'+str(exercise_id)]} {relation2id['practice_correct']}\n")
                total_triples += 1
            for exercise_id in practice_wrong:
                f.write(f"{entity2id['User'+str(user_id)]} {entity2id['Exercise'+str(exercise_id)]} {relation2id['practice_wrong']}\n")
                total_triples += 1
        for concept_id, concept_info in concept_data.items():
            belong_to = concept_info.get('belong_to', ())
            successor= concept_info.get('successor', ())
            for exercise_id in belong_to:
                f.write(f"{entity2id['Concept'+str(concept_id)]} {entity2id['Exercise'+str(exercise_id)]} {relation2id['belong_to']}\n")
                total_triples += 1
            for concept_id2 in successor:
                f.write(f"{entity2id['Concept'+str(concept_id)]} {entity2id['Concept'+str(concept_id2)]} {relation2id['successor']}\n")
                total_triples += 1
    return total_triples

def generate_entity2id(concept_data, user_data, exercise_data):
    with open('entity2id.txt', 'w') as f:
        total_entities = -1
        entity2id = {}
        
        for user_id in user_data.keys():
            entity2id[f"User{user_id}"] = total_entities + 1
            f.write(f"User{user_id} {total_entities + 1}\n")
            total_entities += 1

        for exercise_id in exercise_data.keys():
            entity2id[f"Exercise{exercise_id}"] = total_entities + 1
            f.write(f"Exercise{exercise_id} {total_entities + 1}\n")
            total_entities += 1
            
        for concept_id, concept_info in concept_data.items():
            entity2id[f"Concept{concept_id}"] = total_entities + 1
            f.write(f"Concept{concept_id} {total_entities + 1}\n")
            total_entities += 1


    return entity2id

def generate_relation2id():
    with open('relation2id.txt', 'w') as f:
        total_relations = 0
        relation2id = {
            'belong_to': total_relations+2,
            'successor': total_relations + 3,
            'practice_correct':total_relations,
            'practice_wrong':total_relations+1
        }
        f.write(f"{total_relations + 4}\n")
        f.write(f"belong_to {total_relations+2}\n")
        f.write(f"successor {total_relations + 3}\n")
        f.write(f"practice_correct {total_relations + 1}\n")
        f.write(f"practice_wrong {total_relations}\n")
        return relation2id


# Generate relation2id.txt
relation2id = generate_relation2id()

# Generate entity2id.txt
entity2id = generate_entity2id(concept_info, user_info, exer_info)

# Generate train2id.txt
total_triples = generate_train2id(user_info,concept_info, entity2id, relation2id)

print(f"Total triples: {total_triples}")
