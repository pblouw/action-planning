'''
Created on Jun 15, 2015

@author: bptripp
'''

import numpy as np
import nengo.spa as spa

# relation types ... 
IS_A = 'IsA'
HAS_AFFORDANCE = 'HasAffordance'
HAS_PRECONDITION = 'HasPrecondition' 
HAS_EFFECT = 'HasEffect'

# special concepts ... 
# WHAT = 'WHAT'
# WHERE = 'WHERE'
PRECONDITIONS = 'PRECONDITIONS'
EFFECTS = 'EFFECTS'
OBJECT = 'OBJECT'
# NOT = 'NOT'
# NULL = 'NULL'

def get_action_vocab(relations, D, spatial_keys, object_keys): 
    """
    Builds vocabularies for affordances and states from a list of concept 
    relationships.  
    
    Arguments:
    --------- 
    relations - List of concept relationships. Each relationship should be a 
        3-tuple (A, relation, B), where relation is in {IsA, HasAffordance, 
        HasPrecondition, HasEffect}, and A and B are either keys or HRR 
        expressions. 
    D - dimension of pointers
    spatial_keys: keys for spatial concepts (e.g. 'IN')
    object_keys: keys for objects (associated affordances not mandatory)
    
    Returns: 
    --------
    Vocabularies for affordances, states, spatial concepts, objects, and also 
    affordance component concepts (e.g. 'EFFECTS')
    """
    
    general_vocab = spa.vocab.Vocabulary(D) 
    
    affordance_vocab = spa.vocab.Vocabulary(D)
    state_vocab = spa.vocab.Vocabulary(D)
    
    for relation in relations: 
        if relation[1] == HAS_AFFORDANCE and not relation[2] in affordance_vocab.pointers: 
            affordance_vector = encode_affordance(relations, relation[2], general_vocab)
            affordance_vocab.add(relation[2], affordance_vector)
        if relation[1] == HAS_EFFECT or relation[1] == HAS_PRECONDITION: 
            state_key = relation[2].replace('*', '_')
            state_vector = general_vocab.parse(relation[2]).v
            if not state_key in state_vocab.keys:
                state_vocab.add(state_key, state_vector)
            
    spatial_vocab = copy_pointers(general_vocab, spatial_keys)
    object_vocab = copy_pointers(general_vocab, object_keys)
    component_vocab = copy_pointers(general_vocab, [PRECONDITIONS, EFFECTS, OBJECT])
    
    return affordance_vocab, state_vocab, spatial_vocab, object_vocab, component_vocab
    
def encode_affordance(relations, key, vocab):
    result = np.zeros(vocab.dimensions)
    for relation in relations: 
        if relation[1] == HAS_AFFORDANCE and relation[2] == key: 
            result = result + vocab.parse('%s*%s' % (OBJECT,relation[0])).v
#         if relation[1] == HAS_PRECONDITION and relation[0] == key: 
#             result = result + vocab.parse('%s*%s' % (PRECONDITIONS, relation[2])).v
        if relation[1] == HAS_EFFECT and relation[0] == key: 
            result = result + vocab.parse('%s*%s' % (EFFECTS, relation[2])).v
    return result

def copy_pointers(source_vocab, keys):
    result = spa.vocab.Vocabulary(source_vocab.dimensions)
    for key in keys: 
        result.add(key, source_vocab.parse(key))
    return result

def copy_all(source_vocab, dest_vocab):
    for key in source_vocab.keys:
        dest_vocab.add(key, source_vocab[key])

def get_kettle_relations():
    relations = [
        ('TAP', 'HasAffordance', 'FILL_KETTLE_FROM_TAP'), 
        ('KETTLE', 'HasAffordance', 'PUT_KETTLE_UNDER_TAP'), 
        ('KETTLE', 'HasAffordance', 'BOIL_KETTLE'), 
        ('KETTLE', 'HasAffordance', 'PLUG_IN_KETTLE'),
        ('KETTLE', 'HasAffordance', 'UNPLUG_KETTLE'),
        ('FILL_KETTLE_FROM_TAP', 'HasPrecondition', 'KETTLE*UNDER*TAP'), 
        ('FILL_KETTLE_FROM_TAP', 'HasEffect', 'WATER*IN*KETTLE'), 
        ('PUT_KETTLE_UNDER_TAP', 'HasPrecondition', 'KETTLE*UNPLUGGED'), 
        ('PUT_KETTLE_UNDER_TAP', 'HasEffect', 'KETTLE*UNDER*TAP'), 
        ('BOIL_KETTLE', 'HasPrecondition', 'WATER*IN*KETTLE'), 
        ('BOIL_KETTLE', 'HasPrecondition', 'KETTLE*PLUGGED_IN'), 
        ('BOIL_KETTLE', 'HasEffect', 'WATER*BOILED'), 
        ('PLUG_IN_KETTLE', 'HasEffect', 'KETTLE*PLUGGED_IN'),
        ('UNPLUG_KETTLE', 'HasEffect', 'KETTLE*UNPLUGGED')
        ]
    return relations

D = 100

spatial_keys = ['IN', 'UNDER', 'ON', 'BESIDE']
object_keys = ['WATER', 'TAP', 'KETTLE']
relations = get_kettle_relations()
affordance_vocab, state_vocab, spatial_vocab, object_vocab, component_vocab = get_action_vocab(relations, D, spatial_keys, object_keys)

print('--AFFORDANCES--')
for key in affordance_vocab.keys: 
    print(key)

print('--STATES--')
for key in state_vocab.keys: 
    print(key)

print('--SPATIALS--')
for key in spatial_vocab.keys: 
    print(key)

print('--OBJECTS--')
for key in object_vocab.keys: 
    print(key)

print('--COMPONENTS--')
for key in component_vocab.keys: 
    print(key)
