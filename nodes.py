'''
Created on Jun 11, 2015

@author: bptripp
'''

import numpy as np
import world
import nengo.spa as spa

def get_key(vocab, pointer, threshold):
    key = None
    similarities = vocab.dot(pointer)
    ind = np.argmax(similarities)
    if similarities[ind] >= threshold:
        key = vocab.keys[ind]
    return key
    
class MotorSystem:
    """
    A thing that receives action commands in the form of semantic pointers and 
    tries to perform these actions in the World. 
    """
    
    def __init__(self, vocab, world):
        self.vocab = vocab
        self.world = world
        self.THRESHOLD = 0.3
        self.last_thing = None
        self.last_action = None
        self.integration_time = 0.01 #if input consistent for this long then act
        self.integration_start_time = 0
        self.action_time = 0.15
        self.action_start_time = -self.action_time
    
    def act(self, time, thing, action, subject=None, place=None):
        """
        Note: Subject and place are not used at the moment. They will probably
        be needed some day.
        
        thing: the object with which the action is performed (a semantic pointer)
        action: the action (a semantic pointer)
        subject: optional thing to which the action is performed (semantic pointer, 
            can be zero)
        place: optional location at or to which the action is performed (semantic
            pointer, can be zero)
        """
        
        thing_text = get_key(self.vocab, thing, self.THRESHOLD)
        if not thing_text == self.last_thing: 
            self.integration_start_time = time
            self.last_thing = thing_text             
        
        action_text = get_key(self.vocab, action, self.THRESHOLD)
        if not action_text == self.last_action: 
            self.integration_start_time = time
            self.last_action = action_text

#         print(thing_text)
        
        if thing_text is not None and action_text is not None:
            # print('integrating ' + action_text + ' with ' + thing_text)
                    
            if not self.integrating(time) and not self.acting(time): 
                print('performing action ' + action_text + ' with ' + thing_text)
                self.world.do(thing_text, action_text)
                self.action_start_time = time

        return self.acting(time)


    def integrating(self, time):
        return time < (self.integration_start_time + self.integration_time)
    
    def acting(self, time):
        return time < (self.action_start_time + self.action_time)
    
    def __call__(self, time, value):
        """
        Arguments: 
        ----------
        time: Time within simulation
        value: concatenation of thing and action semantic pointers
        
        Returns: 
        --------
        1 if an action is in progress, 0 if not  
        """
        
        thing = value[:self.vocab.dimensions]
        action = value[self.vocab.dimensions:]
        
#         print('time: %f' % time)
#         print(np.dot(thing, self.vocab['KETTLE'].v))
        
        return 1 if self.act(time, thing, action) else 0 
               
        
        
class VisualSystem:
    """
    A thing that receives location commands in the form of semantic pointers 
    and returns the states of any objects at these locations.   
    """
    
    def __init__(self, vocab, world):
        self.vocab = vocab
        self.world = world
        self.THRESHOLD = 0.2
        self.last_location = None
        self.last_perception = spa.pointer.SemanticPointer(np.zeros(self.vocab.dimensions))
        self.integration_time = 0.1 #if location consistent for this long then sense
        self.integration_start_time = 0 
        
    def sense(self, time, location):
        if not isinstance(location, spa.pointer.SemanticPointer): 
            location = spa.pointer.SemanticPointer(location)
                        
        # TODO: should we allow compound locations, e.g. on-counter & beside-sink?
        # TODO: could also specify kind
        inv_relation = self.vocab.parse('~(IN+ON+UNDER+BESIDE)') #this seems touchy -- getting some incorrect results with D=100
        thing = location.convolve(inv_relation)
        relation = location.convolve(thing.__invert__())
        
        thing_key = get_key(self.vocab, thing, self.THRESHOLD)
        relation_key = get_key(self.vocab, relation, self.THRESHOLD)
        
        loc = None
        if thing_key is not None and relation_key is not None:
            loc = world.Location(relation_key, self.world.get(thing_key))
        
        print('integrating: ' + str(loc))

        if loc == None or not loc == self.last_location: 
            self.integration_start_time = time
            self.last_location = loc
            
        if not self.integrating(time) : 
            self.last_perception = spa.pointer.SemanticPointer(np.zeros(self.vocab.dimensions))
            things_at_location = self.world.at_location(loc) 
            for t in things_at_location: 
                print(t)
                self.last_perception = self.last_perception + self.encode_thing(t)
                 
        return self.last_perception
    
    def integrating(self, time):
        return time < (self.integration_start_time + self.integration_time)
        
    def __call__(self, time, input):
        """
        Arguments: 
        ----------
        time: Time within simulation
        input: Convolution of relation and thing that make up a location (e.g. BESIDE*BED)
        
        Returns: 
        --------
        Bundle of states and locations of things, e.g. BOOK*OPEN + BOOK*BESIDE*BED + LIGHT*OFF
        """
        return self.sense(time, input)
        
        
    def encode_thing(self, thing):
        """
        Arguments
        ---------
        thing: A world.Thing
        
        Returns
        -------
        An HRR representation of the thing's type, location, and state. 
        """
        
        kind_key = thing.kinds[0]
        kind_ptr = self.vocab[kind_key]         

        result = spa.pointer.SemanticPointer(np.zeros(self.vocab.dimensions))
        
        state = thing.get_state() 
        for key in state.keys():            
            state_ptr = self.vocab[state[key]]
            result = result + kind_ptr.convolve(state_ptr)
        
        for location in self.world.get_location(thing): 
            rel_ptr = self.vocab[location.relationship]
            other_kind_ptr = self.vocab[location.thing.kinds[0]]
            loc_vector = rel_ptr.convolve(other_kind_ptr)
            result = result + loc_vector
        
        return result
        
