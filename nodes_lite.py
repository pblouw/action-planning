'''
Created on Jun 11, 2015

@author: bptripp

Edits by pblouw for Cogsci paper. Note that this is currently a bit 
of a mess, and I'm planning on cleaning it up shortly to eliminate 
all of the code duplication etc. 
'''

import numpy as np
import world
import nengo.spa as spa
import numpy as np


def normalize(a): 
    """ Normalizes a vector to length 1 (this should be done with composite HRRs.) 

    Arguments: 
    a: a vector
    Returns:
    aa: normalized to unit length
    """
    
    result = a
    n = np.linalg.norm(a)
    if n > 0:
        result = a / n
    return result

def get_key(vocab, pointer, threshold):
    key = None
    similarities = vocab.dot(pointer)
    ind = np.argmax(similarities)
    if similarities[ind] >= threshold:
        key = vocab.keys[ind]
    return key


class CheckSystem:
    """
    A thing that receives location commands in the form of semantic pointers 
    and returns the states of any objects at these locations.   
    """
    
    def __init__(self, vocab, world):

        self.vocab = vocab
        self.world = world
        self.THRESHOLD = 0.3
        self.last_location = None
        self.last_perception = spa.pointer.SemanticPointer(np.zeros(self.vocab.dimensions))
        self.sensing = False
        self.start_time = 0
        self.ignore = False
        self.ignore_start = 0
        # self.integration_start_time = 0 

        self.state_vocab = spa.Vocabulary(self.vocab.dimensions)
        
    def sense(self, time, inp):
        val = 5
        key = get_key(self.vocab, inp, 0.3)
        inp = normalize(inp)
        interval = 5 if 'DONE' in self.vocab.keys else 0.05



        state_to_key = {'UNPLUGGED':'KETTLE_UNPLUGGED',
                        'UNDER-TAP':'KETTLE_UNDER_TAP',
                        # 'PLUGGED_IN':'KETTLE_PLUGGED_IN',
                        'BOILED':'WATER_BOILED',
                        'IN-KETTLE':'WATER_IN_KETTLE'}

        if self.ignore and (time > self.ignore_start + 5):
            self.ignore = False

        if self.sensing == True:
            if time - self.start_time > interval:
                self.sensing = False
                # print self.sensing, ' Sense State at ', time
            return val

        if not self.ignore:
            for thing in self.world.things: 
                states = thing.get_state().values()
                for state in states:
                    if state in state_to_key:   
                        if self.vocab.parse(state_to_key[state]).compare(inp) > self.THRESHOLD:
                            self.sensing = True
                            self.ignore = True
                            self.ignore_start = time
                            # print self.sensing, ' Sense State at ', time
                            self.start_time = time
                            return val


            for loc_set in self.world.locations.values():
                for loc in loc_set:
                    if str(loc) in state_to_key:
                        if self.vocab.parse(state_to_key[str(loc)]).compare(inp) > self.THRESHOLD:
                            self.sensing = True
                            self.ignore = True
                            self.ignore_start = time
                            # print self.sensing, ' Sense State at ', time
                            self.start_time = time 
                            return val
        return 0
    

    def __call__(self, time, input):
        """
        Arguments: 
        ----------
        time: Time within simulation
        input: A Semantic Pointer encoding a precondition to be checked. 
        
        Returns: 
        --------
        1 if precondition is satisfied by world state, 0 otherwise. 
        """
        return self.sense(time, input)


class MotorSystem:
    """
    A thing that receives action commands in the form of semantic pointers and 
    tries to perform these actions in the World. 
    """
    
    def __init__(self, vocab, world):
        self.vocab = vocab
        self.world = world
        self.THRESHOLD = 0.4
        self.last_thing = None
        self.last_action = None
        self.integration_time = 0.005 #if input consistent for this long then act
        self.integration_start_time = 0
        self.action_time = 0.1
        self.action_start_time = -self.action_time
        self.n = 0

        self.last_action_time = None
    
    def act(self, time, action, subject=None, place=None):
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
        actions = ['FILL_KETTLE_FROM_TAP', 'PUT_KETTLE_UNDER_TAP', 'BOIL_KETTLE', 'PLUG_IN_KETTLE','UNPLUG_KETTLE']

        action_to_thing = {'UNPLUG_KETTLE':'KETTLE',
                           'PLUG_IN_KETTLE':'KETTLE',
                           'PUT_KETTLE_UNDER_TAP':'KETTLE',
                           'FILL_KETTLE_FROM_TAP':'TAP',
                           'BOIL_KETTLE':'KETTLE'}
      
        
        action_text = get_key(self.vocab, action, self.THRESHOLD)
        if action_text not in actions:
            return 0

        if not action_text == self.last_action: 
            self.integration_start_time = time
            self.last_action = action_text
        
        if action_text is not None:

            if not self.integrating(time) and not self.acting(time): 
                print 'performing action ' + action_text
                self.world.do(action_to_thing[action_text], action_text)
                self.action_start_time = time

        return self.acting(time)


    def integrating(self, time):
        return time < (self.integration_start_time + self.integration_time)
    
    def acting(self, time):
        return time < (self.action_start_time + self.action_time)
    
    def __call__(self, time, action):
        """
        Arguments: 
        ----------
        time: Time within simulation
        value: concatenation of thing and action semantic pointers
        
        Returns: 
        --------
        1 if an action is in progress, 0 if not  
        """
        
        # thing = value[:self.vocab.dimensions]
        # action = value[self.vocab.dimensions:]
          
        return 1 if self.act(time, action) else 0 
               

class VisualSystem:
    """
    A thing that receives location commands in the form of semantic pointers 
    and returns the states of any objects at these locations.   
    """
    
    def __init__(self, vocab, world):
        self.vocab = vocab
        self.world = world
        self.THRESHOLD = 0.3
        self.last_location = None
        self.last_perception = spa.pointer.SemanticPointer(np.zeros(self.vocab.dimensions))
        self.sensing = False
        self.start_time = 0
        self.ignore = False
        self.ignore_start = 0
        # self.integration_start_time = 0 

        self.state_vocab = spa.Vocabulary(self.vocab.dimensions)
        
    def sense(self, time, inp):
        val = 5
        key = get_key(self.vocab, inp, 0.3)
        inp = normalize(inp)
        interval = 0.5 if 'DONE' in self.vocab.keys else 0.05

        state_to_key = {'UNPLUGGED':'PUT_KETTLE_UNDER_TAP',
                        'UNDER-TAP':'FILL_KETTLE_FROM_TAP',
                        'PLUGGED_IN':'UNPLUG_KETTLE',
                        'IN-KETTLE':'PLUG_IN_KETTLE'}

        # state_to_key = {'PLUGGED_IN':'KETTLE_UNPLUGGED',
        #                 'UNDER-TAP':'KETTLE_UNDER_TAP',
        #                 # 'BOILED':'WATER_BOILED',
        #                 'IN-KETTLE':'WATER_IN_KETTLE'}

        if self.ignore and (time > self.ignore_start + 0.35):
            self.ignore = False

        if self.sensing == True:
            if time - self.start_time > interval:
                self.sensing = False
                # print self.sensing, ' Sense State at ', time
            return val

        if not self.ignore:
            for thing in self.world.things: 
                states = thing.get_state().values()
                for state in states:
                    if state == 'PLUGGED_IN':
                        try:
                            if self.world.locations['WATER'][0].__str__() == 'IN-KETTLE':
                                vec = self.vocab['BOIL_KETTLE']
                                if vec.compare(inp) > self.THRESHOLD:
                                    self.sensing = True
                                    self.ignore = True
                                    self.ignore_start = time
                                    # print self.sensing, ' Sense State at ', time
                                    self.start_time = time
                                    return val
                        except:
                            pass

                    if state in state_to_key:   
                        # else:
                         if self.vocab.parse(state_to_key[state]).compare(inp) > self.THRESHOLD:
                            self.sensing = True
                            self.ignore = True
                            self.ignore_start = time
                            # print self.sensing, ' Sense State at ', time
                            self.start_time = time
                            return val

            for loc_set in self.world.locations.values():
                for loc in loc_set:
                    if str(loc) in state_to_key:
                        if self.vocab.parse(state_to_key[str(loc)]).compare(inp) > self.THRESHOLD:
                            self.sensing = True
                            self.ignore = True
                            self.ignore_start = time
                            # print self.sensing, ' Sense State at ', time
                            self.start_time = time 
                            return val

        return 0




        
    def __call__(self, time, input):
        """
        Arguments: 
        ----------
        time: Time within simulation
        input: A Semantic Pointer encoding a precondition to be checked. 
        
        Returns: 
        --------
        1 if precondition is satisfied by world state, 0 otherwise. 
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
        
