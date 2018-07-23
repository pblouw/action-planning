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


class MotorSystem:
    """
    A thing that receives action commands in the form of semantic pointers and 
    tries to perform these actions in the World. 
    """
    action_to_thing = {'UNPLUG_KETTLE':'KETTLE',
                       'PLUG_IN_KETTLE':'KETTLE',
                       'PUT_KETTLE_UNDER_TAP':'KETTLE',
                       'FILL_KETTLE_FROM_TAP':'TAP',
                       'BOIL_KETTLE':'KETTLE'}

    def __init__(self, vocab, world):
        self.vocab = vocab
        self.world = world
        self.threshold = 0.4
        self.last_action = None
        self.integration_time = 0.005  # if input consistent for this long then act
        self.integration_start_time = 0
        self.action_time = 0.1
        self.action_start_time = -self.action_time # allows action on intial input
    
    def act(self, time, action, subject=None, place=None):
        """
        Note: Subject and place are not used at the moment. They will probably
        be needed some day.
        
        action: the action (a semantic pointer)
        subject: optional thing to which the action is performed (semantic pointer, 
            can be zero)
        place: optional location at or to which the action is performed (semantic
            pointer, can be zero)
        """              
        action_text = get_key(self.vocab, action, self.threshold)
        if action_text not in self.action_to_thing.keys():
            return False

        if not action_text == self.last_action: 
            self.integration_start_time = time
            self.last_action = action_text
        
        if action_text is not None:

            if not self.integrating(time) and not self.acting(time): 
                print('performing action ' + action_text)
                self.world.do(self.action_to_thing[action_text], action_text)
                self.action_start_time = time

        return self.acting(time)


    def integrating(self, time):
        '''Checks whether input is stable enough over time to trigger change to world model'''
        return time < (self.integration_start_time + self.integration_time)
    
    def acting(self, time):
        '''Checks whether system is currently acting on the world model'''
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
        return 1 if self.act(time, action) else 0 
              

class VisualSystem:
    """
    A thing that receives location commands in the form of semantic pointers 
    and returns the states of any objects at these locations. In this application,
    the system is used to check whether an action's preconditions are satisfied, or
    whether the main goal is satisfied.
    """
    def __init__(self, vocab, world, trial=0, logging=False):
        self.vocab = vocab
        self.world = world
        self.threshold = 0.3
        self.last_location = None
        self.sensing = False
        self.start_time = 0
        self.ignore = False
        self.ignore_start = 0
        self.trial = trial
        self.logging = logging

        if 'DONE' in self.vocab.keys:
            self.state_to_key = {'UNPLUGGED':'KETTLE_UNPLUGGED',
                                 'UNDER-TAP':'KETTLE_UNDER_TAP',
                                 'BOILED':'WATER_BOILED',
                                 'IN-KETTLE':'WATER_IN_KETTLE'}
        else:
            self.state_to_key = {'UNPLUGGED':'PUT_KETTLE_UNDER_TAP',
                                 'UNDER-TAP':'FILL_KETTLE_FROM_TAP',
                                 'PLUGGED_IN':'UNPLUG_KETTLE',
                                 'IN-KETTLE':'PLUG_IN_KETTLE'}
        
    def sense(self, time, inp):
        val = 5
        key = get_key(self.vocab, inp, self.threshold)
        inp = normalize(inp)
        interval = 5 if 'DONE' in self.vocab.keys else 0.05

        scale = 5 if 'DONE' in self.vocab.keys else 0.35
        # Stop ignoring 350ms after ignoring is triggered
        if self.ignore and (time > self.ignore_start + scale):
            self.ignore = False

        # Return positive signal if sensing for duration of interval
        if self.sensing == True:
            if time - self.start_time > interval:
                self.sensing = False
            return val

        if not self.ignore:

            # Check object-based preconditions
            for thing in self.world.things: 
                states = thing.get_state().values()
                for state in states:
                    if state == 'PLUGGED_IN': # check for state with two preconditions
                        if self.both_preconditions(inp):
                            self.initialize_start_flags(time)
                            return val

                    if state in self.state_to_key: 
                        hrr = self.vocab.parse(self.state_to_key[state])
                        if hrr.compare(inp) > self.threshold:
                            self.initialize_start_flags(time)
                            return val

            # Check location-based preconditions  
            for loc_set in self.world.locations.values():
                for loc in loc_set:
                    if str(loc) in self.state_to_key:
                        hrr = self.vocab.parse(self.state_to_key[str(loc)])
                        if hrr.compare(inp) > self.threshold:
                            self.initialize_start_flags(time)
                            return val

        return 0

    def both_preconditions(self, inp): 
        try:
            if self.world.locations['WATER'][0].__str__() == 'IN-KETTLE':
                vec = self.vocab['BOIL_KETTLE']
                if vec.compare(inp) > self.threshold:
                    return True
        except:
            return False

    def initialize_start_flags(self, time):
        self.sensing = True
        self.ignore = True
        self.ignore_start = time
        self.start_time = time 
        if 'DONE' in self.vocab.keys and self.logging:
            self.write_to_log('2_goal_log.npy','2_time_log.npy', time)

    def write_to_log(self, goal_log_name, time_log_name, time):
        goal_log = np.load(goal_log_name)
        time_log = np.load(time_log_name)

        goal_log[self.trial] = 1 
        time_log[self.trial] = time

        np.save('2_goal_log.npy', goal_log)
        np.save('2_time_log.npy', time_log)

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
        
