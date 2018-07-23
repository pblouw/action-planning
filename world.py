# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 13:21:29 2014

@author: bptripp
"""
import random 

relative_locations = ['ON', 'IN', 'UNDER', 'BESIDE'] # these define regions of space relative to things

class World(object): 
    """
    A symbolic environment wrapper that the planner can interact with. The 
    planner can sense properties of this environment directly, and cause effects
    directly, rather than going through sensory and motor systems (which would be 
    more realistic but also off topic). 
    """
    
    def __init__(self):
        self.things = []
        self.locations = {}
    
    def add(self, thing, locations=[]):    
        """
        Add a thing to the world, possibly at a certain approximate position, 
        where the position is given by locations relative to other things. 
        """
        
        assert isinstance(thing, Thing)
        if isinstance(locations, Location):
            locations = [locations]
        self.things.append(thing)
        self.locations[thing.ID] = locations #TODO: enforce uniqueness?
    
    def remove(self, thing): 
        """
        Take a thing out of the world. 
        """
        
        if thing in self.things: 
            self.things.remove(thing)
            self.locations.pop(thing.ID)
            
    def _can_put(self, thing, location): 
        """
        Returns true by default. This can be overridden to take geometric constraints 
        into account (e.g. if something is too big to go in something else, or if the
        something else is full, etc.)
        """
        assert isinstance(thing, Thing)
        assert isinstance(location, Location)
        return True
        
    def get(self, ID): 
        """
        Retrieve a thing by it's ID. 
        """
        for thing in self.things: 
            if thing.ID == ID: 
                return thing        
        raise Exception('%s is not part of this World' % ID)
    
    def put(self, thing, locations): 
        """
        This implementation keeps track of Locations to which it is moved and 
        returns them verbatim in get_location(). A subclass might keep track of 
        3D coordinates, make decisions about where within a Location exactly a 
        put() call should put the Thing, and evaluate the truth of has_location() 
        calls based on coordinates. 
        
        Arguments: 
        thing - The thing to put somewhere. 
        locations - List of locations relative to other objects. A subclass may 
            have logic that choses a more specific location within these 
            constraints. 
        """
        assert type(locations) is list
        for location in locations: 
            if not self._can_put(thing, location): 
                raise Exception('Cannot put %s %s' % (thing, location))
        
        self.locations[thing.ID] = locations #default behaviour is to just adopt the given list        
            
    def get_location(self, thing): 
        """
        Returns: 
        List of Locations that represent overlapping regions of space (e.g. [on-counter, beside-sink])        
        """
        return self.locations[thing.ID]
        
    def has_location(self, thing, locations):
        """
        Returns: 
        True if the given Location matches the location of the given Thing. By 
        default, True if there is an exact match between the given Location and 
        one of the thing's LocationConditions. Subclasses may do something more 
        sophisticated, e.g. evaluate implied locations such as e.g. return True 
        if the given Location is in-fridge and the current location is in-container 
        where container is in-fridge, or interpret based on 3D coordinates. 
        """
        if isinstance(locations, Location): 
            locations = [locations]
            
        for location in locations:
            assert isinstance(location, Location)
            match = False
            if isinstance(thing, Thing):
                for l in self.get_location(thing):
                    if l == location:
                        match = True
                        break
            if not match: 
                return False
        return True
    
    def at_location(self, locations): 
        """
        Returns a list of things that are at given locations. The locations are 
        multiple constraints on a single location (e.g. 'on the counter beside
        the sink')
        """
        result = []
        for thing in self.things: 
            if self.has_location(thing, locations): 
                result.append(thing)
        return result
    
    def scan(self): 
        """
        Returns a random thing. 
        
        A few calls could be used to get a sense of the kinds of things that
        are in an environment, for the purpose of categorizing the environment
        so that AtLocation relations (from ConceptNet) could be used to guess 
        at what else is in the environment. 
        """
        return random.choice(self.things)
        
    def do(self, ID, action, what=None, where=None): 
        """
        Tries to perform an action. 
        
        Arguments: 
        thing - Name of the Thing of which the action is an affordance. 
        action - Name of action to perform. 
        what - An optional other Thing involved in the action. 
        where - An optional Location that somehow parameterizes the action. 
        
        Returns: 
        True if the action was performed successfully; False if not. 
        """
        thing = None
        for t in self.things: 
            if t.ID == ID: 
                thing = t
        if thing == None: 
            raise Exception('thing %s not found' % ID)
            
        action = action.lower()
        
        method = None
        for m in dir(thing): 
            if m == action and callable(getattr(thing, m)): 
                method = getattr(thing, m)
                break
            
        if method is not None: 
            arg_names = method.__code__.co_varnames
            if 'what' in arg_names and 'where' in arg_names: 
                return method(what=what, where=where)
            elif 'what' in arg_names: 
                return method(what=what)
            elif 'where' in arg_names: 
                return method(where=where)
            else: 
                return method()
        else: 
            raise Exception('method %s not found' % action)
                                
    def print_state(self):
        for thing in self.things: 
            print(thing.ID)
            print(thing.state)
            for location in self.locations[thing.ID]:
                print(location)
            print("------------------")
                
class Thing(object): 
    """
    An actual physical thing. Affordances are to be defined on subclasses as 
    python methods with kwargs "what" and "where", which return True if successful. 
    """
    
    def __init__(self, the_world, ID, kinds, possible_states={}, locations=[]): 
        """
        Arguments: 
        the_world - World to which the Thing belongs
        ID - A unique label to distinguish this Thing from other Things. 
        kinds - List of names of types that this Thing belongs to (e.g. 'container')
        """
        self.world = the_world
        self.ID = ID
        self.kinds = kinds
        
        self.world.add(self, locations)
        
        if not isinstance(self.kinds, list): 
            self.kinds = [self.kinds]
            
        self.state = {}        

        # create lookup table for kinds of states
        assert isinstance(possible_states, dict)
        self._state_kinds = {}
        for key in possible_states.keys():
            self.state[key] = possible_states[key][0]
            for val in possible_states[key]:
                self._state_kinds[val] = key
    
    def set_state(self, state):
        kind = self._state_kinds[state]
        self.state[kind] = state
        
    def has_state(self, state):
        """
        Returns: 
        True if this Thing currently has the given state (e.g. 'hot'), False otherwise
        """
        kind = self._state_kinds[state] if self._state_kinds.has_key(state) else None
        return self.state.has_key(kind) and self.state[kind] == state
        
    def get_state(self): 
        """
        Returns: 
        Dictionary of state_type->state_name, e.g. 'temperature': 'cold'
        """
        return self.state
        
        
class MoveableThing(Thing): 
    """
    A Thing that can be moved. 
    """
    def __init__(self, the_world, ID, kinds, possible_states={}, locations=[]): 
        Thing.__init__(self, the_world, ID, kinds, possible_states, locations)
    
    def put(self, where=None): 
        if where == None:
            return False
            
        if not isinstance(where, list): 
            where = [where]
            
        self.world.put(self, where)
        
        return True
        
#TODO: may be cleaner to add Position class that can include multiple Location constraints 
#   and supports comparisons (currently passing lists of locations around as args)
class Location(object): 
    """
    A symbolic location relative to another object. 
    """
    
    def __init__(self, relationship, thing):
        assert relationship in relative_locations
        assert isinstance(thing, Thing) or isinstance(thing, str)
        self.relationship = relationship
        self.thing = thing
    
    def __str__(self): 
        tn = self.thing if not isinstance(self.thing, Thing) else self.thing.ID
        return '%s-%s' % (self.relationship, tn)
        
    def __eq__(self, obj):
        if obj == None:
            return False
        
        assert isinstance(obj, Location)        
        assert isinstance(obj.thing, Thing)
        
        return self.relationship == obj.relationship and self.thing == obj.thing
        
class Condition(object): 
    """
    A pairing of Thing and either State or Location. We allow string placeholders
    for Things and Locations to support parameterization of Actions. 
    """
    
    def __init__(self, thing):
        assert isinstance(thing, Thing) or isinstance(thing, str)
        self.thing = thing
            
class StateCondition(Condition): 
    def __init__(self, thing, state):
        Condition.__init__(self, thing)
        assert isinstance(state, str)
        self.state = state

class LocationCondition(Condition):
    def __init__(self, thing, location):
        Condition.__init__(self, thing)
        assert isinstance(location, Location) or isinstance(location, str)
        self.location = location

    def __str__(self): 
        return '%s-%s' % (self.thing, self.location)

