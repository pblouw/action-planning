'''
Created on Jun 12, 2015

@author: bptripp
'''
import world
import nodes
import nengo.spa as spa
import numpy as np

dim = 200;

class Water(world.Thing): 
    def __init__(self, the_world): 
        possible_states={'TEMPERATURE': ('COLD', 'WARM', 'HOT', 'BOILED')}
        world.Thing.__init__(self, the_world, 'WATER', ['WATER', 'LIQUID'], possible_states=possible_states)
        
class Kettle(world.Thing): 
    def __init__(self, the_world):
        world.Thing.__init__(self, the_world, 'KETTLE', 'KETTLE', possible_states={'PLUGGED': ('PLUGGED_IN', 'UNPLUGGED')})
        self.tap = None #set this after construction
        
    def plug_in_kettle(self): 
        self.set_state('PLUGGED_IN')
        return True
        
    def unplug_kettle(self): 
        self.set_state('UNPLUGGED')
        return True
        
    def put_kettle_under_tap(self): 
        if not self.get_state()['PLUGGED'] == 'UNPLUGGED': 
            return False            
        self.world.put(self, [world.Location('UNDER', self.tap)])
        return True        
        
    def boil_kettle(self): 
        if not self.get_state()['PLUGGED'] == 'PLUGGED_IN': 
            return False
        if not self.world.has_location(self.world.get('WATER'), world.Location('IN', self)):
            return False
        self.world.get('WATER').set_state('BOILED')
        return True        
    
class Tap(world.Thing):
    def __init__(self, the_world): 
        world.Thing.__init__(self, the_world, 'TAP', 'TAP', possible_states={'OFFON': ('OFF', 'ON')})
        self.kettle = None #set these after construction
        self.water = None        
        
    def fill_kettle_from_tap(self): 
        if not self.world.has_location(self.kettle, world.Location('UNDER', self)):
            return False
        self.world.put(self.water, [world.Location('IN', self.kettle)])
        return True

def get_world(): 
    w = world.World()
    water = Water(w)
    kettle = Kettle(w)
    tap = Tap(w)
    kettle.tap = tap 
    tap.kettle = kettle
    tap.water = water
    counter = world.Thing(w, 'COUNTER', ['COUNTER'])
    on_counter = world.Location('ON', counter)
    w.put(kettle, [on_counter])
    return w

    
def test_motor_system():
    world = get_world()

    D = 100
    vocab = spa.Vocabulary(D)

    keys = ['WATER', 'TAP', 'KETTLE', 'IN', 'UNDER', 'COLD', 'BOILED', 'PLUGGED_IN', 
                'UNPLUGGED', 'FILL_KETTLE_FROM_TAP', 'BOIL_KETTLE', 'PLUG_IN_KETTLE', 'UNPLUG_KETTLE', 'PUT_KETTLE_UNDER_TAP']
    
    for key in keys:         
        vocab.add(key, vocab.create_pointer())
        
    motor = nodes.MotorSystem(vocab, world)


    world.print_state()
    
    kettle = vocab.__getitem__('KETTLE')
    unplug = vocab.__getitem__('UNPLUG_KETTLE')
    
    for t in np.arange(.01, .5, .01):
        acting = motor.act(t, kettle, unplug)
        print('acting: ' + str(acting))
        
    world.print_state()
    
def test_visual_system():    
    world = get_world()
    
    D = 500
    vocab = spa.Vocabulary(D)

    keys = ['WATER', 'TAP', 'KETTLE', 'IN', 'UNDER', 'COLD', 'BOILED', 'PLUGGED_IN', 
                'UNPLUGGED', 'FILL_KETTLE_FROM_TAP', 'BOIL_KETTLE', 'PLUG_IN_KETTLE', 'UNPLUG_KETTLE', 'PUT_KETTLE_UNDER_TAP']
    
    for key in keys:         
        vocab.add(key, vocab.create_pointer())
    
    visual = nodes.VisualSystem(vocab, world)
    location = vocab.parse('ON*COUNTER')

    for t in np.arange(.01, .5, .01):
        state = visual.sense(t, location)
        print(np.linalg.norm(state.v))
    
test_motor_system()
test_visual_system()
