import numpy as np
import nengo
import nengo.spa as spa
import world
import kitchen
import nodes

D = 192

def input_state(t):
    return 'STARTING_PLAN'

def input_things(t): 
    return 'KETTLE+TAP'
    
def input_goal(t): 
    return 'WATER_BOILED'

model = spa.SPA()
with model:
    vocab = spa.vocab.Vocabulary(D)

    
    # actions ... 
    vocab.add('BOIL_KETTLE', vocab.parse('OBJECT*KETTLE+EFFECTS*WATER_BOILED+PRECONDITIONS*WATER_IN_KETTLE+KETTLE_PLUGGED_IN'))
    vocab.add('UNPLUG_KETTLE', vocab.parse('OBJECT*KETTLE+EFFECTS*KETTLE_UNPLUGGED'))
    vocab.add('PLUG_IN_KETTLE', vocab.parse('OBJECT*KETTLE+EFFECTS*KETTLE_PLUGGED_IN'))
    vocab.add('FILL_KETTLE_FROM_TAP', vocab.parse('OBJECT*TAP+EFFECTS*WATER_IN_KETTLE+PRECONDITIONS*KETTLE_UNDER_TAP'))
    vocab.add('PUT_KETTLE_UNDER_TAP', vocab.parse('OBJECT*KETTLE+EFFECTS*KETTLE_UNDER_TAP+PRECONDITIONS*KETTLE_UNPLUGGED'))

    model.ultimate_goal = spa.Buffer(D, vocab=vocab)
    model.things = spa.Buffer(D, vocab=vocab)
    # model.state = spa.Memory(D, vocab=vocab)
    model.state = spa.Buffer(D, vocab=vocab)
    model.immediate_goal = spa.AssociativeMemory(vocab, wta_output=True, n_neurons_per_ensemble=50)
    model.template = spa.AssociativeMemory(vocab, wta_output=True, n_neurons_per_ensemble=50)
    # model.precond = spa.Buffer(D, vocab=vocab)
    # model.effects = spa.Buffer(D, vocab=vocab)
    model.precond = spa.AssociativeMemory(vocab, wta_output=True, n_neurons_per_ensemble=100)
    model.effects = spa.AssociativeMemory(vocab, wta_output=True, n_neurons_per_ensemble=50)
    model.action = spa.Buffer(D, vocab=vocab)
    model.thing = spa.AssociativeMemory(vocab, wta_output=True, n_neurons_per_ensemble=50)
    model.acting = spa.Buffer(D, vocab=vocab)

    
    model.input = spa.Input(things=input_things, ultimate_goal=input_goal)
    
    cortical_actions = spa.Actions(
        'template = immediate_goal*EFFECTS+things*OBJECT', 
        'precond = .75*template*~PRECONDITIONS',
        'effects = .75*template*~EFFECTS', 
        'thing = action*~OBJECT'
    )
    
    vocab.parse('STARTING_PLAN+STARTING_ACTION')
    
    
    states = 'WATER_BOILED+WATER_IN_KETTLE+KETTLE_UNPLUGGED+KETTLE_PLUGGED_IN+KETTLE_UNDER_TAP'
    bg_actions = spa.Actions(
        '1.5*dot(state, STARTING_PLAN) --> immediate_goal=ultimate_goal', 
        '.8*dot(precond, %s) --> immediate_goal=immediate_goal+precond-effects' % states, 
        '.5 --> state=STARTING_ACTION', 
        '1.2*dot(state, STARTING_ACTION) --> action=template, state=STARTING_ACTION', 
    )
    
    model.cortical = spa.Cortical(cortical_actions)
    model.bg = spa.BasalGanglia(bg_actions)
    model.thal = spa.Thalamus(model.bg)
    
    world = kitchen.get_kitchen()
    motor_fun = nodes.MotorSystem(vocab, world)
    model.motor = nengo.Node(motor_fun, size_in=2*D, size_out=1)
    nengo.Connection(model.action.state.output, model.motor[D:])
    nengo.Connection(model.thing.output, model.motor[:D])
    nengo.Connection(model.motor, model.acting.state.input, transform=vocab.parse('ACTING').v[:,np.newaxis])
    
    
    
    
    
    
    
    