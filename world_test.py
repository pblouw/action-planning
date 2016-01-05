import numpy as np
import nengo
import nengo.spa as spa
import world
import kitchen
import nodes

D = 16

def input_action():
    return 'BOIL_KETTLE'
    
def input_object():
    return 'KETTLE'

model = nengo.Network()
with model:
    vocab = spa.vocab.Vocabulary(D)
    vocab.parse('BOIL_KETTLE+PUT_KETTLE_UNDER_TAP')
    vocab.parse('KETTLE+TAP')
    vocab.parse('ACTING')
    
    model.action = spa.Buffer(D, vocab=vocab)
    model.object = spa.Buffer(D, vocab=vocab)
    model.acting = spa.Buffer(D, vocab=vocab)
    
    world = kitchen.get_kitchen()
    motor_fun = nodes.MotorSystem(vocab, world)
    model.motor = nengo.Node(motor_fun, size_in=2*D, size_out=1)
    
    nengo.Connection(model.action.state.output, model.motor[D:])
    nengo.Connection(model.object.state.output, model.motor[:D])
    nengo.Connection(model.motor, model.acting.state.input, transform=vocab.parse('ACTING').v[:,np.newaxis])
    
    # acting = nengo.Ensemble(n_neurons=100, dimensions=1)
    # nengo.Connection(model.motor, acting)


    
    
    
    
    
    
    
    
    
    
    
    
    
    