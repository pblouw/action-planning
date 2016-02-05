import numpy as np
import nengo
import nengo.spa as spa
import world
import kitchen
import nodes
import matplotlib.pyplot as plt

D = 256


for _ in range(5):

    test_world = kitchen.get_kitchen()
    # test_world.do('KETTLE','UNPLUG_KETTLE')
    print 'BEFORE:'
    test_world.print_state()

    effects = ['WATER_IN_KETTLE', 'KETTLE_UNDER_TAP', 'WATER_BOILED', 'KETTLE_PLUGGED_IN', 'KETTLE_UNPLUGGED']
    actions = ['FILL_KETTLE_FROM_TAP', 'PUT_KETTLE_UNDER_TAP', 'BOIL_KETTLE', 'PLUG_IN_KETTLE','UNPLUG_KETTLE']
    precons = ['KETTLE_UNDER_TAP','KETTLE_UNPLUGGED','WATER_IN_KETTLE','KETTLE_PLUGGED_IN','HAS_CORD']
    signals = ['PLAN','EXECUTE','NULL','DONE']

    effect_vocab = spa.Vocabulary(D)
    action_vocab = spa.Vocabulary(D)
    precon_vocab = spa.Vocabulary(D)
    signal_vocab = spa.Vocabulary(D)
    count_vocab = spa.Vocabulary(D)

    for effect in effects:
        effect_vocab.parse(effect)
        
    for action in actions:
        action_vocab.parse(action)

    for precon in precons:
        precon_vocab.parse(precon)  

    for sig in signals:
        signal_vocab.parse(sig)
        
    action_vocab['INDEX'].make_unitary()

    goal_check_vocab = effect_vocab.create_subset(effect_vocab.keys)
    goal_check_vocab.add('DONE', signal_vocab['DONE'].v)

    mapping = np.zeros((D, len(action_vocab.keys)))

    # Check for no outstanding preconditions.
    mapping[:,0] = 0.4*signal_vocab['NULL'].v
    mapping[:,1] = 0.4*signal_vocab['NULL'].v
    mapping[:,2] = 0.4*signal_vocab['NULL'].v
    mapping[:,3] = 0.4*signal_vocab['NULL'].v
    mapping[:,4] = 0.95*signal_vocab['EXECUTE'].v
    mapping[:,5] = 0.4*signal_vocab['NULL'].v

    inp_vecs = np.zeros((5, D))
    out_vecs = np.zeros((5, D))

    inp_vecs[0,:] = action_vocab['FILL_KETTLE_FROM_TAP'].v
    inp_vecs[1,:] = action_vocab['PUT_KETTLE_UNDER_TAP'].v
    inp_vecs[2,:] = action_vocab['BOIL_KETTLE'].v

    out_vecs[0,:] = effect_vocab['KETTLE_UNDER_TAP'].v
    out_vecs[1,:] = effect_vocab['KETTLE_UNPLUGGED'].v
    out_vecs[2,:] = effect_vocab['WATER_IN_KETTLE'].v + 1.2*effect_vocab['KETTLE_PLUGGED_IN'].v


    from nengo.networks import AssociativeMemory, Product 

    motor_sys = nodes.MotorSystem(action_vocab, test_world)
    vision_sys = nodes.VisualSystem(action_vocab, test_world)
    check_sys = nodes.CheckSystem(goal_check_vocab, test_world)

    with spa.SPA(label='Planner Test') as model:

        # For managing control signals
        model.ctrl = spa.Memory(dimensions=D, tau=0.05) 
        model.switch = spa.Memory(dimensions=D, vocab=signal_vocab, synapse=0.1, tau=0.15)

        # Main state representations    
        model.m_goal = spa.Memory(dimensions=D, vocab=effect_vocab)
        model.i_goal = spa.Memory(dimensions=D, vocab=effect_vocab, synapse=0.005, tau=0.15)
        model.action = spa.Memory(dimensions=D, vocab=action_vocab, synapse=0.005, tau=0.05)
        model.effect = spa.Memory(dimensions=D, vocab=effect_vocab, synapse=0.05, tau=0.05)
        model.precon = spa.Memory(dimensions=D, vocab=effect_vocab, synapse=0.005, tau=0.05)

        # Associative Memories 
        model.goal_to_action = spa.AssociativeMemory(input_vocab=effect_vocab, output_vocab=action_vocab, 
                                                     input_keys=effects, output_keys=actions, wta_output=True)
        
        model.action_to_effect = spa.AssociativeMemory(input_vocab=action_vocab, output_vocab=effect_vocab,
                                                       input_keys=actions, output_keys=effects, wta_output=True)
        
        model.action_to_signal = AssociativeMemory(input_vectors=action_vocab.vectors, output_vectors=mapping.T)
        model.action_to_precon = AssociativeMemory(input_vectors=inp_vecs, output_vectors=out_vecs)  
        
        # Multiple Precons Test
        model.router = spa.State(dimensions=D, vocab=action_vocab)
        nengo.Connection(model.router.output, model.action_to_precon.input)
        nengo.Connection(model.action_to_precon.output, model.precon.state.input)
        
        # Stack implementation
        model.push = spa.Memory(dimensions=D, vocab=action_vocab, synapse=0.005, tau=0.05)
        model.stack = spa.Memory(dimensions=D, vocab=action_vocab, synapse=0.005, tau=0.5)
        model.top = spa.Memory(dimensions=D, vocab=action_vocab, synapse=0.005, tau=0.03)
        
        model.sig = spa.Memory(dimensions=D, vocab=signal_vocab, synapse=0.1, tau=0.65)
        model.clean_action = spa.AssociativeMemory(action_vocab, threshold=0.15, wta_output=True)
        
        # For switching between planning and acting
        model.const = nengo.Node(1, size_out=1)
        model.gate = spa.Memory(dimensions=1, neurons_per_dimension=250)
        model.prod = Product(n_neurons=200, dimensions=1) 

        nengo.Connection(model.const, model.prod.A)
        nengo.Connection(model.gate.state.output, model.prod.B)
        nengo.Connection(model.prod.output, model.sig.state.input, transform=0.35*signal_vocab['PLAN'].v.reshape(D,1))
        
        # For checking goal completion
        model.check = nengo.Node(check_sys, size_in=D, size_out=1)
        nengo.Connection(model.m_goal.state.output, model.check[:])
        nengo.Connection(model.check, model.switch.state.input, transform=signal_vocab['DONE'].v.reshape(D,1))
        
        bg_actions = spa.Actions(
            'dot(ctrl, PLAN)       -->  i_goal=m_goal, ctrl=GET_ACTION',
            'dot(ctrl, GET_ACTION) -->  goal_to_action=1.5*i_goal, ctrl=GET_PRECON',
            'dot(ctrl, GET_PRECON) -->  router=2*action, push=1.5*(stack*INDEX+action), ctrl=SET_GOAL',
            'dot(ctrl, SET_GOAL)   -->  i_goal=1.5*i_goal+1.2*precon-effect, stack=push, ctrl=GET_ACTION',
            
            'dot(switch, EXECUTE)  -->  ctrl=TOP_STACK, gate=1, switch=-1*EXECUTE',
            'dot(ctrl, TOP_STACK)  -->  clean_action=stack, ctrl=POP_STACK',
            'dot(ctrl, POP_STACK)  -->  push=(stack-top)*~INDEX, ctrl=SET_STACK',
            'dot(ctrl, SET_STACK)  -->  stack=10*push, ctrl=TOP_STACK',
            'dot(sig, PLAN)        -->  ctrl=PLAN, gate=-0.45',
            'dot(switch, DONE)     -->  ctrl=DONE',
            '0.5                   -->  ')

        ct_actions = spa.Actions(
            'action_to_effect=action',
            'effect=action_to_effect',
            'action=goal_to_action',
            'top=clean_action')
        
        model.bg = spa.BasalGanglia(bg_actions)
        model.ct = spa.Cortical(ct_actions)
        model.thal = spa.Thalamus(model.bg)

        def set_goal(t):
            return 'WATER_BOILED'

        def set_plan(t):
            if 0.03 < t < 0.07: return 'PLAN'
            else: return '0'

        model.start = spa.Input(m_goal=set_goal, ctrl=set_plan)

        model.motor = nengo.Node(motor_sys, size_in=D, size_out=1)
        model.sense = nengo.Node(vision_sys, size_in=D, size_out=1)

        # Node Connections
        nengo.Connection(model.push.state.output, model.sense[:])
        nengo.Connection(model.top.state.output, model.motor[:])
        nengo.Connection(model.sense, model.switch.state.input, transform=0.6*signal_vocab['EXECUTE'].v.reshape(D,1))
        
        # Compare and AssocMem Connections
        nengo.Connection(model.action.state.output, model.action_to_signal.input)
        nengo.Connection(model.action_to_signal.output, model.switch.state.input, transform=1)
        

    sim = nengo.Simulator(model, seed=np.random.randint(1))
    sim.run(2.5)
    
    print 'AFTER: '
    test_world.print_state()

    print ''
    print ''


