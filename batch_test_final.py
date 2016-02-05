import numpy as np
import nengo
import nengo.spa as spa
import world
import kitchen
import nodes

from nengo.networks import AssociativeMemory, Product 

trials = 20

goal_logger = np.zeros(trials)
time_logger = np.zeros(trials)
stack_logger = np.zeros((trials, 15))

np.save('2_goal_log.npy', goal_logger)
np.save('2_time_log.npy', time_logger)
np.save('2_stack_log.npy', stack_logger)


def permutations(word):
    if len(word)<=1:
        return [word]

    #get all permutations of length N-1
    perms=permutations(word[1:])
    char=word[0]
    result=[]
    #iterate over all permutations of length N-1
    for perm in perms:
        #insert the character into every possible location
        for i in range(len(perm)+1):
            result.append(perm[:i] + char + perm[i:])
    return result

items = permutations('ABCDEFG')
print len(items)

for _ in range(trials):
	world = kitchen.get_kitchen()
	# world.do('KETTLE','UNPLUG_KETTLE')
	# world.do('KETTLE','PUT_KETTLE_UNDER_TAP')
	# world.do('TAP', 'FILL_KETTLE_FROM_TAP')
	world.print_state()


	D = 256

	base_items= ['LOCATION','GOAL','GARAGE','HOLE_MADE','STAFF_LOUNGE','WATER_BOILED','KITCHEN','HOME_DEPOT',
	             'KETTLE_UNPLUGGED','KETTLE_PLUGGED_IN','WATER_IN_KETTLE','KETTLE_UNDER_TAP','EFFECTS','OBJECT']
	base_items += items

	objects = ['DRILL','KETTLE','TAP']
	actions = ['FILL_KETTLE_FROM_TAP', 'PUT_KETTLE_UNDER_TAP', 'BOIL_KETTLE', 'PLUG_IN_KETTLE','UNPLUG_KETTLE']

	base_vocab = spa.Vocabulary(D)

	for item in base_items:
	    base_vocab.parse(item)

	# Ensures shared representations across vocabs for objects and actions
	obj_vocab = base_vocab.create_subset(base_items)
	obj_id_vocab = base_vocab.create_subset(base_items)
	act_id_vocab = base_vocab.create_subset(base_items)
	act_vocab = base_vocab.create_subset(base_items)

	# Build structured semantic pointers for objects
	sp = base_vocab.parse('LOCATION*KITCHEN+LOCATION*STAFF_LOUNGE+LOCATION*HOME_DEPOT+GOAL*WATER_BOILED')
	sp.normalize()
	obj_vocab.add('KETTLE', sp)

	sp = base_vocab.parse('LOCATION*KITCHEN+LOCATION*STAFF_LOUNGE+GOAL*WATER_BOILED')
	sp.normalize()
	obj_vocab.add('TAP', sp)

	sp = base_vocab.parse('LOCATION*GARAGE+LOCATION*HOME_DEPOT+GOAL*HOLE_MADE')
	sp.normalize()
	obj_vocab.add('DRILL', sp)


	# Build structured semantic pointers for actions
	c = 1 # constant for weighting relative importance of structural components 
	sp = base_vocab['OBJECT'] * obj_id_vocab['KETTLE'] + c * (base_vocab['EFFECTS'] * base_vocab['WATER_BOILED'])
	sp.normalize()
	act_vocab.add('BOIL_KETTLE', sp)

	sp = base_vocab['OBJECT'] * obj_id_vocab['KETTLE'] + c * (base_vocab['EFFECTS'] * base_vocab['KETTLE_PLUGGED_IN'])
	sp.normalize()
	act_vocab.add('PLUG_IN_KETTLE', sp)

	sp = base_vocab['OBJECT'] * obj_id_vocab['KETTLE'] + c * (base_vocab['EFFECTS'] * base_vocab['KETTLE_UNPLUGGED'])
	sp.normalize()
	act_vocab.add('UNPLUG_KETTLE', sp)

	sp = base_vocab['OBJECT'] * obj_id_vocab['TAP'] + c * (base_vocab['EFFECTS'] * base_vocab['WATER_IN_KETTLE'])
	sp.normalize()
	act_vocab.add('FILL_KETTLE_FROM_TAP', sp)

	sp = base_vocab['OBJECT'] * obj_id_vocab['KETTLE'] + c * (base_vocab['EFFECTS'] * base_vocab['KETTLE_UNDER_TAP'])
	sp.normalize()
	act_vocab.add('PUT_KETTLE_UNDER_TAP', sp)


	# Create ID semantic pointers for objects and actions
	for obj in objects:
	    obj_id_vocab.parse(obj)

	for action in actions:
	    act_id_vocab.parse(action)

	effects = ['WATER_IN_KETTLE', 'KETTLE_UNDER_TAP', 'WATER_BOILED', 'KETTLE_PLUGGED_IN', 'KETTLE_UNPLUGGED']
	actions = ['FILL_KETTLE_FROM_TAP', 'PUT_KETTLE_UNDER_TAP', 'BOIL_KETTLE', 'PLUG_IN_KETTLE','UNPLUG_KETTLE']
	precons = ['KETTLE_UNDER_TAP','KETTLE_UNPLUGGED','WATER_IN_KETTLE','KETTLE_PLUGGED_IN','HAS_CORD']
	signals = ['PLAN','ACT','DONE']


	signal_vocab = spa.Vocabulary(D)

	for sig in signals:
	    signal_vocab.parse(sig)
	    
	act_id_vocab['INDEX'].make_unitary()

	goal_check_vocab = base_vocab.create_subset(effects)
	goal_check_vocab.add('DONE', signal_vocab['DONE'].v)

	inp_vecs = np.zeros((3, D))
	out_vecs = np.zeros((3, D))

	inp_vecs[0,:] = act_id_vocab['FILL_KETTLE_FROM_TAP'].v
	inp_vecs[1,:] = act_id_vocab['PUT_KETTLE_UNDER_TAP'].v
	inp_vecs[2,:] = act_id_vocab['BOIL_KETTLE'].v

	out_vecs[0,:] = base_vocab['KETTLE_UNDER_TAP'].v
	out_vecs[1,:] = base_vocab['KETTLE_UNPLUGGED'].v
	out_vecs[2,:] = 0.4* base_vocab['WATER_IN_KETTLE'].v + 0.6*base_vocab['KETTLE_PLUGGED_IN'].v


	motor_sys = nodes.MotorSystem(act_id_vocab, world, _)
	vision_sys = nodes.VisualSystem(act_id_vocab, world)
	check_sys = nodes.CheckSystem(goal_check_vocab, world, _)

	with spa.SPA() as model:

	    # For managing control signals
	    model.ctrl = spa.Memory(dimensions=D, tau=0.05) 
	    model.switch = spa.Memory(dimensions=D, vocab=signal_vocab, synapse=0.1, tau=0.15)

	    # Main state representations    
	    model.m_goal = spa.Memory(dimensions=D, vocab=base_vocab)
	    model.i_goal = spa.Memory(dimensions=D, vocab=base_vocab, synapse=0.005, tau=0.05)
	    model.action = spa.Memory(dimensions=D, vocab=act_id_vocab, synapse=0.005, tau=0.03)
	    model.effect = spa.Memory(dimensions=D, vocab=base_vocab, synapse=0.005, tau=0.05)
	    model.precon = spa.Memory(dimensions=D, vocab=base_vocab, synapse=0.005, tau=0.05)
	    model.location = spa.State(dimensions=D, vocab=base_vocab)
	    
	    # Associative Memories 
	    model.loc_to_object = spa.AssociativeMemory(input_vocab=obj_vocab, output_vocab=obj_id_vocab,
	                                                input_keys=objects, output_keys=objects,
	                                                wta_output=False, threshold=.5)
	    
	    model.obj_to_action = spa.AssociativeMemory(input_vocab=act_vocab, output_vocab=act_id_vocab, 
	                                                input_keys=actions, output_keys=actions,
	                                                wta_output=True, threshold=.9)
	    
	    model.action_to_effect = spa.AssociativeMemory(input_vocab=act_id_vocab, output_vocab=base_vocab,
	                                                   input_keys=actions, output_keys=effects, wta_output=True)
	   
	    model.cleanup_action = spa.AssociativeMemory(act_id_vocab, threshold=0.15, wta_output=True)
	    model.action_to_precon = AssociativeMemory(input_vectors=inp_vecs, output_vectors=out_vecs)  
	    
	    
	    nengo.Connection(model.location.output, model.loc_to_object.input, 
	                     transform=base_vocab['LOCATION'].get_convolution_matrix())
	    nengo.Connection(model.m_goal.state.output, model.loc_to_object.input, 
	                     transform=base_vocab['GOAL'].get_convolution_matrix())
	    
	    nengo.Connection(model.loc_to_object.output, model.obj_to_action.input, 
	                     transform=base_vocab['OBJECT'].get_convolution_matrix())
	    nengo.Connection(model.i_goal.state.output, model.obj_to_action.input, 
	                     transform=base_vocab['EFFECTS'].get_convolution_matrix())
	    
	    # Multiple Precons Test
	    model.get_precon = spa.State(dimensions=D, vocab=act_id_vocab)
	    nengo.Connection(model.get_precon.output, model.action_to_precon.input)
	    nengo.Connection(model.action_to_precon.output, model.precon.state.input)
	    
	    # Stack implementation
	    model.push = spa.Memory(dimensions=D, vocab=act_id_vocab, synapse=0.005, tau=0.05)
	    model.stack = spa.Memory(dimensions=D, vocab=act_id_vocab, synapse=0.005, tau=0.5)
	    model.premotor = spa.Memory(dimensions=D, vocab=act_id_vocab, synapse=0.005, tau=0.03)
	    
	    model.sig = spa.Memory(dimensions=D, vocab=signal_vocab, synapse=0.1, tau=0.65)
	     
	    # For switching between planning and acting
	    model.const = nengo.Node(1, size_out=1)
	    model.gate = spa.Memory(dimensions=1, neurons_per_dimension=250)
	    model.prod = Product(n_neurons=200, dimensions=1) 

	    nengo.Connection(model.const, model.prod.A)
	    nengo.Connection(model.gate.state.output, model.prod.B)
	    nengo.Connection(model.prod.output, model.sig.state.input, transform=0.4*signal_vocab['PLAN'].v.reshape(D,1))
	    
	    # For checking goal completion
	    model.check = nengo.Node(check_sys, size_in=D, size_out=1)
	    nengo.Connection(model.m_goal.state.output, model.check[:])
	    nengo.Connection(model.check, model.switch.state.input, transform=signal_vocab['DONE'].v.reshape(D,1))
	    
	    bg_actions = spa.Actions(
	        'dot(ctrl, PLAN)       -->  i_goal=m_goal, ctrl=GET_ACTION',
	        'dot(ctrl, GET_ACTION) -->  action=obj_to_action, ctrl=GET_PRECON',
	        'dot(ctrl, GET_PRECON) -->  get_precon=2*action, push=1.5*(stack*INDEX+action), ctrl=SET_GOAL',
	        'dot(ctrl, SET_GOAL)   -->  i_goal=1.5*i_goal+1.2*precon-effect, stack=push, ctrl=GET_ACTION',
	        
	        'dot(switch, ACT)      -->  ctrl=TOP_STACK, gate=1, switch=-1*ACT',
	        'dot(ctrl, TOP_STACK)  -->  cleanup_action=stack, ctrl=POP_STACK',
	        'dot(ctrl, POP_STACK)  -->  push=(stack-premotor)*~INDEX, ctrl=SET_STACK',
	        'dot(ctrl, SET_STACK)  -->  stack=10*push, ctrl=TOP_STACK',
	        'dot(sig, PLAN)        -->  ctrl=PLAN, gate=-0.45',
	        'dot(switch, DONE)     -->  ctrl=DONE',
	        '0.5                   -->  ')

	    ct_actions = spa.Actions(
	        'action_to_effect=action',
	        'effect=action_to_effect',
	        'premotor=cleanup_action')
	    
	    model.bg = spa.BasalGanglia(bg_actions)
	    model.ct = spa.Cortical(ct_actions)
	    model.thal = spa.Thalamus(model.bg)

	    def set_goal(t):
	        return 'WATER_BOILED'

	    def set_location(t):
	        return 'KITCHEN'
	    
	    def set_plan(t):
	        if 0.03 < t < 0.07: return 'PLAN'
	        else: return '0'

	    model.start = spa.Input(m_goal=set_goal, ctrl=set_plan, location=set_location)

	    model.motor = nengo.Node(motor_sys, size_in=D, size_out=1)
	    model.sense = nengo.Node(vision_sys, size_in=D, size_out=1)

	    # Node Connections
	    nengo.Connection(model.push.state.output, model.sense[:])
	    nengo.Connection(model.premotor.state.output, model.motor[:])
	    
	    nengo.Connection(model.sense, model.switch.state.input, transform=0.6*signal_vocab['ACT'].v.reshape(D,1))
	      

	sim = nengo.Simulator(model, seed=np.random.randint(1))
	sim.run(4)

	print 'AFTER: '
	world.print_state()

	print ''
	print 'On Trial:'
	print _
	print ''




