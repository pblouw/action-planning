
import numpy as np
import nengo 
import sys
from nengo import spa
import matplotlib.pyplot as plt


from collections import defaultdict

model = spa.SPA(label='Effect Cleanup Test')

D = 128

vocab = spa.Vocabulary(D)

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
    ('UNPLUG_KETTLE', 'HasEffect', 'KETTLE*UNPLUGGED')]

actions = []
effects = []

precons = defaultdict(list)

for r in relations:
    if r[1] == 'HasEffect':
        effect = r[2].replace('*', '_')
        action = r[0]
        
        effects.append(effect)
        actions.append(action)

        vocab.parse(action)
        vocab.parse(effect)

    elif r[1] == 'HasPrecondition':
        precon = r[2].replace('*', '_')
        precons[r[0]].append(precon)
        
precon_strings = []
for ps in precons.values():
    precon_strings.append(ps[0])

cycle_items = ['GET_ACTION','GET_PRECON','SET_GOAL']
cycle_vocab = spa.Vocabulary(D)

for item in cycle_items:
    cycle_vocab.parse(item)


effects = ['WATER_IN_KETTLE', 'KETTLE_UNDER_TAP', 'WATER_BOILED', 'KETTLE_PLUGGED_IN', 'KETTLE_UNPLUGGED']
actions = ['FILL_KETTLE_FROM_TAP', 'PUT_KETTLE_UNDER_TAP', 'BOIL_KETTLE', 'UNPLUG_KETTLE','PLUG_IN_KETTLE']
precon_strings = ['KETTLE_UNDER_TAP','KETTLE_UNPLUGGED','WATER_IN_KETTLE','HAS_CORD']



with model:

    model.plan = spa.Memory(dimensions=D, vocab=cycle_vocab, synapse=0.01, tau=0.075) 
    model.goal = spa.Memory(dimensions=D, vocab=vocab, synapse=0.01, tau=0.05)
    model.swap1 = spa.Memory(dimensions=D, vocab=vocab, synapse=0.01, tau=0.05)
    model.swap2 = spa.Memory(dimensions=D, vocab=vocab, synapse=0.01, tau=0.1)

    model.action_cleanup = spa.AssociativeMemory(input_vocab=vocab, output_vocab=vocab, 
                                                 input_keys=actions[:4], output_keys=precon_strings, threshold=0.2)
    model.effect_cleanup = spa.AssociativeMemory(input_vocab=vocab, output_vocab=vocab, 
                                                 input_keys=effects, output_keys=actions, threshold=0.2)

    actions = spa.Actions(
        'dot(plan, GET_ACTION)  -->  effect_cleanup=goal, plan=GET_PRECON',
        'dot(plan, GET_PRECON)  -->  action_cleanup=swap1, plan=SET_GOAL',
        'dot(plan, SET_GOAL)    -->  goal=swap2, plan=GET_ACTION')

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    def set_goal(t):
        if t < 0.1: return 'WATER_BOILED'
        else: return '0'

    def set_plan(t):
        if 0.1 < t < 0.15: return 'GET_ACTION'
        else: return '0'

    model.goal_inp = spa.Input(goal=set_goal)
    model.plan_inp = spa.Input(plan=set_plan)

    nengo.Connection(model.effect_cleanup.output, model.swap1.state.input)
    nengo.Connection(model.action_cleanup.output, model.swap2.state.input)

    effect_cleanup_probe = nengo.Probe(model.effect_cleanup.output, synapse=0.03)
    action_cleanup_probe = nengo.Probe(model.action_cleanup.output, synapse=0.03)

    goal_probe = nengo.Probe(model.goal.state.output, synapse=0.03)
    plan_probe = nengo.Probe(model.plan.state.output, synapse=0.03)
    swap1_probe = nengo.Probe(model.swap1.state.output, synapse=0.03)
    swap2_probe = nengo.Probe(model.swap2.state.output, synapse=0.03)

sim = nengo.Simulator(model)
sim.run(1)

fig = plt.figure(figsize=(12,8))

NUM_COLORS = 15

cm = plt.get_cmap('nipy_spectral')


p1 = fig.add_subplot(6,1,1)
p1.plot(sim.trange(), model.similarity(sim.data, goal_probe))
p1.set_ylabel('Goal')


p2 = fig.add_subplot(6,1,2)
p2.plot(sim.trange(), model.similarity(sim.data, plan_probe))
p2.legend(model.get_output_vocab('plan').keys, fontsize='x-small')
p2.set_ylabel('Plan')

p3 = fig.add_subplot(6,1,3)
p3.plot(sim.trange(), model.similarity(sim.data, swap1_probe))
p3.set_ylabel('Swap1')

p4 = fig.add_subplot(6,1,4)
p4.plot(sim.trange(), model.similarity(sim.data, swap2_probe))
p4.set_ylabel('Swap2')

p5 = fig.add_subplot(6,1,5)
p5.plot(sim.trange(), model.similarity(sim.data, effect_cleanup_probe))
p5.set_ylabel('Effect Clean')

p6 = fig.add_subplot(6,1,6)
p6.plot(sim.trange(), model.similarity(sim.data, action_cleanup_probe))
p6.legend(model.get_output_vocab('goal').keys, fontsize='x-small', bbox_to_anchor=(0.5, 0.15), ncol=3)
p6.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
p6.set_ylabel('Action Clean')

fig.subplots_adjust(hspace=0.2)
plt.show()


