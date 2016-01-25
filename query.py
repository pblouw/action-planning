__author__ = 'bptripp'

# This is a rough draft of a way to select actions to stack, based on effects
# and other contextual information. The basic idea is that there are lots of
# possibilities and we need something like an SQL WHERE clause for finding an
# action that is a good match for the current intention and the context
# (which is largely the location, e.g. kitchen).
#
# There are a couple of possible mappings that seem reasonable:
#   1) goal+location->things then things+intent->action
#   2) location+intent->action
#
# Below is a network that implements #1. I'm not very handy with SPA, so the
# scaling is all wrong (I've added a mess of constants to make it a little better)
# and the WTA competition doesn't work. Also, strangely, the noise is highly
# correlated across associative memory outputs.

import matplotlib.pyplot as plt
import nengo
import nengo.spa as spa

D=512
model = spa.SPA()
with model:
    vocab = spa.vocab.Vocabulary(D)
    # note: watch these aren't exactly the same, or add something else to make the object concepts unique
    vocab.add('KETTLE', vocab.parse('.707*(LOCATION*KITCHEN+LOCATION*STAFF_LOUNGE+LOCATION*HOME_DEPOT+GOAL*WATER_BOILED)'))
    vocab.add('TAP', vocab.parse('.707*(LOCATION*KITCHEN+LOCATION*STAFF_LOUNGE+GOAL*WATER_BOILED)'))
    vocab.add('DRILL', vocab.parse('.707*(LOCATION*GARAGE+LOCATION+HOME_DEPOT+GOAL*HOLE_MADE)'))

    vocab.add('BOIL_KETTLE', vocab.parse('.707*(OBJECT*KETTLE+EFFECTS*WATER_BOILED)'))
    vocab.add('UNPLUG_KETTLE', vocab.parse('.707*(OBJECT*KETTLE+EFFECTS*KETTLE_UNPLUGGED)'))
    vocab.add('PLUG_IN_KETTLE', vocab.parse('.707*(OBJECT*KETTLE+EFFECTS*KETTLE_PLUGGED_IN)'))
    vocab.add('FILL_KETTLE_FROM_TAP', vocab.parse('.707*(OBJECT*TAP+EFFECTS*WATER_IN_KETTLE)'))
    vocab.add('PUT_KETTLE_UNDER_TAP', vocab.parse('.707*(OBJECT*KETTLE+EFFECTS*KETTLE_UNDER_TAP)'))

    # Testing cleanups:
    # 1) goal+location->things
    # 2) things+intention->action

    model.goal = spa.Buffer(D, vocab=vocab)
    model.intention = spa.Buffer(D, vocab=vocab)
    model.location = spa.Buffer(D, vocab=vocab)

    model.things = spa.AssociativeMemory(vocab, wta_output=False, threshold=.5, n_neurons_per_ensemble=100) #this can't be called objects
    model.action = spa.AssociativeMemory(vocab, wta_output=True, n_neurons_per_ensemble=100)

    cortical_actions = spa.Actions(
        'things = (LOCATION*location+GOAL*goal)*.5',
        'action = (OBJECT*things+EFFECTS*intention)*.5'
    )
    model.cortical = spa.Cortical(cortical_actions)

    def set_goal(t):
        return 'WATER_BOILED'
    model.goal_inp = spa.Input(goal=set_goal)

    def set_location(t):
        if t < 0.5: return 'KITCHEN'
        else: return 'GARAGE'
    model.loc_inp = spa.Input(location=set_location)

    def set_intention(t):
        if t < .25: return 'KETTLE_UNPLUGGED'
        else: return 'KETTLE_UNDER_TAP'
    model.intention_inp = spa.Input(intention=set_intention)

    things_probe = nengo.Probe(model.things.output, synapse=0.03)
    action_probe = nengo.Probe(model.action.output, synapse=0.03)


def plot_results(sim):
    fig = plt.figure(figsize=(16,8))

    p1 = fig.add_subplot(3,1,1)
    p1.plot(sim.trange(), model.similarity(sim.data, things_probe))
    p1.set_ylabel('Things')

    p2 = fig.add_subplot(3,1,2)
    p2.plot(sim.trange(), model.similarity(sim.data, action_probe))
    p2.legend(model.get_output_vocab('goal').keys, fontsize='medium', bbox_to_anchor=(1, -0.15), ncol=6)
    p2.set_ylabel('Action')

    plt.show()

sim = nengo.Simulator(model)
sim.reset()
sim.run(1)
plot_results(sim)

