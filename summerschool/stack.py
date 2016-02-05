import numpy as np
import nengo 
from nengo import spa
import matplotlib.pyplot as plt

model = spa.SPA(label='Stack Test')

D = 256

vocab = spa.Vocabulary(D)

symbols = ['PUSH','POP','TOP','ACTION_1','ACTION_2','ACTION_3','INDEX']
for symbol in symbols:
    vocab.parse(symbol)

vocab['INDEX'].make_unitary()

with model:

    model.prompt = spa.Buffer(dimensions=D, vocab=vocab)
    model.state = spa.Buffer(dimensions=D, vocab=vocab)
    model.stack = spa.Memory(dimensions=D, subdimensions=8, neurons_per_dimension=100, vocab=vocab,synapse=0.01, tau=0.2)
    model.top = spa.Memory(dimensions=D, subdimensions=8, neurons_per_dimension=100, vocab=vocab, synapse=0.01, tau=0.05)
    model.clean = spa.AssociativeMemory(vocab, threshold=0.15, n_neurons_per_ensemble=50)

    actions = spa.Actions(
        'dot(prompt, PUSH) --> stack = stack*INDEX+state',
        'dot(prompt, TOP)  --> clean = stack',
        'dot(prompt, POP)  --> stack = (stack-top)*~INDEX')

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    nengo.Connection(model.clean.output, model.top.state.input)
    nengo.Connection(model.top.state.output, model.state.state.input)

    def set_state(t):
        if t < 0.2: return 'ACTION_1'
        if 0.2 < t < 0.4: return 'ACTION_2'
        if 0.4 < t < 0.6: return 'ACTION_3'
        else: return '0'

    def set_prompt(t):
        if 0.05 < t < 0.15: return 'PUSH'
        if 0.25 < t < 0.35: return 'PUSH'
        if 0.45 < t < 0.55: return 'PUSH'
        if 0.65 < t < 0.75: return 'TOP'
        if 0.75 < t < 0.95: return 'POP'
        if 0.95 < t < 1.05: return 'TOP'
        if 1.05 < t < 1.25: return 'POP'
        if 1.25 < t < 1.35: return 'TOP'
        else: return '0'

    model.state_input = spa.Input(state=set_state)
    model.prompt_input = spa.Input(prompt=set_prompt)

    prompt_probe = nengo.Probe(model.prompt.state.output, synapse=0.03)
    state_probe = nengo.Probe(model.state.state.output, synapse=0.03)
    stack_probe = nengo.Probe(model.stack.state.output, synapse=0.03)
    top_probe = nengo.Probe(model.top.state.output, synapse=0.03)
    utility_probe = nengo.Probe(model.bg.input, synapse=0.01)
    action_probe = nengo.Probe(model.thal.actions.output, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(1.5)

fig = plt.figure(figsize=(12,8))

p1 = fig.add_subplot(6,1,1)
p1.plot(sim.trange(), model.similarity(sim.data, prompt_probe))
p1.legend(model.get_output_vocab('prompt').keys, fontsize='x-small')
p1.set_ylabel('Prompt')

p2 = fig.add_subplot(6,1,2)
p2.plot(sim.trange(), model.similarity(sim.data, state_probe))
p2.legend(model.get_output_vocab('state').keys, fontsize='x-small')
p2.set_ylabel('State')

p3 = fig.add_subplot(6,1,3)
p3.plot(sim.trange(), model.similarity(sim.data, top_probe))
p3.legend(model.get_output_vocab('top').keys, fontsize='x-small')
p3.set_ylabel('Top')

p4 = fig.add_subplot(6,1,4)
p4.plot(sim.trange(), model.similarity(sim.data, stack_probe))
p4.legend(model.get_output_vocab('stack').keys, fontsize='x-small')
p4.set_ylabel('Stack')

p5 = fig.add_subplot(6,1,5)
p5.plot(sim.trange(), sim.data[utility_probe])
p5_legend_txt = [a.condition for a in model.bg.actions.actions]
p5.legend(p5_legend_txt, fontsize='x-small')
p5.set_ylabel('Utility')

p6 = fig.add_subplot(6,1,6)
p6.plot(sim.trange(), sim.data[action_probe])
p6_legend_txt = [a.condition for a in model.bg.actions.actions]
p6.legend(p6_legend_txt, fontsize='x-small')
p6.set_ylabel('Actions')

fig.subplots_adjust(hspace=0.2)
plt.show()