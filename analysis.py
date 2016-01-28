__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
import hrr

# test inner product distributions object hits and misses
total_locations = 250
total_goals = 1000
total_things = 25000
total_actions = 10000
total_effects = 2500
n_queries = 100

def make_base_vectors(D):
    keys = hrr.CleanupMemory(['LOCATION', 'GOAL', 'EFFECT'], D)
    locations = hrr.CleanupMemory(['LOC' + str(l) for l in range(total_locations)], D)
    goals = hrr.CleanupMemory(['GOAL' + str(l) for l in range(total_goals)], D)
    effects = hrr.CleanupMemory(['EFFECT' + str(l) for l in range(total_effects)], D)
    return keys, locations, goals, effects

def make_thing(locations, goals, keys, location_indices, goal_indices):
    # make an object HRR vector with these associated locations and goals
    location_sum = np.sum(locations.vectors[:,location_indices], axis=1)
    goal_sum = np.sum(goals.vectors[:,goal_indices], axis=1)
    vector = hrr.bind(keys.get('LOCATION'), location_sum, do_normalize=False) + hrr.bind(keys.get('GOAL'), goal_sum, do_normalize=False)
    return vector

def make_affordance(things, effects, keys, thing_index, effect_indices):
    thing_sum = np.sum(things.vectors[:,thing_index], axis=1)
    effect_sum = np.sum(effects.vectors[:,effect_indices], axis=1)
    vector = hrr.bind(keys.get('THING'), thing_sum, do_normalize=False) + hrr.bind(keys.get('EFFECT'), effect_sum, do_normalize=False)
    return vector

def make_action(locations, effects, keys, location_indices, effect_indices):
    location_sum = np.sum(locations.vectors[:,location_indices], axis=1)
    effect_sum = np.sum(effects.vectors[:,effect_indices], axis=1)
    vector = hrr.bind(keys.get('LOCATION'), location_sum, do_normalize=False) + hrr.bind(keys.get('EFFECT'), effect_sum, do_normalize=False)
    return vector

def make_things(locations, goals, keys, mean_locations, mean_goals, D):
    n_locations = np.random.poisson(lam=mean_locations, size=total_things)
    n_goals = np.random.poisson(lam=mean_goals, size=total_things)

    thing_vectors = np.zeros((D,total_things))
    location_indices = []
    goal_indices = []
    for i in range(total_things):
        li = np.unique(np.random.randint(0, total_locations, size=n_locations[i]))
        gi = np.unique(np.random.randint(0, total_goals, size=n_goals[i]))
        thing_vectors[:,i] = make_thing(locations, goals, keys, li, gi)

        location_indices.append(li)
        goal_indices.append(gi)

    things = hrr.CleanupMemory(['THING' + str(l) for l in range(total_things)], D)
    things.vectors = thing_vectors
    things.location_indices = location_indices
    things.goal_indices = goal_indices
    return things
    # return thing_vectors, location_indices, goal_indices

def make_affordances(things, effects, keys, mean_effects, D):
    """
    Actions that are associated with a specific thing
    """
    n_things = 1
    n_effects = np.random.poisson(lam=mean_effects, size=total_effects)

    vectors = np.zeros((D, total_effects))
    thing_indices = []
    effect_indices = []
    for i in range(total_actions):
        ti = np.random.randint(0, total_things, size=n_things)
        ei = np.random.randint(0, total_effects, size=n_effects[i])
        vectors[:,i] = make_affordance(things, effects, keys, ti, ei)
        thing_indices.append(ti)
        effect_indices.append(ei)

    affordances = hrr.CleanupMemory(['AFF' + str(l) for l in range(total_actions)])
    affordances.vectors = vectors
    affordances.thing_indices = thing_indices
    affordances.effect_indices = effect_indices
    return affordances


def make_actions(locations, effects, keys, mean_locations, mean_effects, D):
    """
    Actions that are not associated with a specific thing, but possibly with a location
    """
    n_locations = np.random.poisson(lam=mean_locations, size=total_actions)
    n_effects = np.random.poisson(lam=mean_effects, size=total_actions)

    action_vectors = np.zeros((D,total_actions))
    location_indices = []
    effect_indices = []
    for i in range(total_actions):
        li = np.unique(np.random.randint(0, total_locations, size=n_locations[i]))
        ei = np.unique(np.random.randint(0, total_effects, size=n_effects[i]))
        action_vectors[:,i] = make_action(locations, effects, keys, li, ei)

        location_indices.append(li)
        effect_indices.append(ei)

    actions = hrr.CleanupMemory(['ACTION' + str(l) for l in range(total_things)], D)
    actions.vectors = action_vectors
    actions.location_indices = location_indices
    actions.effect_indices = effect_indices
    return actions

def thing_query(mean_locations, mean_goals, D):
    """
    Runs an experiment for statistics on goal&location->thing queries.
    """
    keys, locations, goals, effects = make_base_vectors(D)
    things = make_things(locations, goals, keys, mean_locations, mean_goals, D)

    matches = []
    partial_matches = []
    non_matches = []
    for i in range(n_queries):
        print(i)
        location_index = np.random.randint(0, total_locations)
        goal_index = np.random.randint(0, total_goals)
        query = make_thing(locations, goals, keys, [location_index], [goal_index])
        inner_products = np.dot(things.vectors.T, query)
        for j in range(total_things):
            loc_match = location_index in things.location_indices[j]
            goal_match = goal_index in things.goal_indices[j]
            if loc_match and goal_match:
                matches.append(inner_products[j])
            elif loc_match or goal_match:
                partial_matches.append(inner_products[j])
            else:
                non_matches.append(inner_products[j])

    return matches, partial_matches, non_matches

def action_query(mean_locations, mean_effects, D):
    """
    Runs an experiment for statistics on location&effect->action queries.
    """
    keys, locations, goals, effects = make_base_vectors(D)
    actions = make_actions(locations, effects, keys, mean_locations, mean_effects, D)

    matches = []
    partial_matches = []
    non_matches = []
    for i in range(n_queries):
        print(i)
        location_index = np.random.randint(0, total_locations)
        effect_index = np.random.randint(0, total_effects)
        query = make_action(locations, effects, keys, [location_index], [effect_index])
        inner_products = np.dot(actions.vectors.T, query)
        for j in range(total_actions):
            loc_match = location_index in actions.location_indices[j]
            effect_match = effect_index in actions.effect_indices[j]
            if loc_match and effect_match:
                matches.append(inner_products[j])
            elif loc_match or effect_match:
                partial_matches.append(inner_products[j])
            else:
                non_matches.append(inner_products[j])

    return matches, partial_matches, non_matches

def get_hits_per_query(matches):
    return float(len(matches)) / n_queries

def get_specificity(matches, partial_matches, non_matches):
    sorted = np.sort(matches)
    n = len(matches)
    if n == 0:
        return 0

    threshold_index = int(np.floor(n/10.))
    threshold = sorted[threshold_index]
    false_alarms = np.bincount(np.array(non_matches > threshold)) + np.bincount(np.array(partial_matches > threshold))
    if len(false_alarms) == 1:
        print('no false alarms')
        return 1
    else:
        return float(n) / (n + false_alarms[1])

# # example histogram ...
# mean_locations = 1
# mean_goals = mean_locations
# dimension = 500
# matches, partial_matches, non_matches = thing_query(mean_locations, mean_goals, dimension)
# bins = np.linspace(-0.7, 2.7, 35)
# plt.figure()
# plt.subplot(3,1,1)
# plt.hist(non_matches, bins)
# plt.ylabel('non-matches', fontsize=18)
# plt.subplot(3,1,2)
# plt.hist(partial_matches, bins)
# plt.ylabel('partial matches', fontsize=18)
# plt.subplot(3,1,3)
# plt.hist(matches, bins)
# plt.ylabel('matches', fontsize=18)
# plt.xlabel('inner product', fontsize=18)
# plt.show()

mean_locations_list = [1,2,3,4,5]
dimension_list = [250,500,1000]

all_hits_per_query = []
all_specificity = []
for dimension in dimension_list:
    hits = []
    specificity = []
    for mean_locations in mean_locations_list:
        # mean_goals = mean_locations
        # matches, partial_matches, non_matches = thing_query(mean_locations, mean_goals, dimension)
        mean_effects = 1
        matches, partial_matches, non_matches = action_query(mean_locations, mean_effects, dimension)
        hits.append(get_hits_per_query(matches))
        specificity.append(get_specificity(matches, partial_matches, non_matches))
    all_hits_per_query.append(hits)
    all_specificity.append(specificity)

all_hits_per_query = np.array(all_hits_per_query)
all_specificity = np.array(all_specificity)

import cPickle
file = open('./action-analysis.pkl', 'wb')
cPickle.dump((all_hits_per_query, all_specificity), file)
file.close()

print(all_hits_per_query)
print(all_specificity)

# plot hits and specificity ...
plt.figure()
plt.subplot(2,1,1)
plt.plot(mean_locations_list, all_hits_per_query.T)
plt.subplot(2,1,2)
plt.plot(mean_locations_list, all_specificity.T)
plt.show()
