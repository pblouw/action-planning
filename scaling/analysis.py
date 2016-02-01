__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
import hrr
import cPickle


# Testing inner product distributions object hits and misses.

total_locations = 250
total_goals = 1000
total_things = 25000
total_actions = 10000
total_effects = 2500
n_queries = 100

def make_base_vectors(D):
    keys = hrr.CleanupMemory(['LOCATION', 'GOAL', 'EFFECT', 'THING'], D)
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
    #normalize each thing vector for use as affordance component
    v = things.vectors[:,thing_index].copy()
    for i in range(v.shape[1]):
        v[:,i] = hrr.normalize(v[:,i])

    # thing_sum = np.sum(things.vectors[:,thing_index], axis=1)
    thing_sum = np.sum(v, axis=1)
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

def make_affordances(things, effects, keys, D):
    """
    Actions that are associated with a specific thing
    """
    n_things = 1
    n_effects = 1

    vectors = np.zeros((D, total_actions))
    thing_indices = []
    effect_indices = []
    for i in range(total_actions):
        ti = np.random.randint(0, total_things, size=n_things)
        ei = np.random.randint(0, total_effects, size=n_effects)
        vectors[:,i] = make_affordance(things, effects, keys, ti, ei)
        thing_indices.append(ti)
        effect_indices.append(ei)

    affordances = hrr.CleanupMemory(['AFF' + str(l) for l in range(total_actions)], D)
    affordances.vectors = vectors
    affordances.thing_indices = thing_indices
    affordances.effect_indices = effect_indices
    return affordances


def make_random_cleanup(n, prefix, keys, labels, cleanups, counts, D):
    """
    n - number of concepts in cleanup
    prefix - prefix for concept names (to be appended with a number)
    keys - cleanup of data types (LOCATION, GOAL, etc.)
    cleanups - tuple of cleanup memories from which parts of the created concepts will be drawn
    labels - tuple of labels (e.g. 'LOCATION') of parts
    counts - tuple of counts (each length n) of summed components from each cleanup
    """
    assert len(labels) == len(cleanups)
    assert len(labels) == len(counts)

    cleanup = hrr.CleanupMemory([prefix + str(l) for l in range(n)], D)
    indices = []
    for i in range(n):
        indices_i = []
        vector = np.zeros(D)
        for j in range(len(labels)):
            indices_ij = np.random.randint(0, len(cleanups[j].concepts), size=counts[j][i])
            s = np.sum(cleanups[j].vectors[:,indices_ij], axis=1)
            vector = vector + hrr.bind(keys.get(labels[j]), s, do_normalize=False)
            indices_i.append(indices_ij)
        cleanup.replace(cleanup.concepts[i], vector)
        indices.append(indices_i)
    cleanup.indices = indices
    return cleanup



def make_actions(locations, effects, keys, mean_locations, D):
    """
    Actions that are not associated with a specific thing, but possibly with a location
    """
    n_locations = np.random.poisson(lam=mean_locations, size=total_actions)
    # n_effects = np.random.poisson(lam=mean_effects, size=total_actions)
    n_effects = 1

    action_vectors = np.zeros((D,total_actions))
    location_indices = []
    effect_indices = []
    for i in range(total_actions):
        li = np.unique(np.random.randint(0, total_locations, size=n_locations[i]))
        ei = np.unique(np.random.randint(0, total_effects, size=n_effects))
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

def affordance_query(mean_locations, mean_goals, D):
    """
    Runs an experiment for statistics on goal&location->things; things&effect->action queries.
    """
    keys, locations, goals, effects = make_base_vectors(D)
    things = make_things(locations, goals, keys, mean_locations, mean_goals, D)
    affordances = make_affordances(things, effects, keys, D)

    thing_matches = []
    thing_partial_matches = []
    thing_non_matches = []
    affordance_matches = []
    affordance_partial_matches = []
    affordance_non_matches = []
    for i in range(n_queries):
        location_index = np.random.randint(0, total_locations)
        goal_index = np.random.randint(0, total_goals)
        query = make_thing(locations, goals, keys, [location_index], [goal_index])
        inner_products = np.dot(things.vectors.T, query)
        matching_thing_indices = []
        for j in range(total_things):
            loc_match = location_index in things.location_indices[j]
            goal_match = goal_index in things.goal_indices[j]
            if loc_match and goal_match:
                thing_matches.append(inner_products[j])
                matching_thing_indices.append(j)
            elif loc_match or goal_match:
                thing_partial_matches.append(inner_products[j])
            else:
                thing_non_matches.append(inner_products[j])

        #choose an affordance of a thing in our list, and query for it by thing and effect
        for thing_index in matching_thing_indices:
            effect_index = None
            for affordance_index in range(total_actions):
                if thing_index in affordances.thing_indices[affordance_index]:
                    if len(affordances.effect_indices[affordance_index]) > 0:
                        effect_index = affordances.effect_indices[affordance_index][0]
                        break
            if effect_index is not None:
                #TODO: use result of thing query and check results against correct thing and effect indices
                query = make_affordance(things, effects, keys, matching_thing_indices, [effect_index])

                inner_products = np.dot(affordances.vectors.T, query)
                for j in range(total_actions):
                    thing_match = affordances.thing_indices[j] in matching_thing_indices
                    effect_match = effect_index in affordances.effect_indices[j]
                    if thing_match and effect_match:
                        affordance_matches.append(inner_products[j])
                    elif thing_match or effect_match:
                        affordance_partial_matches.append(inner_products[j])
                    else:
                        affordance_non_matches.append(inner_products[j])
                break

    # plt.figure()
    # plt.subplot(311)
    # plt.hist(affordance_non_matches)
    # plt.subplot(312)
    # plt.hist(affordance_partial_matches)
    # plt.subplot(313)
    # plt.hist(affordance_matches)
    # plt.show()

    return thing_matches, thing_partial_matches, thing_non_matches, affordance_matches, affordance_partial_matches, affordance_non_matches


def action_query(mean_locations, D):
    """
    Runs an experiment for statistics on location&effect->action queries.
    """
    keys, locations, goals, effects = make_base_vectors(D)
    actions = make_actions(locations, effects, keys, mean_locations, D)

    matches = []
    partial_matches = []
    non_matches = []
    for i in range(n_queries):
        # To account for location-effect correlations we draw a random action and choose its
        # first location and effect. If it has zero locations or effects we draw another one.
        found = False
        while not found:
            action_index = np.random.randint(0, total_actions)
            found = len(actions.location_indices[action_index]) > 0 and len(actions.effect_indices[action_index]) > 0
        location_index = actions.location_indices[action_index][0]
        effect_index = actions.effect_indices[action_index][0]
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

def get_precision(matches, partial_matches, non_matches):
    sorted = np.sort(matches)
    n = len(matches)
    if n == 0:
        return None

    threshold_index = int(np.floor(n/10.))
    threshold = sorted[threshold_index]
    false_alarms = np.bincount(np.array(non_matches > threshold), minlength=2) + np.bincount(np.array(partial_matches > threshold), minlength=2)
    print('threshold ' + str(threshold_index) + ' ' + str(threshold) + ' false alarms' + str(false_alarms))
    return float(n) / (n + false_alarms[1])

# # example histogram ...
# np.random.seed(2)
# mean_locations = 2
# mean_goals = mean_locations
# dimension = 500
# matches, partial_matches, non_matches = thing_query(mean_locations, mean_goals, dimension)
# bins = np.linspace(-0.7, 2.7, 35)
# plt.figure()
# p = plt.subplot(3,1,1)
# plt.hist(non_matches, bins, color='gray')
# p.tick_params(axis='x', which='both', labelbottom='off')
# p.tick_params(axis='y', which='both', labelsize=14)
# plt.ylabel('non-matches', fontsize=18)
# p = plt.subplot(3,1,2)
# plt.hist(partial_matches, bins, color='gray')
# p.tick_params(axis='x', which='both', labelbottom='off')
# p.tick_params(axis='y', which='both', labelsize=14)
# plt.ylabel('partial matches', fontsize=18)
# p = plt.subplot(3,1,3)
# plt.hist(matches, bins, color='gray')
# p.tick_params(axis='both', which='both', labelsize=14)
# plt.ylabel('matches', fontsize=18)
# plt.xlabel('inner product', fontsize=18)
# plt.show()

mean_locations_list = [1,2,3,4,5]
dimension_list = [250,500,1000]

do_affordance = True

all_hits_per_query = []
all_precision = []
all_precision_affordance = []
for dimension in dimension_list:
    hits = []
    precision = []
    precision_affordance = []
    for mean_locations in mean_locations_list:
        mean_goals = mean_locations
        # matches, partial_matches, non_matches = thing_query(mean_locations, mean_goals, dimension)
        if do_affordance:
            matches, partial_matches, non_matches, affordance_matches, affordance_partial_matches, affordance_non_matches = affordance_query(mean_locations, mean_goals, dimension)
            precision_affordance.append(get_precision(affordance_matches, affordance_partial_matches, affordance_non_matches))
        else:
            matches, partial_matches, non_matches = action_query(mean_locations, dimension)
        hits.append(get_hits_per_query(matches))
        precision.append(get_precision(matches, partial_matches, non_matches))
    all_hits_per_query.append(hits)
    all_precision.append(precision)
    all_precision_affordance.append(precision_affordance)

all_hits_per_query = np.array(all_hits_per_query)
all_precision = np.array(all_precision)
all_precision_affordance = np.array(all_precision_affordance)

import cPickle
if do_affordance:
    file = open('./action-analysis-affordance3.pkl', 'wb')
else:
    file = open('./action-analysis-action.pkl', 'wb')
cPickle.dump((mean_locations_list, all_hits_per_query, all_precision, all_precision_affordance), file)
file.close()

print(all_hits_per_query)
print(all_precision)
print(all_precision_affordance)


# # plot hits and precision for non-affordance actions ...
# file = open('./action-analysis-action.pkl', 'rb')
# mean_locations_list, all_hits_per_query, all_precision, all_precision_affordance = cPickle.load(file)
# file.close()
# print(all_hits_per_query)
# print(all_precision)
# print(all_precision_affordance)
# plt.figure()
# p = plt.subplot(2,1,1)
# plt.plot(mean_locations_list, np.mean(all_hits_per_query, axis=0))
# plt.ylabel('mean # of matches', fontsize=18)
# p.tick_params(axis='x', which='both', labelbottom='off')
# p.tick_params(axis='y', which='both', labelsize=14)
# p = plt.subplot(2,1,2)
# plt.plot(mean_locations_list, all_precision.T)
# plt.ylabel('precision', fontsize=18)
# plt.xlabel('mean # locations per action', fontsize=18)
# p.tick_params(axis='both', which='both', labelsize=14)
# plt.ylim((0, 1.01))
# plt.legend(['250D', '500D', '1000D'], loc=3)
# plt.show()

# # plot hits and precision ...
# file = open('./action-analysis-affordance.pkl', 'rb')
# mean_locations_list, all_hits_per_query, all_precision, all_precision_affordance = cPickle.load(file)
# file.close()
# plt.figure()
# p = plt.subplot(3,1,1)
# plt.plot(mean_locations_list, np.mean(all_hits_per_query, axis=0))
# p.tick_params(axis='x', which='both', labelbottom='off')
# p.tick_params(axis='y', which='both', labelsize=14)
# plt.ylabel('mean objects', fontsize=16)
# p = plt.subplot(3,1,2)
# plt.plot(mean_locations_list, all_precision.T)
# p.tick_params(axis='x', which='both', labelbottom='off')
# p.tick_params(axis='y', which='both', labelsize=14)
# plt.ylabel('object precision', fontsize=16)
# plt.ylim((0, 1.02))
# p = plt.subplot(3,1,3)
# plt.plot(mean_locations_list, all_precision_affordance.T)
# p.tick_params(axis='both', which='both', labelsize=14)
# plt.ylabel('action precision', fontsize=16)
# plt.xlabel('mean # locations & goals per object', fontsize=16)
# plt.ylim((0, 1.02))
# plt.legend(['250D', '500D', '1000D'], loc=3)
# plt.show()
