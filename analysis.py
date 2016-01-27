__author__ = 'bptripp'

import numpy as np
import matplotlib.pyplot as plt
import hrr

# test inner product distributions object hits and misses
n = 25000
total_locations = 250
total_goals = 1000
n_queries = 100

def action_query(mean_locations, mean_goals, D):
    """
    Runs an experiment for statistics on goal&location->action queries.
    """
    n_locations = np.random.poisson(lam=mean_locations, size=n)
    n_goals = np.random.poisson(lam=mean_goals, size=n)
    locations = hrr.CleanupMemory(['LOC' + str(l) for l in range(total_locations)], D)
    goals = hrr.CleanupMemory(['GOAL' + str(l) for l in range(total_goals)], D)
    keys = hrr.CleanupMemory(['LOCATION', 'GOAL'], D)

    def make_random_thing(n_locations, n_goals):
        # choose random location and goal indices
        location_indices = np.unique(np.random.randint(0, total_locations, size=n_locations))
        goal_indices = np.unique(np.random.randint(0, total_goals, size=n_goals))
        vector = make_thing(location_indices, goal_indices)
        return vector, location_indices, goal_indices

    def make_thing(location_indices, goal_indices):
        # make an object HRR vector with these associated locations and goals
        location_sum = np.sum(locations.vectors[:,location_indices], axis=1)
        goal_sum = np.sum(goals.vectors[:,goal_indices], axis=1)
        vector = hrr.bind(keys.get('LOCATION'), location_sum, do_normalize=False) + hrr.bind(keys.get('GOAL'), goal_sum, do_normalize=False)
        return vector

    thing_vectors = np.zeros((D,n))
    location_indices = []
    goal_indices = []
    for i in range(n):
        vector, li, gi = make_random_thing(n_locations[i], n_goals[i])
        thing_vectors[:,i] = vector
        location_indices.append(li)
        goal_indices.append(gi)

    import time
    start_time = time.time()
    matches = []
    partial_matches = []
    non_matches = []
    for i in range(n_queries):
        print(i)
        location_index = np.random.randint(0, total_locations)
        goal_index = np.random.randint(0, total_goals)
        query = make_thing([location_index], [goal_index])
        inner_products = np.dot(thing_vectors.T, query)
        for j in range(n):
            loc_match = location_index in location_indices[j]
            goal_match = goal_index in goal_indices[j]
            if loc_match and goal_match:
                matches.append(inner_products[j])
            elif loc_match or goal_match:
                partial_matches.append(inner_products[j])
            else:
                non_matches.append(inner_products[j])

    print(time.time()-start_time)
    return matches, partial_matches, non_matches


def get_hits_per_query(matches):
    return float(len(matches)) / n_queries

def get_specificity(matches, partial_matches, non_matches):
    sorted = np.sort(matches)
    n = len(matches)
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
# matches, partial_matches, non_matches = action_query(mean_locations, mean_goals, dimension)
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
        mean_goals = mean_locations
        matches, partial_matches, non_matches = action_query(mean_locations, mean_goals, dimension)
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
