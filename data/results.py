import numpy as np

goal_files = ['2_goal_log.npy', '3_goal_log.npy', '4_goal_log.npy', '5_goal_log.npy']
time_files = ['2_time_log.npy', '3_time_log.npy', '4_time_log.npy', '5_time_log.npy']

for fname in goal_files:
    data = np.load(fname)

    mean = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))
    interval = 1.96 * std_error

    print('Mean Goal Completion for ', fname, ': ', mean)
    print('CIs for ', fname, ': +/-', interval)

for fname in time_files:
    data = np.load(fname)

    mean = np.mean(data)
    std = np.std(data)

    print('Mean Time to Completion for ', fname, ': ', mean)
    print('SD of Time to Completion for ', fname, ': ', std)