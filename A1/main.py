from argparse import ArgumentParser
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

def plot():
    for state, fname in zip([bottom_left, bottom_right], ['bottom left', 'bottom right']):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        keys = list(state.keys())
        count = 0
        for row in range(2):
            for col in range(2):
                n, p = keys[count]
                results = state[(n, p)]
                subplt = ax[row][col]
                subplt.plot(results[0], label="value iteration")
                subplt.plot(results[1], label="policy iteration")
                subplt.plot(results[2], label="modified policy iteration")
                subplt.set_title('n = {}, p = {}'.format(n, p))
                subplt.set_ylabel('value')
                subplt.legend()
                count += 1

        fig.tight_layout()
        fig.suptitle(fname)
        plt.savefig('{}.png'.format(fname))
        plt.clf()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description='Grid world')

    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--p', type=float, default=0.9)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--iteration', type=int, default=2)

    args = parser.parse_args()
    return args


def print_policy():
    for i in range(transition.shape[0]):
        idx = np.nonzero(transition[i])[0]
        optim = idx[np.argmax(values[idx])]
        s = ""
        if optim == i + 1: s = "right"
        if optim == i - 1: s = "left"
        if optim == i + n: s = "down"
        if optim == i - n: s = "up"
        print("{} : {}".format(i, s))


def init():
    reward = np.zeros(n*n)  # up, right, down, left
    transition = np.zeros((n*n, n*n))
    values = np.zeros(n*n)
    up_transition = np.zeros((n*n, n*n))
    down_transition = np.zeros((n*n, n*n))
    left_transition = np.zeros((n*n, n*n))
    right_transition = np.zeros((n*n, n*n))

    transitions = [up_transition, down_transition, left_transition, right_transition]

    reward[0] = 1
    reward[n-1] = 10

    for i in range(transition.shape[0]):
        if not (i == 0 or i == n - 1):
            up = i if i < n else i - n
            down = i if i >= n * (n - 1) else i + n
            left = i if i % n == 0 else i - 1
            right = i if i % n == n - 1 else i + 1
            directions = [up, down, left, right]
            num_other_directions = len(set(directions)) - 1
            for d in directions: transition[i][d] += 1/4
            for t in transitions: t[i][np.array(directions)] = (1-p)/num_other_directions
            up_transition[i][up] = down_transition[i][down] = right_transition[i][right] = left_transition[i][left] = p

    for t in [up_transition, down_transition, left_transition, right_transition, transition]:
        t[0][0] = 1 # top left
        t[n-1][n-1] = 1 # top right

    rewards = list(map(lambda x: np.matmul(x, reward), transitions))

    return reward, transition, values, transitions, rewards


def value_iteration():
    while True:
        global values
        # print(values)
        old_values = values.copy()

        results_bottom_left.append(values[n * (n - 1)])
        results_bottom_right.append(values[-1])
        values = np.max(np.stack(list(map(lambda x: x[0] + discount * np.matmul(x[1], values),
                                          zip(rewards, transitions)))), axis=0)

        improvement = np.linalg.norm(values - old_values, np.inf)
        if improvement < config.epsilon:
            break
    policy_improvement()


def policy_improvement():
    optimal_action = np.argmax(np.stack(list(map(lambda x: x[0] + discount * np.matmul(x[1],
                                                                                       values), zip(rewards, transitions)))), axis=0)
    for i in range(transition.shape[0]):
        if not (i == 0 or i == n-1):
            transition[i] = transitions[optimal_action[i]][i]


def policy_iteration():
    # policy evaluation
    while True:
        global values
        old_values = values.copy()

        # print(values)
        values = np.matmul(np.linalg.inv((np.identity(n*n) - discount * transition)), np.matmul(transition, reward))
        results_bottom_left.append(values[n * (n - 1)])
        results_bottom_right.append(values[-1])
        policy_improvement()

        # if np.array_equal(old_policy, transition):

        improvement = np.linalg.norm(values - old_values, np.inf)
        if improvement < config.epsilon:
            break


def modified_policy_iteration():
    while True:
        global values
        old_values = values.copy()
        for j in range(config.k):
            values = np.matmul(transition, reward) + discount * np.matmul(transition, values)
        results_bottom_left.append(values[n * (n - 1)])
        results_bottom_right.append(values[-1])

        policy_improvement()

        improvement = np.linalg.norm(values - old_values, np.inf)
        if improvement < config.epsilon:
            break


def action():
    cur = n * (n - 1)
    print("Start: {}".format(cur))
    for i in range(n*n):
        if cur == 0 or cur == n-1: break
        prev = cur
        idx = np.nonzero(transition[cur])[0]
        cur = np.random.choice(idx, p=transition[cur][idx])
        s = ""
        if cur == prev + 1: s = "right"
        elif cur == prev - 1: s = "left"
        elif cur == prev + n: s = "down"
        elif cur == prev - n: s = "up"
        print("{:5s} -> {}".format(s, cur))


if __name__ == '__main__':
    bottom_left = {}
    bottom_right = {}
    # for n in [5]:
    #     for p in [0.9]:
    for n in [5, 50]:
        for p in [0.9, 0.7]:
            print("n = {}, p = {}:".format(n, p))
            bottom_left[(n, p)] = []
            bottom_right[(n, p)] = []
            for iteration in range(3):
                results_bottom_left = []
                results_bottom_right = []
                config = get_args()
                discount = 0.9
                reward, transition, values, transitions, rewards = init()
                valid_transition = transition.copy()
                start = time.time()
                if iteration == 0:
                    print("value iteration")
                    value_iteration()
                elif iteration == 1:
                    print("policy iteration")
                    policy_iteration()
                elif iteration == 2:
                    print("modified policy iteration")
                    modified_policy_iteration()
                print(1000 * (time.time() - start))

                bottom_left[(n, p)].append(results_bottom_left.copy())
                bottom_right[(n, p)].append(results_bottom_right.copy())

    plot()



