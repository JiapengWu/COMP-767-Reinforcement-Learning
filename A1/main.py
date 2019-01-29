from argparse import ArgumentParser
import numpy as np
import argparse
import matplotlib.pyplot as plt

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

    reward[0] = 1
    reward[n-1] = 10

    for i in range(n*n):
        try:
            if i < n:
                # print(i)
                transition[i][i-1] = 1/3
                transition[i][i+1] = 1/3
                transition[i][i+n] = 1/3
            # right
            elif i % n == n-1:
                transition[i][i-n] = 1/3
                transition[i][i+n] = 1/3
                transition[i][i-1] = 1/3
            # bottom
            elif i >= n*(n-1):
                transition[i][i-n] = 1/3
                transition[i][i+1] = 1/3
                transition[i][i-1] = 1/3
            # left
            elif i % n == 0:
                transition[i][i-n] = 1/3
                transition[i][i+n] = 1/3
                transition[i][i+1] = 1/3
            else:
                transition[i][i + 1] = 1 / 4
                transition[i][i - 1] = 1 / 4
                transition[i][i + n] = 1 / 4
                transition[i][i - n] = 1 / 4
        except: pass

    # top left
    transition[0] = 0
    transition[0][0] = 1
    # top right
    transition[n-1] = 0
    transition[n-1][n-1] = 1
    # bottom left
    transition[n*(n-1)] = 0
    transition[n*(n-1)][n*(n-1) + 1] = 1/2
    transition[n*(n-1)][n*(n-2)] = 1/2
    # bottom right
    transition[n*n-1] = 0
    transition[n*n-1][n*n-2] = 1 / 2
    transition[n*n-1][n*(n-1)-1] = 1 / 2

    up_transition = transition.copy()
    down_transition = transition.copy()
    left_transition = transition.copy()
    right_transition = transition.copy()

    transitions = [up_transition, down_transition, left_transition, right_transition]
    print(transition)
    for i in range(transition.shape[0]):
        if not (i == 0 or i == n - 1):
            idx = np.nonzero(transition[i])[0]
            print(idx)
            for t in transitions:
                t[i][idx] = (1 - p) / (len(idx) - 1)
            try:
                if transition[i][i - n] != 0:
                    up_transition[i][i - n] = p
                else:
                    up_transition[i][idx] = 0
                    up_transition[i][i - n]
            except:
                pass
            try:
                if transition[i][i + n] != 0:
                    down_transition[i][i - n] = p
            except:
                pass
            try:
                if transition[i][i - 1] != 0:
                    left_transition[i][i - 1] = p
            except:
                pass
            try:
                if transition[i][i + 1] != 0:
                    right_transition[i][i + 1] = p
            except:
                pass

    print(up_transition)
    exit()

    up_reward = np.matmul(up_transition, reward)
    down_reward = np.matmul(down_transition, reward)
    left_reward = np.matmul(left_transition, reward)
    right_reward = np.matmul(right_transition, reward)
    rewards = [up_reward, down_reward, left_reward, right_reward]

    return reward, transition, values, transitions, rewards


def policy_iteration():
    # policy evaluation
    while True:
        # old_values = values.copy()
        old_policy = transition.copy()
        global values

        values = np.matmul(np.linalg.inv((np.identity(n*n) - discount * transition)), reward)
        # values = rewards + discount * np.matmul(transition, values)

        results_bottom_left.append(values[n * (n - 1)])
        results_bottom_right.append(values[-1])
        policy_improvement()

        if np.array_equal(old_policy, transition):
            break


def value_iteration():
    while True:
        global values
        old_values = values.copy()

        results_bottom_left.append(values[n * (n - 1)])
        results_bottom_right.append(values[-1])

        values = np.max(np.stack(list(map(lambda x: x[0] + discount * np.matmul(x[1], values),
                                          zip(rewards, transitions)))), axis=0)

        improvement = np.linalg.norm(values - old_values, np.inf)
        if improvement < config.epsilon:
            break

    policy_improvement()


def modified_policy_iteration():
    while True:
        # values = np.matmul(np.linalg.inv((np.eye(n*n) - discount * transition)), rewards)
        global values
        old_values = values.copy()
        for j in range(config.k):
            values = reward + discount * np.matmul(transition, values)

        results_bottom_left.append(values[n * (n - 1)])
        results_bottom_right.append(values[-1])

        policy_improvement()
        improvement = np.linalg.norm(values - old_values, np.inf)
        if improvement < config.epsilon:
            values = reward + discount * np.matmul(transition, values)
            break


def policy_improvement():
    optimal_action = np.argmax(np.stack(list(map(lambda x: x[0] + discount * np.matmul(x[1], values),
                                                     zip(rewards, transitions)))), axis=0)
    for i in range(transition.shape[0]):
        if not (i == 0 or i == n-1):
            idx = np.nonzero(valid_transition[i])[0]
            optim = -1
            if   optimal_action[i] == 0: optim = i - n
            elif optimal_action[i] == 1: optim = i + n
            elif optimal_action[i] == 2: optim = i - 1
            elif optimal_action[i] == 3: optim = i + 1
            transition[i] = 0
            transition[i][idx] = (1 - p) / (len(idx) - 1)
            transition[i][optim] = p


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
    for n in [3, 10]:
        for p in [0.7, 0.9]:
            bottom_left[(n, p)] = []
            bottom_right[(n, p)] = []
            for iteration in range(3):
                results_bottom_left = []
                results_bottom_right = []
                config = get_args()
                discount = 0.9
                reward, transition, values, transitions, rewards = init()
                valid_transition = transition.copy()

                if iteration == 0:
                    value_iteration()
                elif iteration == 1:
                    policy_iteration()
                elif iteration == 2:
                    modified_policy_iteration()

                bottom_left[(n, p)].append(results_bottom_left.copy())
                bottom_right[(n, p)].append(results_bottom_right.copy())

                print(values)
                # action()
    plot()



