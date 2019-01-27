from argparse import ArgumentParser
import numpy as np
import argparse


class GridWorld():
    def __init__(self, config):
        self.config = config
        self.n_rows = config.n
        self.pos = 0


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description='Grid world')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--p', type=float, default=0.9)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--iteration', type=int, default=3)
    parser.add_argument('--epsilon', type=str2bool, default="false")

    args = parser.parse_args()
    return args


def print_policy():
    for i in range(transition.shape[0]):
        idx = np.nonzero(transition[i])[0]
        # print(idx)
        # print(values[idx])
        optim = idx[np.argmax(values[idx])]
        s = ""
        if optim == i + 1: s = "right"
        if optim == i - 1: s = "left"
        if optim == i + n: s = "down"
        if optim == i - n: s = "up"
        print("{} : {}".format(i, s))

        # transition[i][idx] = (1-p)/(len(idx) - 1)
        # transition[i][optim] = p


def init():
    rewards = np.zeros(n*n)  # up, right, down, left
    transition = np.zeros((n*n, n*n))
    values = np.zeros(n*n)

    rewards[0] = 1
    rewards[n-1] = 10

    for i in range(n*n):
        try:
            # top
            if i == 1:
                a = 3
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
    transition[0][1] = 1/2
    transition[0][n] = 1/2
    # top right
    transition[n-1] = 0
    transition[n-1][2*n-1] = 1/2
    transition[n-1][n-2] = 1/2
    # bottom left
    transition[n*(n-1)] = 0
    transition[n*(n-1)][n*(n-1) + 1] = 1/2
    transition[n*(n-1)][n*(n-2)] = 1/2
    # bottom right
    transition[n*n-1] = 0
    transition[n*n-1][n*n-2] = 1 / 2
    transition[n*n-1][n*(n-1)-1] = 1 / 2
    # print(transition.tolist())
    # print(rewards)

    return rewards, transition, values


def policy_iteration():
    # policy evaluation
    while True:
        # values = np.matmul(np.linalg.inv((np.eye(n*n) - discount * transition)), rewards)
        global values
        old_values = values.copy()
        values = rewards + discount * np.matmul(transition, values)

        # policy improvement
        # iterate = False
        for i in range(transition.shape[0]):

            idx = np.nonzero(valid_transition[i])[0]
            # print(idx)
            # print(values[idx])
            optim = idx[np.argmax(values[idx])]

            # if transition[i][optim] != 1:
            #     iterate = True
            transition[i] = 0
            transition[i][optim] = 1
        print(values[n * (n - 1)])
        print(values[-1])
        print()
        improvement = np.linalg.norm(values - old_values, np.inf)
        if (improvement < 1): break
        # print(transition)


def value_iteration():
    up_transition = np.zeros((n*n, n*n))
    down_transition = np.zeros((n*n, n*n))
    left_transition = np.zeros((n*n, n*n))
    right_transition = np.zeros((n*n, n*n))

    for i in range(transition.shape[0]):
            try:
                if transition[i][i-n] != 0:
                    up_transition[i][i-n] = 1
            except:pass
            try:
                if transition[i][i+n] != 0:
                    down_transition[i][i - n] = 1
            except:pass
            try:
                if transition[i][i-1] != 0:
                    left_transition[i][i - 1] = 1
            except:pass
            try:
                if transition[i][i+1] != 0:
                    right_transition[i][i + 1] = 1
            except:pass

    transitions = [up_transition, down_transition, left_transition, right_transition]
    while True:
        global values
        old_values = values.copy()
        values = np.max(np.stack(list(map(lambda x: rewards + discount * np.matmul(x, values), transitions))), axis=0)
        print(values[n*(n-1)])
        print(values[-1])
        improvement = np.linalg.norm(values - old_values, np.inf)
        if improvement < 1: break


def modified_policy_iteration():
    while True:
        # values = np.matmul(np.linalg.inv((np.eye(n*n) - discount * transition)), rewards)
        global values
        old_values = values.copy()
        for j in range(config.k):
            values = rewards + discount * np.matmul(transition, values)

        # policy improvement
        # iterate = False
        for i in range(transition.shape[0]):

            idx = np.nonzero(valid_transition[i])[0]
            # print(idx)
            # print(values[idx])
            optim = idx[np.argmax(values[idx])]

            # if transition[i][optim] != 1:
            #     iterate = True
            transition[i] = 0
            transition[i][optim] = 1
        print(values[n * (n - 1)])
        print(values[-1])
        print()
        improvement = np.linalg.norm(values - old_values, np.inf)
        if (improvement < 1): break


def epsilon_greedy_transform():
    for i in range(transition.shape[0]):
        idx = np.nonzero(valid_transition[i])[0]
        # print(idx)
        # print(values[idx])
        optim = idx[np.argmax(values[idx])]
        transition[i] = 0
        transition[i][idx] = (1-p)/(len(idx)-1)
        transition[i][optim] = p
    print(transition)


def action():
    cur = n * (n - 1)
    print("Start: {}".format(cur))
    for i in range(n*n):
        prev = cur
        idx = np.nonzero(transition[cur])[0]
        cur = np.random.choice(idx, p=transition[cur][idx])
        s = ""
        if cur == prev + 1: s = "right"
        elif cur == prev - 1: s = "left"
        elif cur == prev + n: s = "down"
        elif cur == prev - n: s = "up"
        print("{:5s} -> {}".format(s, cur))


def test():
    if config.epsilon:
        epsilon_greedy_transform()
    action()


if __name__ == '__main__':
    config = get_args()
    n = config.n
    p = config.p
    discount = config.discount
    rewards, transition, values = init()
    valid_transition = transition.copy()

    if config.iteration == 1:
        policy_iteration()
    elif config.iteration == 2:
        value_iteration()
    elif config.iteration == 3:
        modified_policy_iteration()
    test()
