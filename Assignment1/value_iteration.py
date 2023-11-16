import numpy as np
import matplotlib.pyplot as plt


# initialize the state and actions


def train():
    X = np.full((6, 6), float('inf'))
    X[2, 0:-1] = float('nan')
    X[4, 3:] = float('nan')

    # create terminal state
    X[0, 1] = 0

    print(X)

    # define actions
    actions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    M = np.zeros((6, 6))
    policies = np.array(list(zip(M.ravel(), M.ravel())), dtype=('i4,i4')).reshape(X.shape)
    while True:
        is_optimal = True
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                action_vals = np.full(len(actions), float('inf'))

                if not np.isnan(X[i, j]):
                    action_vals[0] = X[i, j]
                    a = 1
                    for action in actions[1:]:
                        if 0 <= i + action[0] < X.shape[0] and 0 <= j + action[1] < X.shape[1]:
                            if not np.isnan(X[i + action[0], j + action[1]]):
                                G = 1 + X[i + action[0], j + action[1]]

                                action_vals[a] = G
                        a += 1

                    old_val = X[i, j]
                    X[i, j] = min(action_vals)

                    if old_val != X[i, j]:
                        is_optimal = False
                        policies[i, j] = actions[np.argmin(action_vals)]

        if is_optimal:
            break
    return np.rot90(X), np.rot90(policies)


def plot_results(X, policies):
    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(X)

    for (i, j), z in np.ndenumerate(X):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    scale = 0.25
    plt.subplots(figsize=(6, 6))
    for r, row in enumerate(policies):
        for c, cell in enumerate(row):

            plt.arrow(c, 5 - r, scale * float(cell[0]), scale * float(cell[1]), head_width=0.1)
    plt.show()


vals, policies = train()
plot_results(vals, policies)
