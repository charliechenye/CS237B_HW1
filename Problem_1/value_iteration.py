import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_problem, visualize_value_function


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    tf_terminal_mask = tf.cast(terminal_mask, tf.bool)

    # perform value iteration
    for _ in range(1000):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        Va = []

        for a in range(adim):
            Va.append(tf.where(tf_terminal_mask,
                x=reward[:, a],
                y=reward[:, a] + gam * tf.linalg.matvec(Ts[a], V)))

        V_new = tf.reduce_max(Va, axis=0)
        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V

# Gather one trajectory according to the policy #################################
def simulate_trajectory(problem, V, R, gam, goal_idx, 
                        max_step=100, init_x=0, init_y=0):
    Ts = problem["Ts"]
    pos2idx = problem["pos2idx"]
    idx2pos = problem["idx2pos"]
    sdim, adim = Ts[0].shape[-1], len(Ts) 
    x_t = pos2idx[init_x, init_y] # initial state

    tau_x = [0]
    tau_y = [0]
    # simulate trajectory up to max_step
    for _ in range(max_step):
        # compute optimal action
        Qs = []
        for u in range(adim):
            Qs.append(R[x_t, u] + gam * tf.reduce_sum(V * Ts[u][x_t]))
        u_t = tf.argmax(Qs, axis=0)

        # sample from transition dynamics
        x_t_plus_1 = np.random.choice(np.arange(sdim), p=Ts[u_t][x_t].numpy())

        pt = idx2pos[x_t_plus_1]
        tau_x.append(pt[0])
        tau_y.append(pt[1])

        if x_t_plus_1 == goal_idx:
            break

        x_t = x_t_plus_1

    plt.plot(tau_x, tau_y, "r")
 
# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.title("value iteration")
    plt.show()


if __name__ == "__main__":
    main()
