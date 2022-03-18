import random
import numpy as np
import math

n_of_trees = 100
size = math.ceil(math.log2(2 * n_of_trees))


def oracle_matrix():
    o_matrix = np.zeros((n_of_trees*2, n_of_trees*2))
    for i in range(n_of_trees*2):
        if i % 2 == 0:
            o_matrix[i][i] = -1
        else:
            o_matrix[i][i] = 1
    return o_matrix


def oracle(amplitudes, o):
    amplitudes = (o @ amplitudes.T).T
    return amplitudes


def inv_about_mean_matrix(psi):
    d = 2 * psi.T @ psi - np.identity(n_of_trees * 2)
    return d


def inv_about_mean(amplitudes, d):
    amplitudes = (d @ amplitudes.T).T
    return amplitudes


def get_amplitudes(file_name):
    file = open(file_name, 'r')
    y_proba = float(file.readline())
    class1_prob = list(map(float, file.read().split()))
    # print(np.mean(class1_prob))
    psi = np.ndarray((1, n_of_trees * 2))
    for i in range(n_of_trees):
        psi[0][2*i] = math.sqrt(1/n_of_trees * class1_prob[i])
        psi[0][2*i+1] = math.sqrt(1/n_of_trees * (1 - class1_prob[i]))
    return psi, y_proba


def measurement(q_register, repeats=200):
    max_amplitude = 0
    max_state = -1
    for _ in range(repeats):
        rand_state = random.randint(0, n_of_trees * 2-1)
        if math.fabs(q_register[0][rand_state]) > max_amplitude:
            max_amplitude = math.fabs(q_register[0][rand_state])
            max_state = rand_state
    return max_state


def apply_q(repeats, amplitudes, d, o):
    for _ in range(repeats):
        amplitudes = inv_about_mean(amplitudes, d)
        amplitudes = oracle(amplitudes, o)
    return amplitudes


def q_search(filename):
    """
    y_proba - a probability from random forest
    :param filename: file with y_proba and probabilities from trees 
    :return: j
    """
    l = 0
    c = random.random() + 1
    a_on_zero_state, y_proba = get_amplitudes(filename)
    amplitudes = a_on_zero_state.copy()
    o = oracle_matrix()
    d = inv_about_mean_matrix(a_on_zero_state)
    j = 1
    while True:
        l += 1
        m = math.ceil(c ** l)
        i_state = measurement(amplitudes)
        if i_state % 2 == 0 or j > 5:
            return j
        amplitudes = a_on_zero_state.copy()
        j = random.randint(1, l)
        amplitudes = apply_q(j, amplitudes, d, o)


mean = 0
rep = 100
for _ in range(rep):
    j_r = q_search("input_data/correct_x/123.txt")
    mean += j_r
print(mean / rep)
