import numpy as np


def manski_bounds(p_y_given_x_c, p_x_given_c):
    manski_lb = np.multiply(p_y_given_x_c, p_x_given_c)
    manski_ub = 1  # not implemented
    return manski_lb, manski_ub


def int_bound_expc1(p_y_given_x_c1_c2, p_x_given_c1_c2, p_ydox_given_x_c1, p_y_given_x_c1, p_x_given_c1,
                    p_c2_given_c1):
    int_expc1_lb = 1 + (p_y_given_x_c1_c2 - 1) * p_x_given_c1_c2 + np.divide(p_x_given_c1, p_c2_given_c1) * (
            1 - p_y_given_x_c1) + np.divide(p_ydox_given_x_c1, p_c2_given_c1) - np.divide(1, p_c2_given_c1)
    int_expc1_ub = 1  # not implemented
    return int_expc1_lb, int_expc1_ub


def int_bounds_expc1_expc2(p_y_x, p_y_dox, p_y_dox_given_c1, p_y_dox_given_c2, p_c2_given_c1,
                           p_c1_given_c2,
                           p_c1_c2, p_y_given_x_c1_c2, p_x_given_c1_c2, p_y_given_x_c1, p_y_given_x_c2,
                           p_x_given_c1, p_x_given_c2):
    lb = np.divide((p_y_x - p_y_dox), p_c1_c2) + np.divide(p_y_dox_given_c1, p_c2_given_c1) + np.divide(
        p_y_dox_given_c2, p_c1_given_c2) + np.multiply(p_y_given_x_c1_c2, p_x_given_c1_c2) - np.divide(
        (p_y_given_x_c2 * p_x_given_c2), p_c1_given_c2) - np.divide(
        (p_y_given_x_c1 * p_x_given_c1), p_c2_given_c1)
    ub = 1  # not implemented
    return lb, ub
