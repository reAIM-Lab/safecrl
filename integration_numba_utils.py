import numpy as np
import numba
from scipy import integrate
from numba import cfunc, carray, njit, config, prange, jit
from numba import intc, float64
from numba.core.types.misc import CPointer
from scipy import LowLevelCallable
from rvlib import Normal

integration_lower = -10.
integration_upper = 10.


# NUMBA_DISABLE_JIT = 1


# config.DISABLE_JIT = True
def create_jit_integrand_function(integrand_function, args, args_dtype):
    jitted_function = numba.njit(integrand_function)

    @numba.cfunc(float64(float64, float64, CPointer(args_dtype)))
    def wrapped(x1, x2, user_data_p):
        # Array of structs
        user_data = numba.carray(user_data_p, 1)

        # Extract the data
        x3 = user_data[0].a
        array1 = user_data[0].foo
        array2 = user_data[0].bar

        return jitted_function(x1, x2, x3, array1, array2)

    return wrapped


def jit_integrand_function(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values)

    return LowLevelCallable(wrapped.ctypes)


@njit
def p_x_given_c_u(x, c, u, alpha_x, x_th):
    int_dp = np.dot(np.asarray(alpha_x).ravel(), np.concatenate((np.asarray([u]), c.ravel()))) - x_th
    if int_dp >= 0:
        p_x = 1.
    else:
        p_x = 0.
    return x * p_x + (1 - x) * (1 - p_x) + 1e-10


@njit
def p_y_given_x_c_u(y, x, c, u, alpha_y, y_th):
    c_x = np.concatenate((np.asarray(c), np.asarray([x])), axis=0)
    u_c_x = np.concatenate((np.asarray([u]), np.asarray(c_x)), axis=0)  # .astype(dtype=np.float32)
    int_dp = np.dot(np.asarray(alpha_y).reshape(-1), u_c_x) - y_th  # .astype(dype=np.float32)
    if int_dp >= 0:
        p_y = 1.
    else:
        p_y = 0.
    return y * p_y + (1 - y) * (1 - p_y) + 1e-10


@njit
def p_c_given_u(c, u, alpha_c, beta_c):
    c_diff = np.asarray(c).ravel() - (
            (np.asarray(alpha_c).reshape(-1, 1)).dot(np.asarray(u).reshape(1, 1)) + np.asarray(beta_c).reshape(-1,
                                                                                                               1)).ravel()  # .astype(
    # np.float32)
    c_diff = np.asarray(c_diff)
    n_dist = Normal(0, 1)
    return np.prod(n_dist.pdf(c_diff))  # .astype(dtype=np.float32)


@njit
def p_u_joint(u, u_mean, u_cov):
    n_dist = Normal(u_mean, u_cov)
    return n_dist.pdf(u)  # .astype(dtype=np.float32)


@njit
def p_y_do_x_given_c_func(u, y, x, c, alpha_y, y_th, alpha_c, beta_c, u_mean, u_cov):
    # u, y, x, c, alpha_y, y_th, alpha_c, beta_c, u_mean, u_cov = args
    return p_y_given_x_c_u(y, x, c, u, alpha_y, y_th) * p_c_given_u(c, u, alpha_c, beta_c) * p_u_joint(u, u_mean, u_cov)


@jit_integrand_function
def p_y_do_x_given_c_func_wrapper(args):
    u, y, x, c1, c2, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    c = np.asarray([c1, c2])
    alpha_y = np.asarray([alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3])
    alpha_c = np.asarray([alpha_c_0, alpha_c_1])
    beta_c = np.asarray([beta_c_0, beta_c_1])
    return p_y_do_x_given_c_func(u, y, x, c, alpha_y, y_th, alpha_c, beta_c, u_mean, u_cov)


@njit
def p_x_given_c_func_wrapper(u, x, c_0, c_1, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0,
                             beta_c_1, u_mean, u_cov):
    c = np.asarray([c_0, c_1])
    alpha_x = np.asarray([alpha_x_0, alpha_x_1, alpha_x_2])
    alpha_c = np.asarray([alpha_c_0, alpha_c_1])
    beta_c = np.asarray([beta_c_0, beta_c_1])
    return p_x_given_c_u(x, c, u, alpha_x, x_th) * p_c_given_u(c, u, alpha_c,
                                                               beta_c) * p_u_joint(u, u_mean,
                                                                                   u_cov)


@jit_integrand_function
def p_y_do_x_given_c2_func(args):
    u, c1, y, x, c2, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    c = np.asarray([c1, c2])
    alpha_y = np.asarray([alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3])
    alpha_c = np.asarray([alpha_c_0, alpha_c_1])
    beta_c = np.asarray([beta_c_0, beta_c_1])
    return p_y_do_x_given_c_func(u, y, x, c, alpha_y, y_th, alpha_c, beta_c, u_mean, u_cov)


@jit_integrand_function
def p_y_do_x_given_c1_func(args):
    u, c2, y, x, c1, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    c = np.asarray([c1, c2])
    alpha_y = np.asarray([alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3])
    alpha_c = np.asarray([alpha_c_0, alpha_c_1])
    beta_c = np.asarray([beta_c_0, beta_c_1])
    return p_y_do_x_given_c_func(u, y, x, c, alpha_y, y_th, alpha_c, beta_c, u_mean, u_cov)


@njit
def p_y_given_x_c_func_wrapper(u, y, x, c_0, c_1, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_x_0,
                               alpha_x_1,
                               alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov):
    c = np.asarray([c_0, c_1])
    alpha_y = np.asarray([alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3])
    alpha_x = np.asarray([alpha_x_0, alpha_x_1, alpha_x_2])
    alpha_c = np.asarray([alpha_c_0, alpha_c_1])
    beta_c = np.asarray([beta_c_0, beta_c_1])
    return p_y_given_x_c_u(y, x, c, u, alpha_y, y_th) * p_x_given_c_u(x, c, u, alpha_x, x_th) * p_c_given_u(c, u,
                                                                                                            alpha_c,
                                                                                                            beta_c) * p_u_joint(
        u, u_mean, u_cov)


@jit_integrand_function
def p_y_given_x_c1_func(args):
    u, c2, y, x, c1, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    # @njit
    # def p_y_given_x_c1_func(u, c2, y, x, c1, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov):
    return p_y_given_x_c_func_wrapper(u, y, x, c1, c2, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_x_0,
                                      alpha_x_1,
                                      alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov)
    # return 0


@jit_integrand_function
def p_x_given_c1_func(args):
    u, c2, x, c1, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    return p_x_given_c_func_wrapper(u, x, c1, c2, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0,
                                    beta_c_1, u_mean, u_cov)


@jit_integrand_function
def p_x_given_c2_func(args):
    u, c1, x, c2, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    return p_x_given_c_func_wrapper(u, x, c1, c2, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0,
                                    beta_c_1, u_mean, u_cov)


def p_y_given_x_c1_all(y_vec, x_vec, c1_vec, alpha_y, y_th, alpha_x, x_th, alpha_c, beta_c, u_mean, u_cov):
    p_y_given_x_c1_var = np.zeros(y_vec.shape[0])
    y_comp_array = 1 - y_vec
    for i in prange(y_vec.shape[0]):
        p_y_c1_x = integrate.nquad(p_y_given_x_c1_func, [[integration_lower, integration_upper],
                                                         [integration_lower, integration_upper]],
                                   (y_vec[i], x_vec[i], c1_vec[i, 0], *alpha_y, y_th, *alpha_x, x_th, *alpha_c,
                                    *beta_c, u_mean, u_cov))[0]

        p_y_comp_c1_x = integrate.nquad(p_y_given_x_c1_func, [[integration_lower, integration_upper],
                                                              [integration_lower, integration_upper]],
                                        (y_comp_array[i], x_vec[i], c1_vec[i, 0], *alpha_y, y_th, *alpha_x, x_th,
                                         *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_y_given_x_c1_var[i] = p_y_c1_x / (p_y_c1_x + p_y_comp_c1_x)
        # print(p_y_given_x_c1_var[i])
        if i % 10 == 0:
            print('pygivenxc1', x_vec[i], i)
    return p_y_given_x_c1_var


def p_x_given_c1_all(x_vec, c1_vec, alpha_x, x_th, alpha_c, beta_c, u_mean, u_cov):
    p_x_given_c1_var = np.zeros((x_vec.shape[0], 2))
    for i in prange(x_vec.shape[0]):
        p_x_c1 = integrate.nquad(p_x_given_c1_func, [[integration_lower, integration_upper],
                                                     [integration_lower, integration_upper]],
                                 (x_vec[i], c1_vec[i, 0], *alpha_x, x_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_x_c1_comp = integrate.nquad(p_x_given_c1_func, [[integration_lower, integration_upper],
                                                          [integration_lower, integration_upper]],
                                      (1 - x_vec[i], c1_vec[i, 0], *alpha_x, x_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_x_given_c1_var[i, 0] = p_x_c1 / (p_x_c1 + p_x_c1_comp + 1e-10)
        p_x_given_c1_var[i, 1] = p_x_c1_comp / (p_x_c1 + p_x_c1_comp + 1e-10)

        if i % 10 == 0:
            print('pxgivenc1:', i)

    return p_x_given_c1_var


def p_x_given_c2_all(x_vec, c2_vec, alpha_x, x_th, alpha_c, beta_c, u_mean, u_cov):
    p_x_given_c1_var = np.zeros((x_vec.shape[0], 2))
    for i in prange(x_vec.shape[0]):
        p_x_c1 = integrate.nquad(p_x_given_c2_func, [[integration_lower, integration_upper],
                                                     [integration_lower, integration_upper]],
                                 (x_vec[i], c2_vec[i, 0], *alpha_x, x_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_x_c1_comp = integrate.nquad(p_x_given_c2_func, [[integration_lower, integration_upper],
                                                          [integration_lower, integration_upper]],
                                      (1 - x_vec[i], c2_vec[i, 0], *alpha_x, x_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_x_given_c1_var[i, 0] = p_x_c1 / (p_x_c1 + p_x_c1_comp + 1e-10)
        p_x_given_c1_var[i, 1] = p_x_c1_comp / (p_x_c1 + p_x_c1_comp + 1e-10)

        if i % 10 == 0:
            print('pxgivenc1:', i)

    return p_x_given_c1_var


@jit_integrand_function
def p_y_given_x_c2_func(args):
    u, c1, y, x, c2, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    return p_y_given_x_c_func_wrapper(u, y, x, c1, c2, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alpha_x_0,
                                      alpha_x_1,
                                      alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov)


@jit_integrand_function
def p_x_given_c_func(args):
    u, x, c_0, c_1, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    return p_x_given_c_func_wrapper(u, x, c_0, c_1, alpha_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1,
                                    beta_c_0, beta_c_1, u_mean, u_cov)


@jit_integrand_function
def p_y_given_x_c_func(args):
    u, y, x, c_0, c_1, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alph_x_0, alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean, u_cov = args
    return p_y_given_x_c_func_wrapper(u, y, x, c_0, c_1, alpha_y_0, alpha_y_1, alpha_y_2, alpha_y_3, y_th, alph_x_0,
                                      alpha_x_1, alpha_x_2, x_th, alpha_c_0, alpha_c_1, beta_c_0, beta_c_1, u_mean,
                                      u_cov)


def p_y_given_x_c_all(y_vec, x_vec, c_vec, alpha_y, y_th, alpha_x, x_th, alpha_c,
                      beta_c, u_mean, u_cov):
    p_y_given_x_c_var = np.zeros((y_vec.shape[0], 3))
    for i in prange(y_vec.shape[0]):
        p_y_given_x_c_var[i, 1] = integrate.nquad(p_y_given_x_c_func, [[integration_lower, integration_upper]], (
            y_vec[i], x_vec[i], *c_vec[i], *alpha_y, y_th, *alpha_x, x_th, *alpha_c, *beta_c, u_mean,
            u_cov))[0]
        p_y_given_x_c_var[i, 2] = \
            integrate.nquad(p_y_given_x_c_func, [[integration_lower, integration_upper]], (
                1 - y_vec[i], x_vec[i], *c_vec[i], *alpha_y, y_th, *alpha_x, x_th, *alpha_c, *beta_c, u_mean,
                u_cov))[0]
        p_y_given_x_c_var[i, 0] = p_y_given_x_c_var[i, 1] / (p_y_given_x_c_var[i, 1] + p_y_given_x_c_var[i, 2] + 1e-10)

        if i % 10 == 0:
            print('pygivenxc:', x_vec[i], i)
    return p_y_given_x_c_var


def p_x_given_c_all(x_vec, c_vec, alpha_x, x_th, alpha_c, beta_c, u_mean, u_cov):
    p_x_given_c_var = np.zeros(x_vec.shape[0])
    for i in prange(x_vec.shape[0]):
        p_x_c = integrate.nquad(p_x_given_c_func, [[integration_lower, integration_upper]], (
            x_vec[i], *c_vec[i], *alpha_x, x_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_x_comp_c = \
            integrate.nquad(p_x_given_c_func, [[integration_lower, integration_upper]], (
                1 - x_vec[i], *c_vec[i], *alpha_x, x_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_x_given_c_var[i] = p_x_c / (p_x_c + p_x_comp_c + 1e-10)

        # if i % 10 == 0:
        #    print('pxgivenc:', x_vec[i], i)
    return p_x_given_c_var


def p_y_do_x_given_c1_all(y_vec, x_vec, c1_vec, alpha_y, y_th, alpha_c, beta_c, u_mean, u_cov):
    p_y_do_x_given_c1_var = np.zeros((y_vec.shape[0], 3))
    for i in prange(y_vec.shape[0]):
        p_y_do_x_given_c1_var[i, 1] = \
            integrate.nquad(p_y_do_x_given_c1_func,
                            [[integration_lower, integration_upper], [integration_lower, integration_upper]], (
                                y_vec[i], x_vec[i], *c1_vec[i], *alpha_y, y_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_y_do_x_given_c1_var[i, 2] = \
            integrate.nquad(p_y_do_x_given_c1_func,
                            [[integration_lower, integration_upper], [integration_lower, integration_upper]], (
                                1 - y_vec[i], x_vec[i], *c1_vec[i], *alpha_y, y_th, *alpha_c, *beta_c, u_mean, u_cov))[
                0]
        p_y_do_x_given_c1_var[i, 0] = p_y_do_x_given_c1_var[i, 1] / (
                p_y_do_x_given_c1_var[i, 1] + p_y_do_x_given_c1_var[i, 2] + 1e-10)

        if i % 10 == 0:
            print('pygivendoxc1:', x_vec[i], i)
    return p_y_do_x_given_c1_var


def p_y_do_x_given_c_all(y_vec, x_vec, c_vec, alpha_y, y_th, alpha_c, beta_c, u_mean, u_cov):
    p_y_do_x_given_c_var = np.zeros((y_vec.shape[0], 3))
    for i in prange(y_vec.shape[0]):
        p_y_do_x_given_c_var[i, 1] = \
            integrate.nquad(p_y_do_x_given_c_func_wrapper, [[integration_lower, integration_upper]], (
                y_vec[i], x_vec[i], *c_vec[i], *alpha_y, y_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_y_do_x_given_c_var[i, 2] = \
            integrate.nquad(p_y_do_x_given_c_func_wrapper, [[integration_lower, integration_upper]], (
                1 - y_vec[i], x_vec[i], *c_vec[i], *alpha_y, y_th, *alpha_c, *beta_c, u_mean, u_cov))[0]
        p_y_do_x_given_c_var[i, 0] = p_y_do_x_given_c_var[i, 1] / (
                p_y_do_x_given_c_var[i, 1] + p_y_do_x_given_c_var[i, 2] + 1e-10)

        if i % 10 == 0:
            print('pygivendoxc:', x_vec[i], i)
    return p_y_do_x_given_c_var


@njit
def manski_bounds_all(p_y_given_x_c_vec, p_x_given_c_vec):
    manski_vec = np.zeros((p_y_given_x_c_vec.shape[0], 2))
    for i in prange(p_y_given_x_c_vec.shape[0]):
        manski_vec[i, 0] = p_y_given_x_c_vec[i] * p_x_given_c_vec[i]
        manski_vec[i, 1] = 1  # not implemented
    return manski_vec


@njit
def int_bound_expc1_all(p_y_given_x_c1_c2_vec, p_x_given_c1_c2_vec, p_ydox_given_x_c1_vec, p_y_given_x_c1_vec,
                        p_x_given_c1_vec,
                        p_c2_given_c1_vec):
    int_expc1_vec = np.zeros((p_y_given_x_c1_c2_vec.shape[0], 2))
    for i in prange(p_y_given_x_c1_c2_vec.shape[0]):
        int_expc1_vec[i, 0] = 1 + (p_y_given_x_c1_c2_vec[i] - 1) * p_x_given_c1_c2_vec[i] + (
                p_x_given_c1_vec[i] / p_c2_given_c1_vec[i]) * (
                                      1 - p_y_given_x_c1_vec[i]) + (p_ydox_given_x_c1_vec[i] / p_c2_given_c1_vec[i]) - (
                                      1 / p_c2_given_c1_vec[i])
        int_expc1_vec[i, 1] = 1  # not implemented
    return int_expc1_vec


@njit
def int_bound_expc1_expc2_all(p_y_x_vec, p_y_dox_vec, p_y_dox_given_c1_vec, p_y_dox_given_c2_vec, p_c2_given_c1_vec,
                              p_c1_given_c2_vec,
                              p_c1_c2_vec, p_y_given_x_c1_c2_vec, p_x_given_c1_c2_vec, p_y_given_x_c1_vec,
                              p_y_given_x_c2_vec,
                              p_x_given_c1_vec, p_x_given_c2_vec):
    int_expc1_expc2_vec = np.zeros((p_y_x_vec.shape[0], 2))
    for i in prange(p_y_x_vec.shape[0]):
        int_expc1_expc2_vec[i, 0] = ((p_y_x_vec[i] - p_y_dox_vec[i]) / p_c1_c2_vec[i]) + (
                p_y_dox_given_c1_vec[i] / p_c2_given_c1_vec[i]) + (p_y_dox_given_c2_vec[i] / p_c1_given_c2_vec[i]) + (
                                            p_y_given_x_c1_c2_vec[i] * p_x_given_c1_c2_vec[i]) - (
                                            (p_y_given_x_c2_vec[i] * p_x_given_c2_vec[i]) / p_c1_given_c2_vec[i]) - (
                                            (p_y_given_x_c1_vec[i] * p_x_given_c1_vec[i]) / p_c2_given_c1_vec[i])
        int_expc1_expc2_vec[i, 1] = 1  # not implemented
    return int_expc1_expc2_vec
