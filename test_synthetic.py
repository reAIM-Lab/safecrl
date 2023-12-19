import os.path
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

from bound_computation_utils import *
import pandas as pd
from scipy.special import expit
from scipy.stats import multivariate_normal
from integration_numba_utils import *
import time
from functools import partial
from rvlib import Normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SyntheticDataNoBounds:
    def __init__(self, c_dim=5, u_dim=5, x_dim=1, y_dim=1, n=1000):
        self.c_dim = c_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.U_mean = +0.8
        self.n = n

        # U x C x X x Y
        self.Y_card = np.asarray([0, 1])

        self.X_card = np.asarray([0, 1])

        self.c1_dims = np.asarray([0])
        self.c2_dims = np.asarray([1])

        # self.alpha_c = np.random.uniform(0, 1.2, size=(self.c_dim, 1))
        self.alpha_c = np.asarray([0.2, 0.3]).reshape((self.c_dim, 1))
        # self.beta_c = np.random.uniform(+0.1, 0.5, size=(self.c_dim, 1))
        self.beta_c = np.asarray([-0.2, +1.3]).reshape((self.c_dim, 1))

        self.u_mean = np.asarray([self.U_mean] * self.u_dim)
        # A = np.asarray(0.05 * np.random.randn(self.u_dim, self.u_dim))
        # self.u_cov = A.dot(A.T) + self.u_dim * np.eye(self.u_dim)
        self.u_cov = 1.0087194321164326 * np.eye(self.u_dim)

        self.c_mean = (self.alpha_c.dot(self.u_mean) + self.beta_c)[:, 0]
        self.c_cov = (self.alpha_c.dot(self.u_cov.dot(self.alpha_c.T)) + np.eye(self.c_dim))

        self.u_mean = self.u_mean[0]
        # self.c_mean = np.asarray([0.99968156, 0.97652948])
        # self.c_cov = np.asarray([[1.43750443, 0.57013633], [0.57013633, 1.74297633]])

        # self.alpha_x = np.random.uniform(-0.3, 0.3, size=self.c_dim + self.u_dim)
        self.alpha_x = np.asarray([-0.3, -0.1, -0.2])
        # p_vec0 = np.asarray(np.random.uniform(-0.85, -0.5, size=self.c_dim + self.u_dim))
        # self.alpha_y = np.concatenate((p_vec0, +3.5 * np.ones(1)))  # .astype(np.float32)
        self.alpha_y = np.asarray([-2.6, 8.2, 0.4])

        self.x_th = 0.
        self.y_th = 0.
        self.data_mat = np.zeros((self.n, self.u_dim + self.c_dim + self.x_dim + self.y_dim))
        self.exp_mat_c1 = np.zeros((self.n, self.u_dim + self.c_dim + self.x_dim + self.y_dim))
        self.exp_mat_c2 = np.zeros((self.n, self.u_dim + self.c_dim + self.x_dim + self.y_dim))

    def sample_y(self, u_vec, c_vec, x_vec):
        return expit(
            self.alpha_y[0] * u_vec[:, 0] + self.alpha_y[1] * np.multiply(x_vec, c_vec[:, 0]) + self.alpha_y[
                2] * np.multiply((1 - x_vec), c_vec[:, 1]) +
            np.random.normal(0, 1, size=u_vec.shape[0])) > 0.5

    def load_simulated_toy_highdim_cont(self):
        self.data_mat[:, :self.u_dim] = np.random.normal(loc=self.U_mean, scale=1.0, size=(self.n, self.u_dim))
        self.exp_mat_c1[:, :self.u_dim] = np.random.normal(loc=self.U_mean, scale=1.0, size=(self.n, self.u_dim))
        self.exp_mat_c2[:, :self.u_dim] = np.random.normal(loc=self.U_mean, scale=1.0, size=(self.n, self.u_dim))

        # sample c - hard coded u dimension here - beware!
        for d in range(self.c_dim):
            self.data_mat[:, self.u_dim + d] = self.alpha_c[d, 0] * self.data_mat[:, 0] + \
                                               self.beta_c[d, 0] + 1 * np.random.normal(0, 1, size=self.n)
            self.exp_mat_c1[:, self.u_dim + d] = self.alpha_c[d, 0] * self.exp_mat_c1[:, 0] + \
                                                 self.beta_c[d, 0] + 1 * np.random.normal(0, 1, size=self.n)
            self.exp_mat_c2[:, self.u_dim + d] = self.alpha_c[d, 0] * self.exp_mat_c2[:, 0] + \
                                                 self.beta_c[d, 0] + 1 * np.random.normal(0, 1, size=self.n)

        self.alpha_c = self.alpha_c.ravel()
        self.beta_c = self.beta_c.ravel()

        c1_all = self.data_mat[:, self.u_dim:self.u_dim + self.c_dim]
        c1_all = c1_all[:, self.c1_dims]

        c2_all = self.data_mat[:, self.u_dim:self.u_dim + self.c_dim]
        c2_all = c2_all[:, self.c2_dims]

        # sample x
        self.data_mat[:, self.u_dim + self.c_dim] = np.double(
            expit(np.dot(self.data_mat[:, :self.u_dim + self.c_dim], self.alpha_x) + np.random.normal(0, scale=1, size=
            self.data_mat.shape[0]) - self.x_th) > 0.5)
        self.exp_mat_c1[:, self.u_dim + self.c_dim] = np.random.binomial(n=1, p=0.5, size=self.n)
        self.exp_mat_c2[:, self.u_dim + self.c_dim] = np.random.binomial(n=1, p=0.5, size=self.n)

        # sample y
        self.data_mat[:, self.u_dim + self.c_dim + 1] = self.sample_y(self.data_mat[:, :self.u_dim], self.data_mat[:,
                                                                                                     self.u_dim:self.u_dim + self.c_dim],
                                                                      self.data_mat[:, self.u_dim + self.c_dim + 1])
        self.exp_mat_c1[:, self.u_dim + self.c_dim + 1] = self.sample_y(self.exp_mat_c1[:, :self.u_dim],
                                                                        self.exp_mat_c1[:,
                                                                        self.u_dim:self.u_dim + self.c_dim],
                                                                        self.exp_mat_c1[:, self.u_dim + self.c_dim + 1])
        self.exp_mat_c2[:, self.u_dim + self.c_dim + 1] = self.sample_y(self.exp_mat_c2[:, :self.u_dim],
                                                                        self.exp_mat_c2[:,
                                                                        self.u_dim:self.u_dim + self.c_dim],
                                                                        self.exp_mat_c2[:, self.u_dim + self.c_dim + 1])

        y_x0 = self.sample_y(self.data_mat[:, :self.u_dim], self.data_mat[:, self.u_dim:self.u_dim + self.c_dim],
                             np.zeros(self.data_mat.shape[0]))

        y_x1 = self.sample_y(self.data_mat[:, :self.u_dim], self.data_mat[:, self.u_dim:self.u_dim + self.c_dim],
                             np.ones(self.data_mat.shape[0]))

        plt.figure()
        plt.hist(self.data_mat[:, 0], bins=20)
        plt.savefig('u_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 1], bins=20)
        plt.savefig('c1_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 2], bins=20)
        plt.savefig('c2_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 3], bins=20)
        plt.savefig('x_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 4], bins=20)
        plt.savefig('y_hist.png')

        for i in [0, 1]:
            for j in [0, 1]:
                print('y_i:', i, 'y_j:', j, ':', len(np.intersect1d(np.where(y_x0 == i)[0], np.where(y_x1 == j)[0])))

        print(
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 0) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 0))[0]),
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 0) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 1))[0]),
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 1) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 0))[0]),
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 1) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 1))[0]),
        )

        main_df_obs = pd.DataFrame(self.data_mat, columns=['U', 'C1', 'C2', 'X', 'Y'])
        train_df_obs, test_df_obs = train_test_split(main_df_obs, test_size=0.30, stratify=main_df_obs['X'])

        main_df_expc1 = pd.DataFrame(self.exp_mat_c1, columns=['U', 'C1', 'C2', 'X', 'Y'])
        train_df_expc1, test_df_expc1 = train_test_split(main_df_expc1, test_size=0.30, stratify=main_df_expc1['X'])

        main_df_expc2 = pd.DataFrame(self.exp_mat_c2, columns=['U', 'C1', 'C2', 'X', 'Y'])
        train_df_expc2, test_df_expc2 = train_test_split(main_df_expc2, test_size=0.30, stratify=main_df_expc2['X'])

        data_dict = {'train_df_obs': train_df_obs, 'train_df_expc1': train_df_expc1, 'train_df_expc2': train_df_expc2,
                     'test_df_obs': test_df_obs, 'test_df_expc1': test_df_expc1, 'test_df_expc2': test_df_expc2,
                     'c1_dims': ['C1'], 'c2_dims': ['C2'], 'target': ['Y'], 'treatment': ['X'],
                     'sysbp_scaler': None}

        with open(os.path.join('data', 'synthetic_data_dict_no_bounds_n_%d_d_%d.pkl' % (self.n, self.c_dim)),
                  'wb') as f:
            pkl.dump(data_dict, f)


class SyntheticDataContinuousInt:
    def __init__(self, c_dim=5, u_dim=5, x_dim=1, y_dim=1, n=1000):
        self.c_dim = c_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.U_mean = 0.0
        self.n = n

        # U x C x X x Y
        self.Y_card = np.asarray([0, 1])

        self.X_card = np.asarray([0, 1])

        self.c1_dims = np.asarray([0])
        self.c2_dims = np.asarray([1])

        # self.alpha_c = np.random.uniform(0, 1.2, size=(self.c_dim, 1))
        self.alpha_c = np.asarray([0.2, 0.3]).reshape((self.c_dim, 1))
        # self.beta_c = np.random.uniform(+0.1, 0.5, size=(self.c_dim, 1))
        self.beta_c = np.asarray([-1.1, +1.3]).reshape((self.c_dim, 1))

        self.u_mean = np.asarray([self.U_mean] * self.u_dim)
        # A = np.asarray(0.05 * np.random.randn(self.u_dim, self.u_dim))
        # self.u_cov = A.dot(A.T) + self.u_dim * np.eye(self.u_dim)
        self.u_cov = 1.0087194321164326 * np.eye(self.u_dim)

        self.c_mean = (self.alpha_c.dot(self.u_mean) + self.beta_c)[:, 0]
        self.c_cov = (self.alpha_c.dot(self.u_cov.dot(self.alpha_c.T)) + np.eye(self.c_dim))

        self.u_mean = self.u_mean[0]
        # self.c_mean = np.asarray([0.99968156, 0.97652948])
        # self.c_cov = np.asarray([[1.43750443, 0.57013633], [0.57013633, 1.74297633]])

        # self.alpha_x = np.random.uniform(-0.3, 0.3, size=self.c_dim + self.u_dim)
        self.alpha_x = np.asarray([0.3, -0.1, -0.2])
        # p_vec0 = np.asarray(np.random.uniform(-0.85, -0.5, size=self.c_dim + self.u_dim))
        # self.alpha_y = np.concatenate((p_vec0, +3.5 * np.ones(1)))  # .astype(np.float32)
        self.alpha_y = np.asarray([-2.2, 0.17, 0.04, 1.1])

        self.x_th = 0.
        self.y_th = 0.
        self.data_mat = np.zeros((self.n, self.u_dim + self.c_dim + self.x_dim + self.y_dim))

    def p_c_all(self, c_vec):
        return multivariate_normal.pdf(x=c_vec, mean=self.c_mean, cov=self.c_cov)

    def log_p_c_joint(self, c):
        # log_p = multivariate_normal.logpdf(x=c, mean=self.c_mean, cov=self.c_cov)
        return multivariate_normal.logpdf(x=c, mean=self.c_mean, cov=self.c_cov)

    def p_c2_given_c1(self, c, reverse):
        c1 = c[self.c1_dims]
        c2 = c[self.c2_dims]

        mu_c2 = self.c_mean[self.c2_dims]
        mu_c1 = self.c_mean[self.c1_dims]

        Cov_21 = self.c_cov[self.c2_dims, :]
        Cov_21 = Cov_21[:, self.c1_dims]

        Cov_11 = self.c_cov[self.c1_dims, :]
        Cov_11 = Cov_11[:, self.c1_dims]

        Cov_22 = self.c_cov[self.c2_dims, :]
        Cov_22 = Cov_22[:, self.c2_dims]

        Cov_12 = self.c_cov[self.c1_dims, :]
        Cov_12 = Cov_12[:, self.c2_dims]
        if reverse:
            # p(c1|c2)
            mu_tilde = mu_c1 + np.dot(Cov_12, np.dot(np.linalg.inv(Cov_22), (c2 - mu_c2)))
            Cov_tilde = Cov_11 - np.dot(Cov_12, np.dot(np.linalg.inv(Cov_22), Cov_21))

            p = multivariate_normal(mean=mu_tilde, cov=Cov_tilde).pdf(c1)
        else:
            # p(c2|c1)
            mu_tilde = mu_c2 + np.dot(Cov_21, np.dot(np.linalg.inv(Cov_11), (c1 - mu_c1)))
            Cov_tilde = Cov_22 - np.dot(Cov_21, np.dot(np.linalg.inv(Cov_11), Cov_12))

            p = multivariate_normal(mean=mu_tilde, cov=Cov_tilde).pdf(c2)

        return p

    def p_c2_given_c1_numba(self, c_vec, reverse):
        mu_c2 = self.c_mean[self.c2_dims]
        mu_c1 = self.c_mean[self.c1_dims]

        Cov_21 = self.c_cov[self.c2_dims, :]
        Cov_21 = Cov_21[:, self.c1_dims]

        Cov_11 = self.c_cov[self.c1_dims, :]
        Cov_11 = Cov_11[:, self.c1_dims]

        Cov_22 = self.c_cov[self.c2_dims, :]
        Cov_22 = Cov_22[:, self.c2_dims]

        Cov_12 = self.c_cov[self.c1_dims, :]
        Cov_12 = Cov_12[:, self.c2_dims]

        c1_vec = c_vec[:, self.c1_dims]
        c2_vec = c_vec[:, self.c2_dims]

        if reverse:
            # p(c1|c2)
            mu_tilde = mu_c1 + np.dot(Cov_12, np.dot(np.linalg.inv(Cov_22), (c2_vec - mu_c2)))
            Cov_tilde = Cov_11 - np.dot(Cov_12, np.dot(np.linalg.inv(Cov_22), Cov_21))

            p = Normal(mu_tilde[0], Cov_tilde[0][0]).pdf(c1_vec[0])
        else:
            # p(c2|c1)
            mu_tilde = mu_c2 + np.dot(Cov_21, np.dot(np.linalg.inv(Cov_11), (c1_vec - mu_c1)))
            Cov_tilde = Cov_22 - np.dot(Cov_21, np.dot(np.linalg.inv(Cov_11), Cov_12))

            p = Normal(mu_tilde[0], Cov_tilde[0][0]).pdf(c1_vec[0])

        return p

    def p_x_given_c(self, x, c):
        p_x_given_c = integrate.nquad(p_x_given_c_func, [[integration_lower, integration_upper]], args=(
            x, *c, *self.alpha_x, self.x_th, *self.alpha_c, *self.beta_c, self.u_mean, self.u_cov))[0]
        p_x_comp_given_c = \
            integrate.nquad(p_x_given_c_func, [[integration_lower, integration_upper]], args=(
                1 - x, *c, *self.alpha_x, self.x_th, *self.alpha_c, *self.beta_c, self.u_mean, self.u_cov))[0]
        return p_x_given_c / (p_x_given_c + p_x_comp_given_c + 1e-10)

    def p_x_given_c_numba(self, x_vec, c_vec):
        return p_x_given_c_all(x_vec, c_vec, self.alpha_x, self.x_th, self.alpha_c, self.beta_c, self.u_mean,
                               self.u_cov)

    def p_x_given_c1(self, x, c1):
        p_x_given_c = integrate.nquad(p_x_given_c1_func, [[integration_lower, integration_upper],
                                                          [integration_lower, integration_upper]],
                                      (x, c1, *self.alpha_x, self.x_th, *self.alpha_c, *self.beta_c, self.u_mean,
                                       self.u_cov))[0]
        p_x_comp_given_c = integrate.nquad(p_x_given_c1_func, [[integration_lower, integration_upper],
                                                               [integration_lower, integration_upper]],
                                           (1 - x, c1, *self.alpha_x, self.x_th, *self.alpha_c, *self.beta_c,
                                            self.u_mean,
                                            self.u_cov))[0]
        return p_x_given_c / (p_x_given_c + p_x_comp_given_c + 1e-10), p_x_comp_given_c / (
                p_x_given_c + p_x_comp_given_c + 1e-10)

    def p_x_given_c1_numba(self, x_vec, c1_vec):
        return p_x_given_c1_all(x_vec, c1_vec, self.alpha_x, self.x_th, self.alpha_c, self.beta_c, self.u_mean,
                                self.u_cov)

    def p_y_given_x_c_func(self, u, y, x, c):
        return p_y_given_x_c_u(y, x, c, u, self.alpha_y, self.y_th) * p_x_given_c_func(
            (u, x, c, self.alpha_x, self.x_th, self.alpha_c, self.beta_c, self.u_mean, self.u_cov))

    def p_y_given_x_c(self, y, x, c):
        p_y_c_x = integrate.nquad(p_y_given_x_c_func, [[integration_lower, integration_upper]], (
            y, x, *c, *self.alpha_y, self.y_th, *self.alpha_x, self.x_th, *self.alpha_c, *self.beta_c, self.u_mean,
            self.u_cov))[0]
        p_y_comp_c_x = \
            integrate.nquad(p_y_given_x_c_func, [[integration_lower, integration_upper]], (
                1 - y, x, *c, *self.alpha_y, self.y_th, *self.alpha_x, self.x_th, *self.alpha_c, *self.beta_c,
                self.u_mean,
                self.u_cov))[0]
        return p_y_c_x / (p_y_c_x + p_y_comp_c_x + 1e-10), p_y_c_x, p_y_comp_c_x

    def p_y_given_x_c_numba(self, y_vec, x_vec, c_vec):
        return p_y_given_x_c_all(y_vec, x_vec, c_vec, self.alpha_y, self.y_th, self.alpha_x, self.x_th, self.alpha_c,
                                 self.beta_c, self.u_mean, self.u_cov)

    def p_y_given_x_c1(self, y, x, c1):
        p_y_c1_x = integrate.nquad(p_y_given_x_c1_func, [[integration_lower, integration_upper],
                                                         [integration_lower, integration_upper]],
                                   (y, x, c1, *self.alpha_y, self.y_th, *self.alpha_x, self.x_th, *self.alpha_c,
                                    *self.beta_c, self.u_mean, self.u_cov))[0]
        p_y_comp_c1_x = integrate.nquad(p_y_given_x_c1_func, [[integration_lower, integration_upper],
                                                              [integration_lower, integration_upper]],
                                        (1 - y, x, c1, *self.alpha_y, self.y_th, *self.alpha_x, self.x_th,
                                         *self.alpha_c, *self.beta_c, self.u_mean, self.u_cov))[0]
        return p_y_c1_x / (p_y_c1_x + p_y_comp_c1_x + 1e-10)

    def p_y_given_x_c1_numba(self, y_all, x_all, c1_all):
        return p_y_given_x_c1_all(y_all, x_all, c1_all, self.alpha_y, self.y_th, self.alpha_x,
                                  self.x_th, self.alpha_c, self.beta_c, self.u_mean, self.u_cov)

    def p_y_do_x_given_c(self, y, x, c):
        p_y_c_dox = \
            integrate.quad(p_y_do_x_given_c_func, integration_lower, integration_upper,
                           (y, x, c, self.alpha_y, self.y_th, self.alpha_c, self.beta_c, self.u_mean, self.u_cov))[0]
        p_y_comp_c_dox = \
            integrate.quad(p_y_do_x_given_c_func, integration_lower, integration_upper,
                           (
                               (1 - y, x, c, self.alpha_y, self.y_th, self.alpha_c, self.beta_c, self.u_mean,
                                self.u_cov)))[
                0]
        return p_y_c_dox / (p_y_c_dox + p_y_comp_c_dox + 1e-10), p_y_c_dox, p_y_comp_c_dox

    def p_y_do_x_given_c_numba(self, y_vec, x_vec, c_vec):
        return p_y_do_x_given_c_all(y_vec, x_vec, c_vec, self.alpha_y, self.y_th, self.alpha_c, self.beta_c,
                                    self.u_mean,
                                    self.u_cov)

    def p_y_do_x_given_c1(self, y, x, c1):
        p_y_c1_dox = integrate.nquad(p_y_do_x_given_c1_func, [[integration_lower, integration_upper],
                                                              [integration_lower, integration_upper]],
                                     (y, x, c1, *self.alpha_y, self.y_th, *self.alpha_c, *self.beta_c,
                                      self.u_mean, self.u_cov))[0]
        p_y_comp_c1_dox = integrate.nquad(p_y_do_x_given_c1_func, [[integration_lower, integration_upper],
                                                                   [integration_lower, integration_upper]],
                                          (1 - y, x, c1, *self.alpha_y, self.y_th, *self.alpha_c, *self.beta_c,
                                           self.u_mean, self.u_cov))[0]
        return p_y_c1_dox / (p_y_c1_dox + p_y_comp_c1_dox + 1e-10), p_y_c1_dox, p_y_comp_c1_dox

    def p_y_do_x_given_c1_numba(self, y_vec, x_vec, c1_vec):
        return p_y_do_x_given_c1_all(y_vec, x_vec, c1_vec, self.alpha_y, self.y_th, self.alpha_c, self.beta_c,
                                     self.u_mean, self.u_cov)

    def manski_bounds_est(self, p_y_given_x_c, p_x_given_c):
        manski_lb, manski_ub = manski_bounds(p_y_given_x_c, p_x_given_c)
        return manski_lb, manski_ub

    def int_bound_expc1_est(self, p_y_given_x_c1_c2, p_x_given_c1_c2, p_ydox_given_x_c1, p_y_given_x_c1, p_x_given_c1,
                            p_c2_given_c1):
        int_expc1_lb, int_expc1_ub = int_bound_expc1(p_y_given_x_c1_c2, p_x_given_c1_c2, p_ydox_given_x_c1,
                                                     p_y_given_x_c1, p_x_given_c1,
                                                     p_c2_given_c1)
        return int_expc1_lb, int_expc1_ub

    def int_bounds_expc1_expc2_est(self, p_y_x, p_y_dox, p_y_dox_given_c1, p_y_dox_given_c2, p_c2_given_c1,
                                   p_c1_given_c2,
                                   p_c1_c2, p_y_given_x_c1_c2, p_x_given_c1_c2, p_y_given_x_c1, p_y_given_x_c2,
                                   p_x_given_c1, p_x_given_c2):
        lb, ub = int_bounds_expc1_expc2(p_y_x, p_y_dox, p_y_dox_given_c1, p_y_dox_given_c2, p_c2_given_c1,
                                        p_c1_given_c2,
                                        p_c1_c2, p_y_given_x_c1_c2, p_x_given_c1_c2, p_y_given_x_c1, p_y_given_x_c2,
                                        p_x_given_c1, p_x_given_c2)
        return lb, ub

    def load_simulated_toy_highdim_cont(self):
        self.data_mat[:, :self.u_dim] = np.random.normal(loc=self.U_mean, scale=1.0, size=(self.n, self.u_dim))

        # sample c - hard coded u dimension here - beware!
        for d in range(self.c_dim):
            self.data_mat[:, self.u_dim + d] = self.alpha_c[d, 0] * self.data_mat[:, 0] + \
                                               self.beta_c[d, 0] + 1 * np.random.normal(0, 1, size=self.n)
        self.alpha_c = self.alpha_c.ravel()
        self.beta_c = self.beta_c.ravel()

        c1_all = self.data_mat[:, self.u_dim:self.u_dim + self.c_dim]
        c1_all = c1_all[:, self.c1_dims]

        c2_all = self.data_mat[:, self.u_dim:self.u_dim + self.c_dim]
        c2_all = c2_all[:, self.c2_dims]

        # sample x
        # sample y
        self.data_mat[:, self.u_dim + self.c_dim] = np.double(
            expit(np.dot(self.data_mat[:, :self.u_dim + self.c_dim], self.alpha_x) + np.random.normal(0, scale=1, size=
            self.data_mat.shape[0]) - self.x_th) > 0.5)
        self.data_mat[:, self.u_dim + self.c_dim + 1] = np.double(
            expit(np.dot(self.data_mat[:, :self.u_dim + self.c_dim + 1], self.alpha_y) - self.y_th) > 0.5)

        y_x0 = np.double(
            expit(np.dot(
                np.concatenate((self.data_mat[:, :self.u_dim + self.c_dim], np.zeros((self.data_mat.shape[0], 1))),
                               axis=1), self.alpha_y) + np.random.normal(0, scale=1, size=self.data_mat.shape[
                0]) - self.y_th) > 0.5)
        y_x1 = np.double(
            expit(np.dot(
                np.concatenate((self.data_mat[:, :self.u_dim + self.c_dim], np.ones((self.data_mat.shape[0], 1))),
                               axis=1), self.alpha_y) + np.random.normal(0, scale=1, size=self.data_mat.shape[
                0]) - self.y_th) > 0.5)

        # for i in range(self.n):
        #    print("y_x0", y_x0[i], "y_x1", y_x1[i], "u c1, c2, x, y:", self.data_mat[i])

        plt.figure()
        plt.hist(self.data_mat[:, 0], bins=20)
        plt.savefig('u_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 1], bins=20)
        plt.savefig('c1_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 2], bins=20)
        plt.savefig('c2_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 3], bins=20)
        plt.savefig('x_hist.png')

        plt.figure()
        plt.hist(self.data_mat[:, 4], bins=20)
        plt.savefig('y_hist.png')

        # for i in range(self.data_mat.shape[0]): print("y_x0", y_x0[i], "y_x1", y_x1[i], self.alpha_y, 'u:',
        # self.data_mat[i, :self.u_dim], 'c:',self.data_mat[i, self.u_dim:self.u_dim + self.c_dim])
        for i in [0, 1]:
            for j in [0, 1]:
                print('y_i:', i, 'y_j:', j, ':', len(np.intersect1d(np.where(y_x0 == i)[0], np.where(y_x1 == j)[0])))

        print(
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 0) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 0))[0]),
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 0) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 1))[0]),
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 1) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 0))[0]),
            len(np.where((self.data_mat[:, self.u_dim + self.c_dim] == 1) & (
                    self.data_mat[:, self.u_dim + self.c_dim + 1] == 1))[0]),
        )

        # exit(1)

        start_time = time.time()
        py_dox1givenc1c2_both = self.p_y_do_x_given_c_numba(np.ones(self.n), np.ones(self.n),
                                                            self.data_mat[:, self.u_dim:self.u_dim + self.c_dim])
        py_dox0givenc1c2_both = self.p_y_do_x_given_c_numba(np.ones(self.n), np.zeros(self.n),
                                                            self.data_mat[:, self.u_dim:self.u_dim + self.c_dim])

        py_dox0givenc1c2 = py_dox0givenc1c2_both[:, 0]
        py_dox1givenc1c2 = py_dox1givenc1c2_both[:, 0]

        py_dox1_avgc1c2_vec = np.zeros(self.n)
        py_dox0_avgc1c2_vec = np.zeros(self.n)
        py_dox1_avgc1c2_vec[:] = np.mean(py_dox1givenc1c2)
        py_dox0_avgc1c2_vec[:] = np.mean(py_dox0givenc1c2)

        py_comp_dox1_avgc1c2_vec = np.zeros(self.n)
        py_comp_dox0_avgc1c2_vec = np.zeros(self.n)
        py_comp_dox1_avgc1c2_vec[:] = np.mean(1 - py_dox1givenc1c2)
        py_comp_dox0_avgc1c2_vec[:] = np.mean(1 - py_dox0givenc1c2)

        py_dox1_avgc1c2_vec = np.divide(py_dox1_avgc1c2_vec, (py_dox1_avgc1c2_vec + py_comp_dox1_avgc1c2_vec))

        py_dox0_avgc1c2_vec = np.divide(py_dox0_avgc1c2_vec, (py_dox0_avgc1c2_vec + py_comp_dox0_avgc1c2_vec))

        print('------------- %s seconds to estimate pydox1givenc and pydox0givenc-----------' % (
                time.time() - start_time))

        py_dox1_givenc1_both = self.p_y_do_x_given_c1_numba(np.ones(self.n), np.ones(self.n), c1_all)
        py_dox0_givenc1_both = self.p_y_do_x_given_c1_numba(np.ones(self.n), np.zeros(self.n), c1_all)
        py_dox1_givenc1 = py_dox1_givenc1_both[:, 0]
        py_dox0_givenc1 = py_dox0_givenc1_both[:, 0]

        py_dox1_avgc1_vec = np.zeros(self.n)
        py_dox0_avgc1_vec = np.zeros(self.n)
        py_dox1_avgc1_vec[:] = np.mean(py_dox1_givenc1)
        py_dox0_avgc1_vec[:] = np.mean(py_dox0_givenc1)

        py_comp_dox1_avgc1_vec = np.zeros(self.n)
        py_comp_dox0_avgc1_vec = np.zeros(self.n)
        py_comp_dox1_avgc1_vec[:] = np.mean(1 - py_dox1_givenc1)
        py_comp_dox0_avgc1_vec[:] = np.mean(1 - py_dox0_givenc1)

        py_dox1_avgc1_vec = np.divide(py_dox1_avgc1_vec, (py_dox1_avgc1_vec + py_comp_dox1_avgc1_vec))
        py_dox0_avgc1_vec = np.divide(py_dox0_avgc1_vec, (py_dox0_avgc1_vec + py_comp_dox0_avgc1_vec))

        p_c = self.p_c_all(self.data_mat[:, self.u_dim:self.u_dim + self.c_dim])

        py_dox1_givenc2_both = self.p_y_do_x_given_c1_numba(np.ones(self.n), np.ones(self.n), c2_all)
        py_dox1_givenc2 = py_dox1_givenc2_both[:, 0]
        py_dox0_givenc2_both = self.p_y_do_x_given_c1_numba(np.ones(self.n), np.zeros(self.n), c2_all)
        py_dox0_givenc2 = py_dox0_givenc2_both[:, 0]

        print('------------- %s seconds to estimate up to pydox1/0givenc1,  and pydox1/0givenc2-----------' % (
                time.time() - start_time))

        px1_givenc = self.p_x_given_c_numba(np.ones(self.n), self.data_mat[:, self.u_dim:self.u_dim + self.c_dim])
        px0_givenc = self.p_x_given_c_numba(np.zeros(self.n), self.data_mat[:, self.u_dim:self.u_dim + self.c_dim])

        print('------------- %s seconds to estimate up to pxgivenc -----------' % (
                time.time() - start_time))

        py_givenx1c_both = self.p_y_given_x_c_numba(np.ones(self.n), np.ones(self.n),
                                                    self.data_mat[:, self.u_dim:self.u_dim + self.c_dim])
        py_givenx1c = py_givenx1c_both[:, 0]
        p_y_x1_vec = np.zeros(self.n)
        p_y_x1_vec[:] = np.mean(np.multiply(py_givenx1c_both[:, 1], px1_givenc))
        p_y_comp_x1_vec = np.zeros(self.n)
        p_y_comp_x1_vec[:] = np.mean(np.multiply(1 - py_givenx1c, px1_givenc))

        py_givenx0c_both = self.p_y_given_x_c_numba(np.ones(self.n), np.zeros(self.n),
                                                    self.data_mat[:, self.u_dim:self.u_dim + self.c_dim])

        py_givenx0c = py_givenx0c_both[:, 0]
        p_y_x0_vec = np.zeros(self.n)
        p_y_x0_vec[:] = np.mean(np.multiply(py_givenx0c, px0_givenc))
        p_y_comp_x0_vec = np.zeros(self.n)
        p_y_comp_x0_vec[:] = np.mean(np.multiply(1 - py_givenx0c, px0_givenc))

        # p(y|x, c1)

        p_y_given_x1_c1_vec = self.p_y_given_x_c1_numba(np.ones(self.n), np.ones(self.n), c1_all)
        p_y_given_x0_c1_vec = self.p_y_given_x_c1_numba(np.ones(self.n), np.zeros(self.n), c1_all)

        print('------------- %s seconds to estimate up to pygivenx0/1c and pygivenx1/0c1 -----------' % (
                time.time() - start_time))
        # p(x|c1)
        p_x_given_c1_vec = self.p_x_given_c1_numba(np.ones(self.n), c1_all)
        p_x1_given_c1_vec = p_x_given_c1_vec[:, 0]
        p_x0_given_c1_vec = p_x_given_c1_vec[:, 1]

        # p(y|x, c2)

        print('------------- %s seconds to estimate up to pxgivenc1 -----------' % (
                time.time() - start_time))

        p_y_given_x1_c2_vec = self.p_y_given_x_c1_numba(np.ones(self.n), np.ones(self.n), c2_all)
        p_y_given_x0_c2_vec = self.p_y_given_x_c1_numba(np.ones(self.n), np.zeros(self.n), c2_all)

        p_x_given_c2_vec = self.p_x_given_c1_numba(np.ones(self.n), c2_all)

        print('------------- %s seconds to estimate up to pygivenxc2 -----------' % (
                time.time() - start_time))

        # p(x|c2)

        p_x1_given_c2_vec = p_x_given_c2_vec[:, 0]
        p_x0_given_c2_vec = p_x_given_c2_vec[:, 1]

        print("--- %s seconds for all integrals ---" % (time.time() - start_time))

        start_time = time.time()
        p_c2_given_c1_vec = np.zeros(self.n)
        p_c1_given_c2_vec = np.zeros(self.n)
        func1 = partial(self.p_c2_given_c1, reverse=False)
        func2 = partial(self.p_c2_given_c1, reverse=True)
        for i in range(self.n):
            p_c2_given_c1_vec[i] = func1(self.data_mat[i, self.u_dim:self.u_dim + self.c_dim])
            p_c1_given_c2_vec[i] = func2(self.data_mat[i, self.u_dim:self.u_dim + self.c_dim])

        p_c2_given_c1_vec = np.asarray(p_c2_given_c1_vec)
        p_c1_given_c2_vec = np.asarray(p_c1_given_c2_vec)

        print("--- %s seconds for numba c1, c2 conditionals ---" % (time.time() - start_time))

        start_time = time.time()

        manski_bounds_x1_vec = manski_bounds_all(py_givenx1c, px1_givenc)
        manski_bounds_x0_vec = manski_bounds_all(py_givenx0c, px0_givenc)

        manski_bounds_x1_vec = np.clip(manski_bounds_x1_vec, 0, 1)
        manski_bounds_x0_vec = np.clip(manski_bounds_x0_vec, 0, 1)

        manski_mat = np.concatenate(
            (np.expand_dims(manski_bounds_x0_vec[:, 0], axis=1), np.expand_dims(manski_bounds_x1_vec[:, 0], axis=1)),
            axis=1)
        d_manski = np.argmax(manski_mat, axis=1)
        print("--- %s seconds for numba manski ---" % (time.time() - start_time))

        start_time = time.time()
        int_bounds_x1_expc1_vec = int_bound_expc1_all(py_givenx1c, px1_givenc, py_dox1_givenc1, p_y_given_x1_c1_vec,
                                                      p_x1_given_c1_vec, p_c2_given_c1_vec)
        int_bounds_x0_expc1_vec = int_bound_expc1_all(py_givenx0c, px0_givenc, py_dox0_givenc1, p_y_given_x0_c1_vec,
                                                      p_x0_given_c1_vec, p_c2_given_c1_vec)

        int_bounds_x1_expc2_vec = int_bound_expc1_all(py_givenx1c, px1_givenc, py_dox1_givenc2,
                                                      p_y_given_x1_c2_vec,
                                                      p_x1_given_c2_vec, p_c1_given_c2_vec)
        int_bounds_x0_expc2_vec = int_bound_expc1_all(py_givenx0c, px0_givenc, py_dox0_givenc2,
                                                      p_y_given_x0_c2_vec,
                                                      p_x0_given_c2_vec, p_c1_given_c2_vec)
        int_bounds_x1_expc1_vec = np.clip(int_bounds_x1_expc1_vec, 0, 1)
        int_bounds_x0_expc1_vec = np.clip(int_bounds_x0_expc1_vec, 0, 1)
        int_bounds_x1_expc2_vec = np.clip(int_bounds_x1_expc2_vec, 0, 1)
        int_bounds_x0_expc2_vec = np.clip(int_bounds_x0_expc2_vec, 0, 1)

        exp1_mat = np.concatenate((np.expand_dims(int_bounds_x0_expc1_vec[:, 0], axis=1),
                                   np.expand_dims(int_bounds_x1_expc1_vec[:, 0], axis=1)), axis=1)
        d_exp1 = np.argmax(exp1_mat, axis=1)
        print("--- %s seconds for numba int expc1 ---" % (time.time() - start_time))

        exp2_mat = np.concatenate((np.expand_dims(int_bounds_x0_expc2_vec[:, 0], axis=1),
                                   np.expand_dims(int_bounds_x1_expc2_vec[:, 0], axis=1)), axis=1)
        d_exp2 = np.argmax(exp2_mat, axis=1)

        int_bounds_x1_expc1_expc2_vec = int_bound_expc1_expc2_all(p_y_x1_vec, py_dox1_avgc1c2_vec, py_dox1_givenc1,
                                                                  py_dox1_givenc2,
                                                                  p_c2_given_c1_vec,
                                                                  p_c1_given_c2_vec,
                                                                  p_c,
                                                                  py_givenx1c,
                                                                  px1_givenc,
                                                                  p_y_given_x1_c1_vec,
                                                                  p_y_given_x1_c2_vec,
                                                                  p_x1_given_c1_vec,
                                                                  p_x1_given_c2_vec)
        int_bounds_x0_expc1_expc2_vec = int_bound_expc1_expc2_all(p_y_x0_vec, py_dox0_avgc1c2_vec, py_dox0_givenc1,
                                                                  py_dox0_givenc2,
                                                                  p_c2_given_c1_vec,
                                                                  p_c1_given_c2_vec,
                                                                  p_c,
                                                                  py_givenx0c,
                                                                  px0_givenc,
                                                                  p_y_given_x0_c1_vec,
                                                                  p_y_given_x0_c2_vec,
                                                                  p_x0_given_c1_vec,
                                                                  p_x0_given_c2_vec)

        int_bounds_x0_expc1_expc2_vec = np.clip(int_bounds_x0_expc1_expc2_vec, 0, 1)
        int_bounds_x1_expc1_expc2_vec = np.clip(int_bounds_x1_expc1_expc2_vec, 0, 1)

        print('manski x1:', manski_bounds_x1_vec[:, 0], 'x0:', manski_bounds_x0_vec[:, 0])
        print('int_expc1 x1', int_bounds_x1_expc1_vec[:, 0], 'x0:', int_bounds_x0_expc1_vec[:, 0])
        print('int_expc1 x1', int_bounds_x1_expc2_vec[:, 0], 'x0:', int_bounds_x0_expc2_vec[:, 0])
        print('int_expc1_expc2 x1', int_bounds_x1_expc1_expc2_vec[:, 0], 'x0:', int_bounds_x0_expc1_expc2_vec[:, 0])

        print('alpha_c:', self.alpha_c, 'beta_c:', self.beta_c, 'alpha_x:', self.alpha_x, 'alpha_y:', self.alpha_y,
              'Umean', self.u_mean, 'ucov', self.u_cov, 'c_mean:', self.c_mean, 'c_cov:', self.c_cov)

        int_bounds_exp1_exp2_obs = np.concatenate((np.expand_dims(int_bounds_x0_expc1_expc2_vec[:, 0], axis=1),
                                                   np.expand_dims(int_bounds_x1_expc1_expc2_vec[:, 0], axis=1)),
                                                  axis=1)
        d_exp1_exp2_obs = np.argmax(int_bounds_exp1_exp2_obs, axis=1)

        d_all = np.argmax(
            np.concatenate((np.expand_dims(np.max(np.concatenate((np.expand_dims(manski_mat[:, 0], axis=1),
                                                                  np.expand_dims(exp1_mat[:, 0], axis=1),
                                                                  np.expand_dims(exp2_mat[:, 0], axis=1),
                                                                  np.expand_dims(int_bounds_exp1_exp2_obs[:, 0],
                                                                                 axis=1)), axis=1),
                                                  axis=1), axis=1),
                            np.expand_dims(np.max(np.concatenate((np.expand_dims(manski_mat[:, 1], axis=1),
                                                                  np.expand_dims(exp1_mat[:, 1], axis=1),
                                                                  np.expand_dims(exp2_mat[:, 1], axis=1),
                                                                  np.expand_dims(int_bounds_exp1_exp2_obs[:, 1],
                                                                                 axis=1)), axis=1),
                                                  axis=1), axis=1)),
                           axis=1), axis=1)

        d_upto_expc2 = np.argmax(
            np.concatenate((np.expand_dims(np.max(np.concatenate((np.expand_dims(manski_mat[:, 0], axis=1),
                                                                  np.expand_dims(exp1_mat[:, 0], axis=1),
                                                                  np.expand_dims(exp2_mat[:, 0], axis=1),
                                                                  ), axis=1),
                                                  axis=1), axis=1),
                            np.expand_dims(np.max(np.concatenate((np.expand_dims(manski_mat[:, 1], axis=1),
                                                                  np.expand_dims(exp1_mat[:, 1], axis=1),
                                                                  np.expand_dims(exp2_mat[:, 1], axis=1),
                                                                  ), axis=1),
                                                  axis=1), axis=1)),
                           axis=1), axis=1)

        d_upto_expc1 = np.argmax(
            np.concatenate((np.expand_dims(np.max(np.concatenate((np.expand_dims(manski_mat[:, 0], axis=1),
                                                                  np.expand_dims(exp1_mat[:, 0], axis=1)), axis=1),
                                                  axis=1), axis=1),
                            np.expand_dims(np.max(np.concatenate((np.expand_dims(manski_mat[:, 1], axis=1),
                                                                  np.expand_dims(exp1_mat[:, 1], axis=1)), axis=1),
                                                  axis=1), axis=1)),
                           axis=1), axis=1)

        d_mat = np.concatenate((np.expand_dims(d_manski, axis=1),
                                np.expand_dims(d_exp1, axis=1),
                                np.expand_dims(d_exp2, axis=1),
                                np.expand_dims(d_exp1_exp2_obs, axis=1),
                                np.expand_dims(d_all, axis=1),
                                np.expand_dims(d_upto_expc1, axis=1),
                                np.expand_dims(d_upto_expc2, axis=1),
                                ), axis=1)
        d_mat_df = pd.DataFrame(d_mat,
                                columns=['x_manski', 'x_exp1', 'x_exp2', 'x_obs_exp1_exp2', 'x_all', 'x_upto_exp1',
                                         'x_upto_exp2'])

        self.data_dict = {'p_c': p_c, 'pydox1_givenc1c2': py_dox1givenc1c2,
                          'pydox0_givenc1c2': py_dox0givenc1c2,
                          'pydox1_givenc1': py_dox1_givenc1,
                          'pydox1_givenc2': py_dox1_givenc2,
                          'pydox0_givenc1': py_dox0_givenc1,
                          'pydox0_givenc2': py_dox0_givenc2,
                          'py_givenx1c': py_givenx1c,
                          'py_givenx0c': py_givenx0c,
                          'px1_givenc': px1_givenc,
                          'px0_givenc': px0_givenc,
                          'px1_givenc1': p_x1_given_c1_vec,
                          'px1_givenc2': p_x1_given_c2_vec,
                          'px0_givenc1': p_x0_given_c1_vec,
                          'px0_givenc2': p_x0_given_c2_vec,
                          'pc2_givenc1': p_c2_given_c1_vec,
                          'pc1_givenc2': p_c1_given_c2_vec,
                          'py_givenx1c1': p_y_given_x1_c1_vec,
                          'py_givenx1c2': p_y_given_x1_c2_vec,
                          'py_givenx0c1': p_y_given_x0_c1_vec,
                          'py_givenx0c2': p_y_given_x0_c2_vec,
                          'X_card': self.X_card, 'Y_card': self.Y_card, 'C_card': None, 'c1_dims': self.c1_dims,
                          'c2_dims': self.c2_dims,
                          'manski_bounds_x1': manski_bounds_x1_vec,
                          'int_bounds_x1_expc1': int_bounds_x1_expc1_vec,
                          'int_bounds_x1_expc2': int_bounds_x1_expc2_vec,
                          'int_bounds_x1_expc1_expc2': int_bounds_x1_expc1_expc2_vec,
                          'manski_bounds_x0': manski_bounds_x0_vec,
                          'int_bounds_x0_expc1': int_bounds_x0_expc1_vec,
                          'int_bounds_x0_expc2': int_bounds_x0_expc2_vec,
                          'int_bounds_x0_expc1_expc2': int_bounds_x0_expc1_expc2_vec,
                          'd_mat': d_mat_df
                          }

        self.df_full_obs = pd.DataFrame(self.data_mat[:, self.u_dim:],
                                        columns=[f'C{kk + 1}' for kk in np.arange(self.c_dim)] + ['d', 'y'])
        with open('./data/simulated_data_cont_raw_n_%d_d_%d.pkl' % (self.n, self.c_dim), 'wb') as f:
            pkl.dump({'prob_data': self.data_dict, 'data_frame': self.df_full_obs}, f)

        return self.data_dict, None, self.df_full_obs


def treatment_label_asp(row):
    if row['RXASP'] == 'Y':
        return 1
    else:
        return 0


def selection_bias_rule(row):
    a_rconsc = np.asarray([0.9, 0.7, -0.6])
    a_sex = np.asarray([0.85, -0.1])
    a_age = np.asarray([0.7, -0.1])
    a_rsbp = np.asarray([0.8, 0.5, -0.01, -0.3, -0.6])

    l_rconsc = a_rconsc[0] * (row['RCONSC'] == 0) + a_rconsc[1] * (row['RCONSC'] == 1) + a_rconsc[2] * (
            row['RCONSC'] == 2)
    l_sex = a_sex[0] * (row['SEX'] == 0) + a_sex[1] * (row['SEX'] == 1)
    l_age = a_age[0] * (row['AGE'] == 0) + a_age[1] * (row['AGE'] == 1)
    l_rsbp = a_rsbp[0] * (row['RSBP'] == 0) + a_rsbp[1] * (row['RSBP'] == 1) + a_rsbp[2] * (row['RSBP'] == 2) + \
             a_rsbp[3] * (row['RSBP'] == 3) + a_rsbp[4] * (row['RSBP'] == 4)
    return int(expit(l_rconsc + l_sex + l_age + l_rsbp) > 0.65)


def generate_stroke_trial_data(data_path='/Users/shalmalijoshi/workspace/policylearning_pi/data/IST_corrected.csv',
                               cv=0):
    data_df = pd.read_csv(data_path, engine='python', encoding_errors='replace')

    data_df['RXHEP2'] = 0
    data_df.loc[data_df['RXHEP'].isin(['M', 'L']), 'RXHEP2'] = 1
    data_df['X'] = data_df.apply(lambda row: treatment_label_asp(row), axis=1)
    data_df['Outcome'] = (data_df.FDEAD == 'Y').astype(int)
    data_df['Outcome'] = 1 - data_df['Outcome']
    data_df['AGE'] = data_df['AGE'].astype(int)
    data_df['AGE_standardized'] = (data_df['AGE'] - np.min(data_df['AGE'])) / (
            np.max(data_df['AGE']) - np.min(data_df['AGE']))

    main_df = data_df[['RCONSC', 'SEX', 'X', 'Outcome', 'AGE_standardized', 'RSBP', 'AGE']]
    main_df['RSBP_discrete'] = main_df['RSBP']
    main_df.insert(loc=0, column='AGE_discrete', value=0)
    main_df.loc[main_df['AGE'] >= 73, 'AGE_discrete'] = 1

    main_df['RSBP_discrete'] = pd.cut(main_df['RSBP_discrete'],
                                      bins=[min(main_df['RSBP_discrete']) - 10, 120, 130, 140, 180,
                                            max(main_df['RSBP_discrete']) + 10],
                                      labels=[0, 1, 2, 3, 4])

    SEX_Card = np.unique(main_df['SEX'])
    RCONSC_Card = np.unique(main_df['RCONSC'])

    SEX_dict = {s: t for s, t in zip(SEX_Card, range(len(SEX_Card)))}
    RCONSC_dict = {s: t for s, t in zip(RCONSC_Card, range(len(RCONSC_Card)))}
    main_df['SEX'] = main_df['SEX'].map(SEX_dict)
    main_df['RCONSC'] = main_df['RCONSC'].map(RCONSC_dict)

    # introduce selection bias
    main_df.insert(loc=0, column='Z', value=0)
    main_df['Z'] = main_df.apply(lambda row: selection_bias_rule(row), axis=1)
    plt.hist(main_df['Z'])
    plt.savefig('selection_bias.png')
    plt.close()

    # split data into training and test sets
    train_df, test_df = train_test_split(main_df, test_size=0.30, stratify=main_df['X'])

    scaler = StandardScaler()
    train_df[['RSBP']] = scaler.fit_transform(train_df[['RSBP']])
    test_df[['RSBP']] = scaler.transform(test_df[['RSBP']])

    # remove training points by introducing selection bias
    train_df_obs = train_df[train_df['Z'] == 1]
    train_df_obs = train_df_obs[['X', 'Outcome', 'AGE_discrete', 'SEX', 'RSBP']]
    test_df_obs = test_df[test_df['Z'] == 1][['X', 'Outcome', 'AGE_discrete', 'SEX', 'RSBP']]

    train_df_expc1 = train_df[['X', 'Outcome', 'AGE_discrete', 'SEX', 'Z']]
    train_df_expc2 = train_df[['X', 'Outcome', 'RSBP', 'Z']]
    test_df_expc1 = test_df[['X', 'Outcome', 'AGE_discrete', 'SEX', 'Z']]
    test_df_expc2 = test_df[['X', 'Outcome', 'RSBP', 'Z']]

    data_dict = {'train_df_obs': train_df_obs, 'train_df_expc1': train_df_expc1, 'train_df_expc2': train_df_expc2,
                 'test_df_obs': test_df_obs, 'test_df_expc1': test_df_expc1, 'test_df_expc2': test_df_expc2,
                 'c1_dims': ['AGE_discrete', 'SEX'], 'c2_dims': ['RSBP'], 'target': ['Outcome'], 'treatment': ['X'],
                 'sysbp_scaler': scaler}

    with open(os.path.join('data', 'IST_data_dict.pkl'), 'wb') as f:
        pkl.dump(data_dict, f)
