# -*- coding: utf-8 -*-
""""
@author: Hitesh Verma
"""
import numpy as np
import random as rd


class SvmSmo:

    def __init__(self, C, tol, max_iter):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.b = 0
        self.w = None

    def fit(self, x, y):
        m = x.shape[0] #datapoint
        # a = np.ones((m, 1))
        a = np.zeros((m, 1))
        idx = 0
        self.w = self.compute_w(a, x, y)
        self.b = self.compute_b(x, y)
        while idx < self.max_iter:
            idx += 1
            count = 0
            a_old = np.copy(a)
            for i in range(0, m):
                count += 1
                x_i, y_i = x[i, :], y[i]
                # Compute model parameters
                self.w = self.compute_w(a, x, y)
                self.b = self.compute_b(x, y)
                e_i = self.error(x_i, y_i)

                if (y[i] * e_i < -self.tol and a[i] < self.C) or (y[i] * e_i > self.tol and a[i] > 0):
                    # print("check")
                    j = rd.randint(0, m - 1) #j != i
                    while i == j:
                        j = rd.randint(0, m - 1)
                    x_j, y_j = x[j, :], y[j]

                    # Compute e_i, e_j
                    e_i = self.error(x_i, y_i)
                    e_j = self.error(x_j, y_j)

                    a_old_i, a_old_j = a[i], a[j]
                    (L, H) = self.compute_l_h(self.C, a_old_i, a_old_j, y_i, y_j)

                    if L == H:
                        continue

                    nu = 2 * self.kernel(x_i, x_j) - self.kernel(x_i, x_i) - self.kernel(x_j, x_j)
                    if nu >= 0:
                        continue

                    # Set new alpha values
                    a[j] = H if a[j] > H else L if a[j] < L else a_old_j - float(y_j * (e_i - e_j)) / nu

                    if abs(a[j] - a_old_j) < 10e-4:
                        # print("alpha-j not updating", count)
                        continue
                    # else:
                        # print("alpha-j is updating", count)

                    #compute a[i] using a[j]
                    a[i] = a_old_i + y_i * y_j * (a_old_j - a[j])
                    #cal b1 and b2
                    # b1 = self.b - e_i - (y_i * (a[i] -a_old_i) * self.kernel(x_i,x_i)) - (y_j * (a[j] - a_old_j) * self.kernel(x_i,x_j))
                    # b2 = self.b - e_j - (y_i * (a[i] -a_old_i) * self.kernel(x_i,x_j)) - (y_i * (a[j] - a_old_j) * self.kernel(x_j,x_j))
                    # #cal b
                    # if 0 < a[i] < self.C:
                    #     self.b = b1
                    # elif 0 < a[j] < self.C:
                    #     self.b = b2
                    # else:
                    #     self.b = (b1 + b2) / 2

            # Check convergence
            if np.linalg.norm(a - a_old) < self.tol:
                print("converging..",idx)
                break

        return self.w, self.b

    # Prediction error
    def error(self, x_k, y_k):
        return self.compute_prediction(x_k) - y_k

    # Prediction
    def predict(self, x):
        return self.compute_prediction(x)

    def compute_prediction(self, x):
        #w = nx1 #x = n x1
        val = np.dot(self.w.T, x.T) + self.b
        predict = np.where(val > 0, 1, -1)
        # predict = 1 if val > 0 else 0
        # return predict
        return predict.astype(int)

    def compute_b(self, x, y):
        b_tmp = y - np.dot(self.w.T, x.T)
        return np.mean(b_tmp)

    @staticmethod
    def compute_w(alpha, x, y):
        #w = nx1 , alpha= mx1 x =mxn y=mx1
        mul = np.multiply(alpha, y) #mx1
        return np.dot(x.T,mul) #nx1

    @staticmethod
    def kernel(x1, x2):
        return np.dot(x1, x2.T)

    @staticmethod
    def compute_l_h(c, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            return max(0, alpha_j - alpha_i), min(c, c - alpha_i + alpha_j)
        else:
            return max(0, alpha_i + alpha_j - c), min(c, alpha_i + alpha_j)

