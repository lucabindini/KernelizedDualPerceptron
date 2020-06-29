import kernels
import performances
import plots

import numpy as np


class KernelizedDualPerceptron:
    def __init__(self, kernel, max_iter):
        self.kernel = kernel
        self.MAX_ITERATION = max_iter
        self.X_train = None
        self.y_train = None
        self.X_validation = None
        self.y_validation = None
        self.X_test = None
        self.num_train_samples, self.num_train_features = None, None
        self.num_test_samples, self.num_test_features = None, None
        self.alpha = None
        self.b = None
        self.R = None
        self.gram = None

    def train(self, X_train, X_validation, y_train, y_validation):
        self.X_train = X_train.to_numpy()
        self.X_validation = X_validation.to_numpy()
        self.y_train = y_train.to_numpy()
        self.num_train_samples, self.num_train_features = self.X_train.shape
        self.alpha = np.zeros(self.num_train_samples)
        self.b = 0
        self.R = self.compute_R()
        self.gram = self.create_gram_matrix()
        validation_error_rate = 1
        validation_errors = []

        print("\nStarting to learn from samples with " + self.kernel + " kernel...")
        for i in range(self.MAX_ITERATION):
            previous_validation_error_rate = validation_error_rate
            old_alpha = self.alpha
            old_b = self.b
            for j in range(self.num_train_samples):
                sum = 0
                for k in range(self.num_train_samples):
                    sum += (self.alpha[k] * self.y_train[k] * self.gram[k][j])
                if (self.y_train[j] * (sum + self.b)) <= 0:
                    self.alpha[j] += 1
                    self.b += (self.y_train[j] * (self.R ** 2))
            validation_error_rate = performances.error_rate(y_validation, self.test(X_validation))
            if validation_error_rate > previous_validation_error_rate:
                self.alpha = old_alpha
                self.b = old_b
                validation_errors.append(validation_error_rate)
                break
            validation_errors.append(validation_error_rate)
        # Uncomment below if you want to plot validation error function (plots are saved into /img folder)
        # plots.validation_error_rate_plot(validation_errors)
        print("End learning")

    def test(self, X_test):
        self.X_test = X_test.to_numpy()
        self.num_test_samples, self.num_test_features = self.X_test.shape
        y_predict = np.zeros(self.num_test_samples)
        for i in range(self.num_test_samples):
            sum = 0
            for j in range(self.num_train_samples):
                if self.kernel == "linear":
                    sum += (self.alpha[j] * self.y_train[j] * kernels.linear_kernel(self.X_train[j],
                                                                                    self.X_test[i]))
                elif self.kernel == "polynomial":
                    sum += (self.alpha[j] * self.y_train[j] * kernels.polynomial_kernel(self.X_train[j],
                                                                                        self.X_test[i]))
                elif self.kernel == "RBF":
                    sum += (self.alpha[j] * self.y_train[j] * kernels.RBF_kernel(self.X_train[j],
                                                                                 self.X_test[i]))
                else:
                    print("Invalid kernel")
            if (sum + self.b) >= 0:
                y_predict[i] = 1
            else:
                y_predict[i] = -1
        return y_predict

    def compute_R(self):
        max_norm = 0
        for i in range(self.num_train_samples):
            if np.linalg.norm(self.X_train[i]) > max_norm:
                max_norm = np.linalg.norm(self.X_train[i], 1)
        return max_norm

    def create_gram_matrix(self):
        gram = np.zeros((self.num_train_samples, self.num_train_samples))
        for i in range(self.num_train_samples):
            for j in range(self.num_train_samples):
                if self.kernel == 'linear':
                    gram[i][j] = kernels.linear_kernel(self.X_train[i], self.X_train[j])
                elif self.kernel == 'polynomial':
                    gram[i][j] = kernels.polynomial_kernel(self.X_train[i], self.X_train[j])
                elif self.kernel == 'RBF':
                    gram[i][j] = kernels.RBF_kernel(self.X_train[i], self.X_train[j])
                else:
                    print("Invalid kernel")
        return gram
