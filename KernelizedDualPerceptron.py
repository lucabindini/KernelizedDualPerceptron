import numpy as np

import kernels


class KernelizedDualPerceptron:
    def __init__(self, kernel, max_iter):
        self.kernel = kernel
        self.max_iter = max_iter
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.num_train_samples, self.num_train_features = None, None
        self.num_test_samples, self.num_test_features = None, None
        self.alpha = None
        self.b = None
        self.R = None
        self.gram = None

    def train(self, X_train, y_train):
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.num_train_samples, self.num_train_features = self.X_train.shape
        self.alpha = np.zeros(self.num_train_samples)
        self.b = 0
        self.R = self.compute_R()
        self.gram = self.create_gram_matrix()
        # print(str(len(self.y_train)) + " - " + str(self.num_train_samples))
        print("Starting to learn from samples...")
        for i in range(self.max_iter):
            for j in range(self.num_train_samples):
                sum = 0
                for k in range(self.num_train_samples):
                    sum += (self.alpha[k] * self.y_train[k] * self.gram[j][k] + self.b)
                if (self.y_train[j] * sum) <= 0:
                    self.alpha[j] += 1
                    self.b += (self.y_train[j] * (self.R ** 2))
        print("End learning")

    def test(self, X_test):
        self.X_test = X_test.to_numpy()
        self.num_test_samples, self.num_test_features = self.X_test.shape
        y_predict = np.zeros(self.num_test_samples)
        print("Starting to create the prediction for test_set...")
        for i in range(self.num_test_samples):
            sum = 0
            for j in range(self.num_train_samples):
                if self.kernel == "linear":
                    sum += (self.alpha[j] * self.y_train[j] * kernels.linear_kernel(self.X_train[j],
                                                                                    self.X_test[i]) + self.b)
                elif self.kernel == "polynomial":
                    sum += (self.alpha[j] * self.y_train[j] * kernels.polynomial_kernel(self.X_train[j],
                                                                                        self.X_test[i]) + self.b)
                elif self.kernel == "RBF":
                    sum += (self.alpha[j] * self.y_train[j] * kernels.RBF_kernel(self.X_train[j],
                                                                                 self.X_test[i]) + self.b)
                else:
                    print("Invalid kernel")
            # print(sum)
            if sum > 0:
                y_predict[i] = 1
            else:
                y_predict[i] = -1
        print("Prediction created")
        return y_predict

    def compute_R(self):
        max_norm = 0
        for i in range(0, self.num_train_samples):
            if np.linalg.norm(self.X_train[i], 1) > max_norm:
                max_norm = np.linalg.norm(self.X_train[i], 1)
        # print("R:" + str(max_norm))
        return max_norm

    def create_gram_matrix(self):
        print("Generating Gram Matrix with " + self.kernel + " kernel...")
        K = np.zeros((self.num_train_samples, self.num_train_samples))
        for i in range(0, self.num_train_samples):
            for j in range(self.num_train_samples):
                if self.kernel == 'linear':
                    K[i, j] = kernels.linear_kernel(self.X_train[i], self.X_train[j])
                elif self.kernel == 'polynomial':
                    K[i, j] = kernels.polynomial_kernel(self.X_train[i], self.X_train[j])
                elif self.kernel == 'RBF':
                    K[i, j] = kernels.RBF_kernel(self.X_train[i], self.X_train[j])
                else:
                    print("Invalid kernel")
        print("Gram Matrix created")
        return K
