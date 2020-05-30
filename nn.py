import numpy as np
import cv2
import pickle

class ANN:
    def __init__(self, layers):
        self.params = {}
        for i in range(0, len(layers)-1):
            #np.random.seed(7)
            np.random.seed(4)
            self.params["W" + str(i + 1)] = np.random.randn(layers[i + 1],layers[i])*1
            self.params["b" + str(i + 1)] = np.zeros((layers[i + 1], 1))

    def sigmoid(self, a):
        return 1/(1 + np.exp(-1 * a))
      
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = 0.5 * (1/m) * np.sum(np.power(AL - Y, 2))
        return np.squeeze(cost)

    def forward_prop(self, X):
        A = X
        L = len(self.params)//2
        cache = {}
        for i in range(1, L+1):
            Z = np.dot(self.params["W" + str(i)], A) + self.params["b" + str(i)]
            cache["Z" + str(i)] = Z
            A = self.sigmoid(Z)
            cache["A" + str(i)] = A
        return A, cache

    def backward_prop(self, X, Y, cache):
        grads = {}
        L = len(self.params)//2
        m = Y.shape[1]
        AL = cache["A"+str(L)]
        dZ = np.multiply(AL - Y, np.multiply(AL, 1 - AL))
        grads["dW" + str(L)] = (1/m) * np.dot(dZ, cache["A" + str(L - 1)].T)
        grads["db" + str(L)]=(1/m) * np.sum(dZ, axis = 1, keepdims = True)
        cache["A0"] = X
        for i in range(L-1, 0, -1):
            y = cache["A" + str(i)]
            dZL_1 = np.dot(self.params["W" + str(i + 1)].T, dZ) * (y - np.power(y, 2))
            grads["dW" + str(i)] = (1/m) * np.dot(dZL_1, cache["A" + str(i - 1)].T)
            grads["db" + str(i)] = (1/m) * np.sum(dZL_1, axis = 1, keepdims = True)
            dZ = dZL_1

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.params) // 2
        for i in range(1, L + 1):
            self.params["W" + str(i)] = self.params["W"+str(i)] - (learning_rate * grads["dW" + str(i)])
            self.params["b" + str(i)] = self.params["b" + str(i)] - (learning_rate * grads["db" + str(i)])

    def test_output(self, X):
        A = X
        L = len(self.params) // 2
        for i in range(1, L + 1):
            A = np.dot(self.params["W" + str(i)], A) + self.params["b" + str(i)]
            A = self.sigmoid(A)
        return A

    def train(self, X, Y):
        for i in range(301): #301 runs
            AL, cache = self.forward_prop(X)
            cost = self.compute_cost(AL,Y)
            if i%10 == 0:
                print("Cost after " + str(i) + ":" + str(cost))
            grads = self.backward_prop(X, Y, cache)
            self.update_parameters(grads, 1.5)
            
        self.saveWeights()
        print("Parameters saved in imgWeights.txt")

    def saveWeights(self):
        f = open("imgWts.txt", "wb")
        pickle.dump(self.params, f)
        f.close()

    def getWeights(self):
        f = open("imgWts.txt", "rb")
        self.params = pickle.load(f)
        f.close()
