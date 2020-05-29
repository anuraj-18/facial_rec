# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:00:58 2018

@author: MAHE
"""

import numpy as np
from scipy import misc
import cv2

def sigmoid(a):
    return 1/(1 + np.exp(-1 * a))

def initialize_parameters(layer_dims):
    parameters={}
    for i in range(0, len(layer_dims)-1):
        #np.random.seed(7)
        np.random.seed(4)
        parameters["W" + str(i + 1)] = np.random.randn(layer_dims[i + 1],layer_dims[i])*1
        parameters["b" + str(i + 1)] = np.zeros((layer_dims[i + 1], 1))
    return parameters
    
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = 0.5 * (1/m) * np.sum(np.power(AL - Y, 2))
    return np.squeeze(cost)

def forward_prop(X, parameters):
    A = X
    L = len(parameters)//2
    cache = {}
    for i in range(1, L+1):
        Z = np.dot(parameters["W" + str(i)], A) + parameters["b" + str(i)]
        cache["Z" + str(i)] = Z
        A = sigmoid(Z)
        cache["A" + str(i)] = A
    return A, cache

def backward_prop(X, Y, parameters, cache):
    grads = {}
    L = len(parameters)//2
    m = Y.shape[1]
    AL = cache["A"+str(L)]
    dZ = np.multiply(AL-Y,np.multiply(AL,1-AL))
    grads["dW" + str(L)] = (1/m) * np.dot(dZ, cache["A" + str(L - 1)].T)
    grads["db" + str(L)]=(1/m) * np.sum(dZ, axis = 1, keepdims = True)
    cache["A0"] = X
    for i in range(L-1, 0, -1):
        y = cache["A" + str(i)]
        dZL_1 = np.dot(parameters["W" + str(i + 1)].T, dZ) * (y - np.power(y, 2))
        grads["dW" + str(i)] = (1/m) * np.dot(dZL_1, cache["A" + str(i - 1)].T)
        grads["db" + str(i)] = (1/m) * np.sum(dZL_1, axis = 1, keepdims = True)
        dZ = dZL_1

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for i in range(1, L+1):
        parameters["W" + str(i)] = parameters["W"+str(i)] - (learning_rate * grads["dW" + str(i)])
        parameters["b" + str(i)] = parameters["b" + str(i)] - (learning_rate * grads["db" + str(i)])
    return parameters

def test_output(X, parameters, layer_dims):
    A = X
    for i in range(1, len(layer_dims)):
        A = np.dot(parameters["W" + str(i)], A) + parameters["b" + str(i)]
        A = sigmoid(A)
    return A

X = []

for i in range(10):
    name = "shubham" + str(i) + ".jpg"
    arr = misc.imread(name) # 640x480x3 array
    arr = np.array(arr) # 3-vector for a pixel
    arr = np.reshape(arr,(1,480*640*3))/255
    X.append(arr[0])
for i in range(5):
    name = "mahika" + str(i) + ".jpg"
    arr = misc.imread(name) # 640x480x3 array
    arr = np.array(arr) # 3-vector for a pixel
    arr = np.reshape(arr, (1, 480*640*3))/255
    X.append(arr[0]
            )
for i in range(1, 5):
    name = "dad" + str(i)+".jpg"
    arr = misc.imread(name) # 640x480x3 array
    arr = np.array(arr) # 3-vector for a pixel
    arr = np.reshape(arr, (1,480*640*3)/255)
    X.append(arr[0])

for i in range(5):
    name = "mum" + str(i) + ".jpg"
    arr = misc.imread(name) # 640x480x3 array
    arr = np.array(arr) # 3-vector for a pixel
    arr = np.reshape(arr, (1,480*640*3))/255
    X.append(arr[0])

X = np.array(X)
X = X.T

Y = []
for i in range(X.shape[1]):
    if i<10:
        k = [1, 0, 0, 0]
        Y.append(k)
    elif i >= 10 and i < 15:
        k = [0, 1, 0, 0]
        Y.append(k)
    elif i >= 15 and i < 19:
        k = [0, 0, 1, 0]
        Y.append(k)
    else:
        k = [0, 0, 0, 1]
        Y.append(k)
Y = np.array(Y)
Y = Y.T
print(Y)

layer_dims = [480*640*3, 7, 7, Y.shape[0]]

parameters = initialize_parameters(layer_dims)

for i in range(301): #301 runs
    AL, cache = forward_prop(X,parameters)
    cost = compute_cost(AL,Y)
    if i%10 == 0:
        print("Cost after " + str(i) + ":" + str(cost))
    grads = backward_prop(X, Y, parameters, cache)
    parameters=update_parameters(parameters, grads, 1.5)
print(AL)

video = cv2.VideoCapture(0)
d={0:"Shubham", 1:"Mahika", 2:"Raj", 3:"Anupma"}

while True:
    X = []
    check, frame = video.read()
    frame = np.reshape(frame, (1,480*640*3))
    X.append(frame[0])
    X = np.array(X)
    X = X.T
    s = test_output(X,parameters,layer_dims)
    m = np.amax(s)
    print(s)
    for i in range(len(s)):
        if s[i][0] == m:
            print("This is " + d[i])
            break
            
    check, frame = video.read()
    
    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    
    if key == ord("q"):
        break
    
video.release()


cv2.destroyAllWindows()
