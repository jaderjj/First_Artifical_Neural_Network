"""
Created on Sat May  2 23:11:35 2020

@author: rileyjefferson (rileyjefferson65@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerANN():
    '''
    My first 2-layer Activation Neural Network for
    classification type problems (0 or 1). Includes
    L2 regularization to reduce overfitting.
    '''

    def __init__(self):
        self.l2_lambda = 0.1 # L2 regularization hyperparameter
        self.learning_rate = 0.01 # learning rate hyperparameter

    @staticmethod
    def sigmoid(z):
        # transform z values into [0,1] range
        y = 1/(1+np.exp(-z))
        return y

    def init_params_and_layers(self,x_train, y_train):
        '''
        Parameters
        ----------
        x_train : Training data features (numpy array).
        Must be of shape (a,b) where a and b are real numbers.
        y_train : Training data labels (numpy array). Must be of shape
        (1,b). y_train[1] must be setup to be either 0 or 1 values.
        Model sees each column as an observation and each row value
        associated as a feature for x.

        Returns
        -------
        parameters : Dictionary containing weight and bias values

        '''
        # Initially set weights at random, following normal distribution.
        # Initially set bias terms to zero.
        parameters = {"weight1": np.random.randn(3, x_train.shape[0]) * 0.1,
                      "bias1": np.zeros((3, 1)),
                      "weight2": np.random.randn(y_train.shape[0], 3) * 0.1,
                      "bias2": np.zeros((y_train.shape[0], 1))}
        # Raise exception if given data not shaped correctly.
        if x_train.shape[1] != y_train.shape[1]:
            raise Exception('Dimension size error.',
                            print(TwoLayerANN.init_params_and_layers.__doc__))
        return parameters

    def forward_propagation(self,x_train, parameters):
        '''
        Parameters
        ----------
        x_train : Training data features (numpy array).
        Must be of shape (a,b) where a and b are real numbers..
        parameters : Dictionary containing weight and bias values

        Returns
        -------
        A2 : Result of z = transpoze(weight)*x_train + bias being sent
        through the second activation function.
        cache : Results of z1, z2 and activation functions.

        '''
        # Activation functions I use are hyperbolic tangent function and
        # sigmoid function, respectively.
        Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
        A1 = np.tanh(Z1)
        Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
        A2 = TwoLayerANN().sigmoid(Z2)
        # Create a cache to hold results.
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        return A2, cache

    def compute_cost(self, A2, y_train, parameters):
        '''
        Parameters
        ----------
        A2 : Result of z = transpoze(weight)*x_train + bias being sent
        through the second activation function.
        y_train : Training data labels (numpy array). Must be of shape
        (1,b).
        parameters : Dictionary containing weight and bias values

        Returns
        -------
        cost : cross entropy loss function of results versus training labels

        '''
        logprobs = np.multiply(np.log(A2), y_train)
        cost = -np.sum(logprobs)/y_train.shape[1]
        return cost

    def backward_propagation(self, parameters, cache, x_train, y_train):
        '''
        Parameters
        ----------
        parameters : Dictionary containing weight and bias values
        cache : Results of z1, z2 and activation functions.
        x_train : Training data features (numpy array).
        Must be of shape (a,b) where a and b are real numbers.
        y_train : Training data labels (numpy array). Must be of shape
        (1,b).

        Returns
        -------
        grads : Gradient descent vectors.

        Note: Gradient descent is a method of updating weights and bias
        vectors after evaluating their cost. This is done through a descent,
        calculus method, weight_new = weight - dL(w)/dw and similar method
        for bias; where L is the loss, or cost function.

        '''
        dZ2 = cache["A2"]-y_train
        dW2 = (np.dot(dZ2, cache["A1"].T)+ (self.l2_lambda * parameters['weight2']))/x_train.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True)/x_train.shape[1]
        dZ1 = np.dot(parameters["weight2"].T, dZ2)*(1 - np.power(cache["A1"], 2))
        dW1 = (np.dot(dZ1, x_train.T) + (self.l2_lambda * parameters['weight1']))/x_train.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True)/x_train.shape[1]
        grads = {"dweight1": dW1,
                 "dbias1": db1,
                 "dweight2": dW2,
                 "dbias2": db2}
        return grads

    def update_params(self, parameters, grads):
        '''

        Parameters
        ----------
        parameters : Dictionary containing weight and bias values
        grads : Gradient descent vectors.

        Returns
        -------
        parameters : Updated dictionary containing weight and bias values.
        Update comes from using gradient descent method, where
        weight_new = weight - dL(w)/dw and similar method for bias;
        where L is the loss, or cost function.
        '''
        parameters = {"weight1":
                      parameters["weight1"]-self.learning_rate*grads["dweight1"],
                      "bias1":
                      parameters["bias1"]-self.learning_rate*grads["dbias1"],
                      "weight2":
                      parameters["weight2"]-self.learning_rate*grads["dweight2"],
                      "bias2":
                      parameters["bias2"]-self.learning_rate*grads["dbias2"]}
        return parameters
    
    def predict_on_test(self, parameters, x_test):
        '''
        Parameters
        ----------
        parameters : Updated dictionary containing weight and bias values.
        x_test : Test data features (numpy array).
        Must be of shape (a,b) where a and b are real numbers.

        Returns
        -------
        y_prediction : prediction array
        '''
        A2, _ = TwoLayerANN().forward_propagation(x_test, parameters)
        y_prediction = np.zeros((1, x_test.shape[1]))
        # if z is bigger than 0.5, our prediction is sign one (y_head=1),
        # if z is smaller than or equal to 0.5
        # the prediction is sign zero
        for i in range(A2.shape[1]):
            if A2[0, i] <= 0.5:
                y_prediction[0, i] = 0
            else:
                y_prediction[0, i] = 1
        return y_prediction

    def two_layer_neural_network(self, x_train, y_train,
                                 x_test, y_test, numb_iterations):
        '''
        Parameters
        ----------
        x_train : Training data features (numpy array).
        Must be of shape (a,b) where a and b are real numbers.
        y_train : Training data labels (numpy array). Must be of shape
        (1,b).
        x_test : Similar to x_train, but test data.
        y_test : Similar to y_train, but test data.
        numb_iterations :  Number of iterations (int)

        Returns
        -------
        parameters : Updated parameters that can be logged for future use
        '''
        cost_list = []
        index_list = []
        # Initialize parameters and layer sizes
        parameters = TwoLayerANN().init_params_and_layers(x_train, y_train)
        for i in range(numb_iterations):
            # forward propagation through activation layers
            A2, cache = TwoLayerANN().forward_propagation(x_train, parameters)
            # compute cost (loss function)
            cost = TwoLayerANN().compute_cost(A2, y_train, parameters)
            # backward propagation and gradient descent
            grads = TwoLayerANN().backward_propagation(parameters, cache,
                                                       x_train, y_train)
            # update parameters
            parameters = TwoLayerANN().update_params(parameters, grads)
            if i % 100 == 0:
                print(end='\n')
                cost_list.append(cost)
                index_list.append(i)
                print("Cost after iteration %i: %f" % (i, cost))
        plt.plot(index_list, cost_list)
        plt.xticks(index_list, rotation='vertical')
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        # Reduce number of x-ticks
        plt.locator_params(nbins=10)
        plt.show()
        # classification
        y_prediction_test = TwoLayerANN().predict_on_test(parameters, x_test)
        y_prediction_train = TwoLayerANN().predict_on_test(parameters, x_train)
        # Print Accuracy
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
        return parameters
