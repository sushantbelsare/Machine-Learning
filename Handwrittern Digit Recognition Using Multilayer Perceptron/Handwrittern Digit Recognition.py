from __future__ import division
import numpy as np
import pprint, pickle
import gzip
from math import pow
from pathlib2 import Path
import random


def vectorize(j):
	e = np.zeros((10,1))
	e[j]=1.0
	return e

def initialize_data():
		with gzip.open('mnist.pkl.gz','rb') as f:
			train_set, valid_set, test_set = pickle.load(f)

		train_x, train_y = train_set
		valid_x, valid_y = valid_set
		test_x, test_y = test_set

		#import matplotlib.cm as cm
		#import matplotlib.pyplot as plt
		#import matplotlib.patches as patches

		#plt.imshow(train_x[5].reshape((28,28)), cmap = cm.Greys_r)

		train_input = [np.reshape(x,(784,1)) for x in train_x]
		train_result = [vectorize(y) for y in train_y]
		train_data = zip(train_input,train_result)

		valid_input = [np.reshape(x,(784,1)) for x in valid_x]
		valid_result = [vectorize(y) for y in valid_y]
		valid_data = zip(valid_input,valid_result)

		test_input = [np.reshape(x,(784,1)) for x in test_x]
		#test_result = [vectorize(y) for y in test_y]
		test_data = zip(test_input,test_y)

		return(train_data, valid_data, test_data)

def initialize_network(num_hidden_layers,num_neurons):

	sizes = []

	for y in range(num_hidden_layers+2):
		if y == 0 :
			sizes.append(784)
		elif y == (num_hidden_layers+1):
			sizes.append(10)
		else:
			sizes.append(num_neurons)

	train_bias = [np.random.randn(y,1) for y in sizes[1:]]

	train_weight = [np.random.randn(y,x) for y,x in zip(sizes[1:], sizes[:-1])]

	return(train_bias, train_weight)

def train(train_bias,train_weight,train_data,epoch,batch_size,learning_rate,test_data=None):
	#calculate the activation for each neuron

	if test_data: n_test = len(test_data)

	for j in range(epoch):
		random.shuffle(train_data)
		mini_batches = [train_data[k:k+batch_size] for k in range(0,len(train_data),batch_size)]

		for mini_batch in mini_batches:

			new_weight = [np.zeros(w.shape) for w in train_weight]
			new_bias = [np.zeros(b.shape) for b in train_bias]

			for train_input,train_result in mini_batch:
				delta_bias, delta_weight = backpropagation(train_input, train_bias, train_weight, train_result)
				new_weight = [nw+dw for nw,dw in zip(new_weight,delta_weight)]
				new_bias = [nb+db for nb,db in zip(new_bias,delta_bias)]

			train_weight = [w - (learning_rate/len(mini_batch))*nw for w,nw in zip(train_weight,new_weight)]
			train_bias = [b - (learning_rate/len(mini_batch))*nb for b,nb in zip(train_bias,new_bias)]

		if test_data:
			print("Epoch {0}: {1}/{2}".format(j,evaluate(train_bias,train_weight,test_data),n_test))

	return(train_bias,train_weight)

def backpropagation(train_input,train_bias,train_weight,train_result):

        new_weight = [np.zeros(w.shape) for w in train_weight]
        new_bias = [np.zeros(b.shape) for b in train_bias]

        activations = [train_input]
        zs = []
        for b,w in zip(train_bias,train_weight):
            activation = np.dot(w,train_input)+b
            train_input = sigmoid(activation)
            activations.append(train_input)
            zs.append(activation)

        delta = cost_derivative(train_input,train_result)*activation_derivative(activations[-1])

        new_bias[-1] = delta
        new_weight[-1] = np.dot(delta,activations[-2].transpose())

        for l in xrange(2,4):
            z = zs[-l]
            delta = np.dot(train_weight[-l+1].transpose(),delta) * activation_derivative(z)
            new_bias[-l] = delta
            new_weight[-l] = np.dot(delta,activations[-l-1].transpose())

        return(new_bias,new_weight)



def cost_derivative(predicted_output,actual_output):
	return(predicted_output - actual_output)

def activation_derivative(z):
	return sigmoid(z)*(1-sigmoid(z))

def evaluate(train_bias,train_weight,test_data):

	test_result = [(np.argmax(feedforward(train_bias,train_weight,test_input)),test_output) for test_input,test_output in test_data]
	return sum(int(x == y) for (x, y) in test_result)

def feedforward(train_bias,train_weight,train_input):

	for b,w in zip(train_bias,train_weight):
		train_input = sigmoid(np.dot(w,train_input)+b)
	return(train_input)

def sigmoid(z):
	return (1.0/(1.0+np.exp(-z)))

def cost(network_result,actual_result):
        cost_array = [pow((x-y),2) for x,y in zip(network_result,actual_result)]
        return (np.sum(cost_array)/2)


if __name__ == "__main__" :

        train_data,valid,test_data = initialize_data()
        b,w = initialize_network(2,16)

        myPath = Path("train_config.p")

        train_bias,train_weight = [],[]
        if myPath.is_file():
            train_config = pickle.load(open("train_config.p","rb"))
            for x,y in train_config:
                train_bias.append(x)
                train_weight.append(y)
        else:
            train_bias,train_weight = train(b,w,train_data,10,30,1,test_data=None)
            train_config = zip(train_bias,train_weight)
            pickle.dump(train_config,open("train_config.p","wb"))

        n = evaluate(train_bias,train_weight,test_data)

        print("Network accuracy : {}".format((float(n*100/len(test_data)))))


