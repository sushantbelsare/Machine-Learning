## Handwritten Digit Recongition Using Multilayer Perceptron
A Multilayer perceptron is a network of neurons called perceptrons. It contains 
minimum of 3 layers(Input Layer, Hidden Layer, Output Layer). Data fed into MLP can be split
into two parts: Train Data, Test Data.

![alt text](http://www.helsinki.fi/~ahonkela/dippa/img191.gif)

Training data consists of features and labels. Features are multidimensional vector values that are fed into the input layer
of network and passed through the network based on activation function. One layer is connected to
another through edges.

![alt text](https://camo.githubusercontent.com/d95fb90b396fc77c614cc6b176dd049066273f96/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f717334746f6a763575356834386c662f6d756c74696c617965725f70657263657074726f6e2e706e673f7261773d31)

Each edge contains weight. A bias is added to the product of input data and weight. Here w0 is bias and w1 through wn are weights.
Based on activation function state of next layer is decided. The activation function shown in above figure is sigmoid function. 
A small change in bias may sometimes affect network significantly.
These parameters are crucial to the network. At the end of the output layer, activation values are compared with the training labels to
calculate how much they differ from expected output. This is called as cost function. Lower the cost, better the accuracy of network.
Although this doesn't apply all the time. If you feed the network with same input data over and over again, you may run into overfitting.

There are many cost functions like Mean Square Error, Cross-Entropy cost function,KL Divergence etc. Choosing the optimal cost function depends upon
type of problem.

![alt text](https://cdn-images-1.medium.com/max/800/0*MdRLxfy4GbQlv97V.)

After calculating error, we try to minimize the cost using gradient descent method. It is analogous to a ball rolling downhill where current
position of ball is our cost and we want to find the direction which can help us minimize the cost. Using this technique we try to modify 
weights and biases that can help us predict better.

![alt text](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

For Handwritten Digit Recongition, we are using MNIST dataset. MNIST dataset contains over 10,000 samples. Each image is 28x28 2D matrix.
We're going change it into 784x1 matrix which means our input layer will have 784 neurons. MNIST dataset has single digit image so range is
0-9. Thus our output layer will have 10 neurons, one for each digit. You can modify number of hidden layers, batch size, number of epochs for
better accuracy. We're going to save training weights and biases so that we don't have to train the system everytime. You can train the system 
with different parameters and save pickle file for each configuration.
