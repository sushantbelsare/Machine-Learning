import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np


def readDatasheet():
    data = pd.read_csv("sonar.all-data.csv")

    X = data[data.columns[0:60]].values
    Y = data[data.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    y = one_hot_encoder(Y)
    print(X.shape)
    return X, y


def one_hot_encoder(label):
    n_labels = len(label)
    n_unique_labels = len(np.unique(label))
    one_hot_encoder = np.zeros((n_labels, n_unique_labels))
    one_hot_encoder[np.arange(n_labels), label] = 1
    return one_hot_encoder


X, Y = readDatasheet()

X, Y = shuffle(X, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

learning_rate = 0.3
training_epoch = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("Number of input columns are {0}".format(n_dim))
n_class = 2
model_path = "NMI"

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_dim]))
y_ = tf.placeholder(tf.float32, [None, n_class])


def mlp(x, w, b):

    layer_1 = tf.add(tf.matmul(x, w['h1']), b['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, w['h2']), b['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, w['h3']), b['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, w['h4']), b['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, w['out']) + b['out']

    return out_layer


weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
    }

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
    }

init = tf.global_variables_initializer()

saver = tf.train.Saver()


y = mlp(x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()

sess.run(init)

mse_history = []
accuracy_history = []

for epoch in range(training_epoch):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y}) #Gradient Descent
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y}) #Cost after Descent
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #Accuracy after descent
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y)) #Error in testData
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)

    print("Current epoch : {0} , cost: {1} , MSE : {2} , Train Accuracy : {3}".format(epoch, cost, mse_, accuracy))

save_path = saver.save(sess, model_path)

plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

pred_y = sess.run(y,feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE : %.4f" % sess.run(mse))
