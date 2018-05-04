## Naval Mine Identifier using Tensorflow

TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math 
library, and is also used for machine learning applications such as neural networks. Tensorflow allows you to use different 
models without needing to write code for them. You can create your own model with different training algorithms and activation functions.
You can use these models to train your data. Tensor is pivotal part of tensorflow. It is an object that describes relation between vectors,
scalars and other tensors.

In order to execute this code, you'll need to download tensorflow for python. Tensorflow comes in two versions: one for cpu execution and 
one for gpu execution. You can either install tensorflow through command line with "pip3 install --upgrade tensorflow" or 
"pip3 install --upgrade tensorflow-gpu". Or if your cpu supports AVX instructions, you can download it from here : https://github.com/fo40225/tensorflow-windows-wheel
In order to use tensorflow with gpu support, you'll need to install CUDA Toolkit 9.0, CuDNN v7.0 and your gpu must have CUDA compute
capability 3.0 or higher.

The csv file in the folder above has 61 columns with last column to tell whether object is Mine or Rock. We're gonna use one hot encoding 
on our data and train the data against our proposed model. We're going to save this data so that we can use it for later.
