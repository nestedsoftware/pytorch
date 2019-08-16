This project contains scripts to demonstrate basic PyTorch usage.  The code requires python 3, numpy, and pytorch. 

To compare a manual backprop calculation with the equivalent PyTorch version, run:
* `python backprop_manual_calculation.py` and
* `python backprop_pytorch.py`

To train a fully connected network (as described in [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html#exercise_358114) of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), by Michael Nielsen) on the mnist dataset, run:
* `python pytorch_mnist.py`

This network achieves about 97% accuracy on the test dataset, which seems consistent with the results in the book (96.59%).

To train a convolutional network (as described in [chapter 6](http://neuralnetworksanddeeplearning.com/chap6.html#problem_834310) of Michael Nielsen's book), run:
* `python pytorch_mnist_convnet.py`

This network achieves better than _99%_ accuracy on the test dataset, which is close to the results in the book (99.23%). However, in the book, a lambda value of _0.1_ is used for L2 weight regularization. My results got _a lot_ worse when using this value for weight decay. It's possible the network in the book only applied L2 regularization to the fully connected layers. My reading suggests _0.1_ is too high for a convolutional neural network, e.g. [How to use Weight Decay to Reduce Overfitting of Neural Networks in Keras](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/). I've included a weight decay of _0.00001_ for reference, but I'm not sure whether it actually makes any improvement over just using no weight decay at all for this particular scenario.
