This project contains scripts to demonstrate basic PyTorch usage.  The code requires python 3, numpy, and pytorch.

To compare a manual backprop calculation with the equivalent PyTorch version, run:

* `python backprop_manual_calculation.py` and
* `python backprop_pytorch.py`

Blog post: [PyTorch Hello World](https://dev.to/nestedsoftware/pytorch-hello-world-37mo)

To train a fully connected network (as described in [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html#exercise_358114) of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), by Michael Nielsen) on the mnist dataset, run:

* `python pytorch_mnist.py`

This network achieves about 97% accuracy on the test dataset, which seems consistent with the results in the book (96.59%). Blog post: [PyTorch Image Recognition with Dense Network](https://dev.to/nestedsoftware/pytorch-image-recognition-dense-network-3nbd)

To train a convolutional network (as described in [chapter 6](http://neuralnetworksanddeeplearning.com/chap6.html#problem_834310) of Michael Nielsen's book), run:

* `python pytorch_mnist_convnet.py`

By default, `python pytorch_mnist_convnet.py` runs the simplest network, `ConvNetSimple`. To see how to run the other networks, you can look at examples in the associated blog post: [PyTorch Image Recognition with Convolutional Networks](https://dev.to/nestedsoftware/pytorch-image-recognition-with-convolutional-networks-4k17). So far, the best consistent results are about _99.6%_.
