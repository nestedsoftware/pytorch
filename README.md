This project contains scripts to demonstrate basic PyTorch usage.  The code requires python 3, numpy, and pytorch.

To compare a manual backprop calculation with the equivalent PyTorch version, run:

```
python backprop_manual_calculation.py
w_l1 = 1.58
b_l1 = -0.14
w_l2 = 2.45
b_l2 = -0.11
a_l2 = 0.8506
updated_w_l1 = 1.5814
updated_b_l1 = -0.1383
updated_w_l2 = 2.4529
updated_b_l2 = -0.1062
updated_a_l2 = 0.8515
```
and
```
python backprop_pytorch.py
network topology: Net(
  (hidden_layer): Linear(in_features=1, out_features=1, bias=True)
  (output_layer): Linear(in_features=1, out_features=1, bias=True)
)
w_l1 = 1.58
b_l1 = -0.14
w_l2 = 2.45
b_l2 = -0.11
a_l2 = 0.8506
updated_w_l1 = 1.5814
updated_b_l1 = -0.1383
updated_w_l2 = 2.4529
updated_b_l2 = -0.1062
updated_a_l2 = 0.8515
```

Blog post: [PyTorch Hello World](https://dev.to/nestedsoftware/pytorch-hello-world-37mo)

To train a fully connected network (as described in [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html#exercise_358114) of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), by Michael Nielsen) on the mnist dataset, run:

```
python pytorch_mnist.py
Test data results: 0.9778
```
This network achieves about 97% accuracy on the test dataset, which seems consistent with the results in the book (96.59%). Blog post: [PyTorch Image Recognition with Dense Network](https://dev.to/nestedsoftware/pytorch-image-recognition-dense-network-3nbd)

To train a convolutional network (as described in [chapter 6](http://neuralnetworksanddeeplearning.com/chap6.html#problem_834310) of Michael Nielsen's book), run the following.

Simple network:

```
python pytorch_mnist_convnet.py
Test data results: 0.9894
```
Two convolutional layers:
```
python pytorch_mnist_convnet.py --net 2conv
Test data results: 0.9915
```
Two convolutional layers with ReLU:
```
python pytorch_mnist_convnet.py --net relu --lr 0.03 --wd 0.00005
Test data results: 0.9926
```
Two convolutional layers and extended training data:
```
python pytorch_mnist_convnet.py --net relu --lr 0.03 --wd 0.00005 --extend_data
Test data results: 0.9942
```
Final network:
```
python pytorch_mnist_convnet.py --net final --epochs 40 --lr 0.005 --extend_data
Test data results: 0.9959
```
Blog post: [PyTorch Image Recognition with Convolutional Networks](https://dev.to/nestedsoftware/pytorch-image-recognition-with-convolutional-networks-4k17).
