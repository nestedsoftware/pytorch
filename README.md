This project contains scripts to demonstrate basic PyTorch usage.  The code requires python 3, numpy, and pytorch.

To compare a manual backprop calculation with the equivalent PyTorch version, run:

* `python backprop_manual_calculation.py` and
* `python backprop_pytorch.py`

Blog post: [PyTorch Hello World](https://dev.to/nestedsoftware/pytorch-hello-world-37mo)

To train a fully connected network (as described in [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html#exercise_358114) of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), by Michael Nielsen) on the mnist dataset, run:

* `python pytorch_mnist.py`

This network achieves about 97% accuracy on the test dataset, which seems consistent with the results in the book (96.59%). Blog post: [PyTorch Image Recognition with Dense Network](https://dev.to/nestedsoftware/pytorch-image-recognition-dense-network-3nbd)

To train a convolutional network (as described in [chapter 6](http://neuralnetworksanddeeplearning.com/chap6.html#problem_834310) of Michael Nielsen's book), run the following.

## Simple network: 

```python
python pytorch_mnist_convnet.py
```
## Two convolutional layers: 
```python
python pytorch_mnist_convnet.py --net 2conv
```
## Two convolutional layers with ReLU: 
```python
python pytorch_mnist_convnet.py --net relu --lr 0.03 --wd 0.00005
```
## Two convolutional layers and extended training data: 
```python
python pytorch_mnist_convnet.py --net relu --lr 0.03 --wd 0.00005 --extend_data
```
## Final network: 
```python
python pytorch_mnist_convnet.py --net final --epochs 40 --lr 0.005 --extend_data
```
Associated blog post: [PyTorch Image Recognition with Convolutional Networks](https://dev.to/nestedsoftware/pytorch-image-recognition-with-convolutional-networks-4k17).
