REPL session that shows the mean and standard deviation values for the MNIST data used in this project. These can be used to apply normalization to the data:

```python
>>> from common import *
>>> from torch.utils.data import DataLoader
>>> training_dataset = get_dataset()
>>> training_loader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=False)
>>> training_data = next(iter(training_loader))
>>> training_data[0].mean()
tensor(0.1307)
>>> training_data[0].std()
tensor(0.3081)
>>>
>>> extended_training_dataset = get_extended_dataset()
loading extended training data from file...
>>> extended_training_loader = DataLoader(extended_training_dataset, batch_size=len(extended_training_dataset), shuffle=False)
>>> extended_training_data = next(iter(extended_training_loader))
>>> extended_training_data[0].mean()
tensor(0.1305)
>>> extended_training_data[0].std()
tensor(0.3081)
>>>
>>> test_dataset = get_dataset(train=False)
>>> test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
>>> test_data = next(iter(test_loader))
>>> test_data[0].mean()
tensor(0.1325)
>>> test_data[0].std()
tensor(0.3105)
```
