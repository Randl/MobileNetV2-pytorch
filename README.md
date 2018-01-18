# ShuffleNet in PyTorch
An implementation of `ShuffleNet` in PyTorch. `ShuffleNet` is an efficient convolutional neural network architecture for mobile devices. According to the paper, it outperforms Google's MobileNet by a small percentage.

## What is ShuffleNet?
In one sentence, `ShuffleNet` is a ResNet-like model that uses residual blocks (called `ShuffleUnits`), with the main innovation being the use of pointwise, or 1x1, *group* convolutions as opposed to normal pointwise convolutions. See [paper](https://arxiv.org/abs/1707.01083) for more details.

## Usage
Clone the repo:
```bash
git clone https://github.com/Randl/ShuffleNet.git
pip install -r requirements.txt
```

Use the model defined in `model.py`:
```python
from model import ShuffleNet

# running on MNIST
net = ShuffleNet(num_classes=10, in_channels=1)
```

or just run ImageNet example:
```bash
python imagenet.py --dataroot "/path/to/imagenet/"
```
## Performance
The `ShuffleNet` implementation has been briefly tested (and is tested now )on the ImageNet dataset and achieves ~43% accuracy after 35 epochs. I'm working on acquiring weights for different setups now.


The x0.25 modification with 8 groups achieved same 43% after 175 epochs.

One epoch takes approximetely half an hour on a single 1080 Ti.