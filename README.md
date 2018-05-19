# MobileNet2 in PyTorch
An implementation of `MobileNet2` in PyTorch. `MobileNet2` is an efficient convolutional neural network architecture for mobile devices. For more information check the paper:
[Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381) 

## Usage
Clone the repo:
```bash
git clone https://github.com/Randl/MobileNet2-pytorch/
pip install -r requirements.txt
```

Use the model defined in `model.py` to run ImageNet example:
```bash
python imagenet.py --dataroot "/path/to/imagenet/"
```

##Results
TODO: regular model

TODO: x0.35, x0.5, x1.4 model

TODO: Smaller input model

TODO: RMSprop with sgd init

TODO: INT8 model