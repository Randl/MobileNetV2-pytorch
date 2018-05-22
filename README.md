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
 For x0.35 model I achieved 1.5% higher top-1 accuracy than claimed.
 
|Classification Checkpoint| MACs (M)   | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy|
|-------------------------|------------|---------------|---------------|---------------|
| [mobilenet_v2_0.35_224] |59          |1.66           |           61.8|           84.0|

* TODO: regular model(smaller batch (64)) **running**
* TODO: regular model(big batch) 
* TODO: x1.4 model(big batch?)
* TODO: 96 input size model
* TODO: regular model(regular batch) 
* TODO: x0.35(regular batch)
* TODO: RMSprop with sgd init (?)
* TODO: INT8 model (pytorch)