# MobileNetv2 in PyTorch

An implementation of `MobileNetv2` in PyTorch. `MobileNetv2` is an efficient convolutional neural network architecture for mobile devices. For more information check the paper:
[Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381) 

## Usage

Clone the repo:
```bash
git clone https://github.com/Randl/MobileNetV2-pytorch
pip install -r requirements.txt
```

Use the model defined in `model.py` to run ImageNet example:
```bash
python imagenet.py --dataroot "/path/to/imagenet/"
```

##Results

For x0.35 model I achieved 0.3% higher top-1 accuracy than claimed.
 
|Classification Checkpoint| MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|
|-------------------------|------------|---------------|---------------|---------------|---------------|---------------|
| [mobilenet_v2_0.35_224] |300         |3.47           |           72.1|          90.48|           71.8|           91.0|

* TODO: x0.35 model(big batch) 
* TODO: regular model(big batch) 
* TODO: x1.4 model
* TODO: 96 input size model
* TODO: INT8 model (pytorch)