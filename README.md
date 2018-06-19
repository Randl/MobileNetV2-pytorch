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

To run continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```
## Results

For x1.0 model I achieved 0.3% higher top-1 accuracy than claimed.
 
|Classification Checkpoint| MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|
|-------------------------|------------|---------------|---------------|---------------|---------------|---------------|
|   [mobilenet_v2_1.0_224]|300         |3.47           |          72.10|          90.48|           71.8|           91.0|
|   [mobilenet_v2_0.5_160]|50          |1.95           |          60.61|          82.87|           61.0|           83.2|

You can test it with
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "results/mobilenet_v2_1.0_224/model_best.pth.tar" -e
python imagenet.py --dataroot "/path/to/imagenet/" --resume "results/mobilenet_v2_0.5_160/model_best.pth.tar" -e --scaling 0.5 --input-size 160
```
