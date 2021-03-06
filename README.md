# Show, Attend and Read - A PyTorch Implementation

Implementation of Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition in AAAI 2019, with PyTorch >= v1.4.0. 

## Task

- [x] Backbone model
- [x] Encoder model
- [x] Decoder model
- [x] Integrated model
- [x] Data processing
- [x] Training pipeline
- [x] Inference pipeline

## Supported Dataset

- [x] Street View Text: http://vision.ucsd.edu/~kai/svt/
- [x] IIIT5K: https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset
- [x] Syn90k: https://www.robots.ox.ac.uk/~vgg/data/text/
- [x] SynthText: https://www.robots.ox.ac.uk/~vgg/data/scenetext/

## Command

### Training

``
python train.py --batch 32 --epoch 5000 --dataset ./svt --dataset_type svt --gpu True
``

### Inference

``
python inference.py --batch 32 --input input_folder --model model_path --gpu True
``

## Results

### SVT
![Statstics for SVT training](https://github.com/liuch37/sar-pytorch/blob/master/misc/svt_results.png)

### IIIT5K
![Statstics for IIIT5K training](https://github.com/liuch37/sar-pytorch/blob/master/misc/iiit5k_results.png)

Input: 

![Input image](https://github.com/liuch37/sar-pytorch/blob/master/misc/iiit_0.jpg)

Output attention map per character:

![Attention map for char 0](https://github.com/liuch37/sar-pytorch/blob/master/misc/iiit_0_0.png)
![Attention map for char 1](https://github.com/liuch37/sar-pytorch/blob/master/misc/iiit_0_1.png)

### Syn90K (10k for training/3k for testing)
![Statstics for Syn90K training](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_results.png)

Input: 

![Input image](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0.jpg)

Output attention map per character:

![Attention map for char 0](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_0.png)
![Attention map for char 1](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_1.png)
![Attention map for char 2](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_2.png)
![Attention map for char 3](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_3.png)
![Attention map for char 4](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_4.png)
![Attention map for char 5](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_5.png)
![Attention map for char 6](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_6.png)
![Attention map for char 7](https://github.com/liuch37/sar-pytorch/blob/master/misc/syn90k_0_7.png)

### SynthText (80k for training/20k for testing)
![Statstics for SynthText training](https://github.com/liuch37/sar-pytorch/blob/master/misc/synthtext_results.png)

Input:

![Input image](https://github.com/liuch37/sar-pytorch/blob/master/misc/synthtext_0.jpg)

Output attention map per character:

![Attention map for char 0](https://github.com/liuch37/sar-pytorch/blob/master/misc/synthtext_0_0.png)
![Attention map for char 1](https://github.com/liuch37/sar-pytorch/blob/master/misc/synthtext_0_1.png)
![Attention map for char 2](https://github.com/liuch37/sar-pytorch/blob/master/misc/synthtext_0_2.png)

## Source

[1] Original paper: https://arxiv.org/abs/1811.00751

[2] Official code by the authors in torch: https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition

[3] A TensorFlow implementation: https://github.com/Pay20Y/SAR_TF


