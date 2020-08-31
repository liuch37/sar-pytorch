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
- [ ] SynthText: https://www.robots.ox.ac.uk/~vgg/data/scenetext/

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

### SVT train and test
![Statstics for SVT training](https://github.com/liuch37/sar-pytorch/blob/master/misc/svt_results.png)

## Source

[1] Original paper: https://arxiv.org/abs/1811.00751

[2] Official code by the authors in torch: https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition

[3] A TensorFlow implementation: https://github.com/Pay20Y/SAR_TF


