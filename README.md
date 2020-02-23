<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/FxL5qM0.jpg" alt="Bot logo"></a>
</p>

<h3 align="center">General Classification</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]() [![GitHub Issues](https://img.shields.io/github/issues/insightcs/general_classification.PyTorch.svg)](https://github.com/insightcs/general_classification.PyTorch/issues) [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/insightcs/general_classification.PyTorch.svg)](https://github.com/insightcs/general_classification.PyTorch/pulls) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> This is a PyTorch implementation of general image classification on any image dataset. Hopefully it'll be of use to others. 
    <br> 
</p>

## 📝 Table of Contents

* [Highlights](#highlights)
* [Pre-Requisites](#pre-requisites)
* [Usage](#usage)
  * [Preparing datasets](#preparing_datasets)
  * [Trainging](#training)
  * [Testing](#testing)
* [TODO](#todo)
* [Acknowledgments](#acknowledgement)

## 🔥 Highlights <a name = "highlights"></a>

* This project included a few of my favourite models, and most models do have pretrained weights from their respective sources or original authors.
  * ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNeXt50 (32x4d), ResNeXt101 (32x4d and 64x4d).
  * [DLA](https://arxiv.org/pdf/1707.06484.pdf).
  * Generic EfficientNet - A generic model that implements many of the efficient models that utilize similar DepthwiseSeparable and InvertedResidual blocks.
    * [EfficientNet(B0-B7)](https://arxiv.org/abs/1905.11946).
    * [MixNet](https://arxiv.org/abs/1907.09595).
    * MobileNet-V1, MobileNet-V2, MobileNet-V3. 
* **Syncronized Batch Normalization on PyTorch**: you can find sync_bn module in `models/sync_batchnorm`, please refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) for details.
* All models can load pretrained weight loader that adapts last linear if necessary, and from 3 to 1 channel input if desired.
* A dynamic global pool implementation that allows selecting from average pooling, max pooling, average + max, or concat([average, max]) at model creation.
* LR scheduler such as Cosine LR, StepLR, PlateauLR. LR scheduler ideas from [AllenNLP](https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers), [FAIRseq](https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler).
* Random Erasing from [Zhun Zhong](https://arxiv.org/abs/1708.04896).
* [Label Smoothing](https://arxiv.org/pdf/1906.02629.pdf).
* [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf) and [RandAugment](https://arxiv.org/pdf/1909.13719.pdf) implementation adapted from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py.
* An inference script(`demo.py`) that predicts some images is provided as an example.
* This projects provides a general text direction classification model, and you can download it in [Google Drive](https://drive.google.com/file/d/1hHZXOUWgvYwVZIa2Joj6XvyNd_isFt5N/view?usp=sharing) or [OneDrive](https://zjueducn-my.sharepoint.com/:u:/g/personal/ljw0608_mail_zju_edu_cn/EV-ZGHkVciVAoBhTjMGGWYEBfBKzNAIheiUwmvlQccE02Q?e=l2bg2f).

## 🍰 Pre-Requisites <a name = "pre-requisites"></a>

* Linux
* Python 2.7 or Python 3.6
* Pytorch>=1.1.0
* torchvision>=0.3.0
* opencv>=3.4.0
* Pillow>=6.2.0
* pyaml
* yacs
* tqdm
* coloredlogs

While not required, for optimal performance it is highly recommended to run the code using a CUDA enabled GPU.

## 📙 Usage <a name="usage"></a>

This part provides basic tutorials about the usage of code. For installation requirements, please see [Pre-Requisites](#pre-requisites).

### Preparing datasets <a name="preparing_datasets"></a>
You can make datasets using `utils/make_datasets.py` , and split training and validation datasets using `utils/split_dataset.py`.

### Training <a name="training"></a>
All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `OUTPUT_DIR` in the config file.

We provide default values for all training parameters in `configs/baseline.yaml`, you can also change them in the training config.

```bash
python train.py --config_file=configs/baseline.yaml --gpu=0,1
```

Before training, you must specify the config file for training by `--config_file`. Optional arguments are:
  - `--gpu`: gpu list to use for training, for example 1,2,3.
  - `--initial_checkpoint`: initialize model from this checkpoint (default: none).
  - `--resume`: resume full model and optimizer state from checkpoint (default: none).
  - `--no_resume_opt`: prevent resume of optimizer state when resuming model.
  - `--start_epoch`: manual epoch number (useful on restarts).

### Testing <a name="testing"></a>
We provide demo scripts(`demo.py`) to evaluate pretrained model, and also some high-level apis for easier integration to other projects. 

This projects provides a general text direction classification model, and you can download it in [Google Drive](https://drive.google.com/file/d/1hHZXOUWgvYwVZIa2Joj6XvyNd_isFt5N/view?usp=sharing) or [OneDrive](https://zjueducn-my.sharepoint.com/:u:/g/personal/ljw0608_mail_zju_edu_cn/EV-ZGHkVciVAoBhTjMGGWYEBfBKzNAIheiUwmvlQccE02Q?e=l2bg2f). You can view pictures for testing in `experiments/test_images`.

```bash
python demo.py --model_path=checkpoints/model_best.pth.tar --images_dir=experiments/test_images --output_dir=experiments/cls_output
```
Before running demo, you must specify the pretrained model by `--model_path`, path of images for testing by `--images_dir` and output path to saved predicting results by `--output_dir`.

## 🔨 TODO <a name = "todo"></a>
A number of additions planned in the future for various projects, including:
- ❎ Improving data load speed during training.
- ❎ Backbone [DPN](https://arxiv.org/pdf/1707.01629.pdf).

## 👬 Acknowledgements <a name = "acknowledgement"></a>

- [rwightman](https://github.com/insightcs/pytorch-image-models)
- [kylelobo](https://github.com/kylelobo/The-Documentation-Compendium)
