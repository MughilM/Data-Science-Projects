# Data Science Projects
This repository contains a group of data science and machine learning projects.
The backbone of all the projects use Hydra, PyTorch, and PyTorch Lightning.
They follow the great template [by ashleve](https://github.com/ashleve/lightning-hydra-template/tree/main).
Some of these projects are also tied to Kaggle competitions, past and current.
The following projects have been coded in this repo.
Finally, the model runs are optioned to publish and output to [Weights and Biases](https://wandb.ai/site)

### Histopathologic Cancer Detection
- [Kaggle competition link](https://www.kaggle.com/c/histopathologic-cancer-detection)
- Main model: ResNet
- This model had a introductory goal. It was to make sure that all the parts of the template were working corerctly.


### TGS Salt Prediction and Oxford Pet III
- Main model backbones: UNet with MobileNet
- This model was used to practice building an image segmentation model using UNet. The UNet structure was built from 
  MobileNet.

### Google Research Contrail Prediction
- [Kaggle competition link](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/overview)
- Main Model: A binary prediction model paired with UNet MobileNet
- This Kaggle competition looks at attempting to predict contrails produced by airplanes in the atmosphere. The given
  data contains many infrared bands of the patch of the sky, which may or may not contain contrails in them. The goal is
  to produce contrail predictions for each pixel in the image - a segmentation problem.

### Stock prediction from candlestick plots
- This project takes a spin on the overused stock prediction problem. Instead of looking at price directly, we will look
  at candlestick plots produced by the corresponding prices. Thus, we treat this as a image recognition problem that 
  operates on the candlestick plots directly
- The data used is proprietary minute-by-minute data of open, close, low, high, and volume. It operates on the NASDAQ
  100 stock from around the early 2000s to August 2018. In total, 103 stocks were operated on, and the resulting images
  comprise 350k files and a bit less than 7 GB of space.