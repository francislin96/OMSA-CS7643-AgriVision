# OMSA-CS7643-AgriVision
## Project Summary
This is our group project repository for CS 7643 - Deep Learning at Georgia Tech. The main focus of this project is to develop several semantic segmentation models for an agricultural field image dataset using different semi-supervised deep learning techniques such as self training and co-training in order to better predict the test set over 9 different semantic classes.

The datasets used for this project come from [Agriculture Vision](https://www.agriculture-vision.com/). The first dataset, published in 2021, is fully labeled and comes prepartitioned into training, validation, and holdout test sets. Each image has RGB and NIR channels and is precropped to a dimension of 512x512. It has masks for each of the annotated classes in the list below.
* 0 - background
* 1 - double_plant
* 2 - drydown
* 3 - endrow
* 4 - nutrient_deficiency
* 5 - planter_skip
* 6 - water
* 7 - waterway
* 8 - weed_cluster

Additionally, we will use a new dataset published for the 2024 [Agriculture Vision challenge](https://www.agriculture-vision.com/agriculture-vision-2024/prize-challenge-2024). This includes 206, 32bit tiff files for the RGB-NIR channels but the size of the images are all around ~9000x9000 and are unnormalized. None of theses new images are labeled, and thus using a semi-supervised training scheme, we aim to push the limits of training a new segmentation model and evaluate it on the test set.

## Data Download
1. Clone this repo into your local machine.
```
$ git clone https://github.com/francislin96/OMSA-CS7643-AgriVision.git
```
2. Download the 2021 Agriculture Vision Dataset. Make sure that you have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed on your machine. Extract the `tar.gz` file and copy the contents directly to `./data/images_2021`
```
$ aws s3 cp s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised ./data/supervised --no-sign-request --recursive
$ tar -xzf ./data/supervised/Agriculture-Vision-2021.tar.gz
$ cp -r ./data/supervised/Agriculture-Vision-2021/* ./data/images_2021/
$ rm -rf ./data/supervised/Agriculture-Vision-2021/
```
3. Download the 2024 unlabeled Agriculture Vision Dataset. Agriculture Vision has it stored in DropBox for the time being so download the zipped file [here](https://www.dropbox.com/scl/fo/7yzzc8hqtvaki2y1md6h4/h?rlkey=su71dij6xfb964zfwe1d6kros&dl=0). Copy it into `./data/`, unzip, and move the contents into `./data/images_2024`
```
$ cp PATH_TO_DOWNLOAD_ZIP_HERE ./data/images_2024/
$ unzip ./data/images_2024
$ rm ./data/images_2024
```

## Environment Setup
1. Set up a virtual environment using the environment manager of your choice. I like using `pyenv` with `virtualenv` to easily manage Python versions and environments. Create a new environment, and set it in the root directory.
```
$ pyenv virtualenv 3.11.8 ENVIRONMENT_NAME
$ pyenv local ENVIRONMENT_NAME
```
2. Install all of the dependencies.
```
$ pip install -r requirements.txt
```

## Model Weights
Once we train a model, have the ability to download weights from AWS, Zenodo, or something like that. They will be too large to host on GitHub...
```
$ wget URL_TO_WEIGHTS.pth
```

## Inference Code or Django App Deployment
Use this section to explain how to run inference scripts and/or stand up the Django application. We can always try to have this deployed in a AWS Sagemaker serverless instance for easy inference but that might be overkill...