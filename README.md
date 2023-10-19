# Automatic Abnormality Detection in Chest Radiographs using Deep Convolutional Neural Network

## Overview
This repository contains code and resources for building a deep learning model to automatically detect abnormalities in chest radiographs. The model is based on a Convolutional Neural Network (CNN) architecture, and it is designed to assist healthcare professionals in identifying potential chest abnormalities, such as lung diseases, Covid-19 infections, e.t.c
## Features
* Pre-trained Models: We provide pre-trained CNN models that you can fine-tune on your own chest radiograph dataset, reducing the need for extensive data collection and annotation.

* Data Preprocessing: A set of data preprocessing scripts and functions to prepare and augment your chest radiograph dataset for training the CNN model.

* Model Training: Scripts and Jupyter notebooks to train, validate, and evaluate the CNN model, with customizable hyperparameters and settings.

* Inference: Tools for making predictions on new chest radiographs and visualizing the model's output, including detection of abnormal regions.

* Visualization: Tools for visualizing training and evaluation metrics, model architectures, and more.

* Integration: Instructions and code for integrating the trained model into your healthcare application or workflow.

## Prerequisites
Before you begin, ensure you have met the following requirements:

* Python 3.6 or higher
* PyTorch
* NumPy
* OpenCV
* Matplotlib
* Jupyter Notebook (for model training and visualization)
* A labeled chest radiograph dataset (you can use your own or obtain a publicly available dataset)

## Getting Started
1. Clone this repository to your local machine:
```
git clone https://github.com/olupy/abnormality-detection.git
```
```
cd abnormality-detection
```
2. Set up your environment and install the required dependencies:
```
pip install -r requirements.txt
```
3. Prepare your chest radiograph dataset. Ensure that you have labeled data with both normal and abnormal cases.

4. Modify the configuration files to specify the paths to your dataset and other settings.
5. Train the CNN model using the provided scripts or Jupyter notebooks:
```
jupyter notebook train.ipynb
```
6. Evaluate the model's performance on a validation set and fine-tune hyperparameters as needed.

7. Once the model is trained to your satisfaction, you can use it for inference and integrate it into your application or workflow.

## Model Evaluation
Evaluate the trained model's performance using various metrics and visualization tools provided in the repository.

## Inference
Use the trained model to make predictions on new chest radiographs and visualize the detected abnormal regions, if applicable.
