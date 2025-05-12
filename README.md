# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.

### DESIGN STEPS:
## STEP 1: 
Import necessary libraries including PyTorch, torchvision, and matplotlib.

## STEP 2: 
Load the MNIST dataset with transforms to convert images to tensors.

## STEP 3: 
Add Gaussian noise to training and testing images using a custom function.

## STEP 4: 
Define the architecture of a convolutional autoencoder:

Encoder: Conv2D layers with ReLU + MaxPool

Decoder: ConvTranspose2D layers with ReLU/Sigmoid

## STEP 5: 
Initialize model, define loss function (MSE) and optimizer (Adam).

## STEP 6: 
Train the model using noisy images as input and original images as target.

## STEP 7: 
Visualize and compare original, noisy, and denoised images.

## PROGRAM
### Name: ANU VARSHINI M B
### Register Number: 212223240010


Include your code here

## OUTPUT

### Model Summary

![image](https://github.com/user-attachments/assets/f1fb5a76-a90e-4684-9cde-5320a6edddf7)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/user-attachments/assets/9d8c5036-43bd-4a68-8b22-3f264ffcb8d8)




## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
