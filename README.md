# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.

### DESIGN STEPS:
## STEP 1:
Import necessary libraries such as NumPy, Matplotlib, TensorFlow/Keras.

## STEP 2:
Load and normalize the MNIST dataset. Add Gaussian noise to both training and test images.

## STEP 3:
Build a convolutional autoencoder model with the following structure:

Encoder: Conv2D + MaxPooling

Decoder: Conv2DTranspose or UpSampling + Conv2D

## STEP 4:
Compile the model using binary_crossentropy or mse loss and adam optimizer.

## STEP 5:
Train the model with noisy images as input and clean images as target.

## STEP 6:
Evaluate the model and visualize original, noisy, and denoised outputs.

## PROGRAM
### Name: ANU VARSHINI M B
### Register Number: 212223240010


Include your code here

## OUTPUT

### Model Summary

Include your model summary

### Original vs Noisy Vs Reconstructed Image

Include a few sample images here.



## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
