# Description
```
Title: Cat the Flag
Author:
```
Provided files
- X_test.pkl
- X.pkl
- y.okl

# Overview
This challenge provided the flag in the form of X_test.pkl, a set of 288 images encoded with the python pickle library. The images are of cats and dogs and the flag is encoded as the 288 classifications with 0 for cats and 1 for dogs.

A training dataset of 648 images is provided in X.pkl with labels provided in y.pkl such that for any X.pkl[i], y.pkl[i] is 0 if X.pkl[i] is a cat and 1 if X.pkl[i] is a dog.

Out solution was to train an image classifier that fit the training data with 99.5% accuracy then use it to classify the images in X_test.pkl. This generates an 288 floating point vales that we then rounded to the nearest integer and converted into characters in 8 bit blocks.

The challenge also included a website that allowed .h5 compressed model file uploads but we believe it was just a distraction