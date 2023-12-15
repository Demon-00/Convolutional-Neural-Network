# Convolutional-Neural-Network
Learn Convolutional Neural Network from basic and its implementation

## Table of contents
- [X] What is CNN ?
- [ ] Why should we use CNN ?
- [ ] Few Definitions
- [ ] Layers in CNN
- [ ] Keras Implementation

## 1. What is CNN ?
Computer vision is evolving rapidly day-by-day. Its one of the reason is deep learning. When we talk about computer vision, a term convolutional neural network( abbreviated as CNN) comes in our mind because CNN is heavily used here. Examples of CNN in computer vision are face recognition, image classification etc. It is similar to the basic neural network. CNN also have learnable parameter like neural network i.e, weights, biases etc.

## 2. Why should we use CNN ?
### Problem with Feedforward Neural Network
Suppose you are working with MNIST dataset, you know each image in MNIST is 28 x 28 x 1(black & white image contains only 1 channel). Total number of neurons in input layer will 28 x 28 = 784, this can be manageable. What if the size of image is 1000 x 1000 which means you need 10⁶ neurons in input layer. Oh! This seems a huge number of neurons are required for operation. It is computationally ineffective right. So here comes Convolutional Neural Network or CNN. In simple word what CNN does is, it extract the feature of image and convert it into lower dimension without loosing its characteristics. In the following example you can see that initial the size of the image is 224 x 224 x 3. If you proceed without convolution then you need 224 x 224 x 3 = 100, 352 numbers of neurons in input layer but after applying convolution you input tensor dimension is reduced to 1 x 1 x 1000. It means you only need 1000 neurons in first layer of feedforward neural network.

https://miro.medium.com/v2/resize:fit:720/format:webp/1*V6hPq-srR86AIWYrgFYLfA.png
