# Convolutional-Neural-Network
Learn Convolutional Neural Network from basic and its implementation

## Table of contents
- What is CNN ?
- Why should we use CNN ?
- Few Definitions
- Layers in CNN

## 1. What is CNN ?
Computer vision is evolving rapidly day-by-day. Its one of the reason is deep learning. When we talk about computer vision, a term convolutional neural network( abbreviated as CNN) comes in our mind because CNN is heavily used here. Examples of CNN in computer vision are face recognition, image classification etc. It is similar to the basic neural network. CNN also have learnable parameter like neural network i.e, weights, biases etc.

## 2. Why should we use CNN ?
### Problem with Feedforward Neural Network
Suppose you are working with MNIST dataset, you know each image in MNIST is 28 x 28 x 1(black & white image contains only 1 channel). Total number of neurons in input layer will 28 x 28 = 784, this can be manageable. What if the size of image is 1000 x 1000 which means you need 10⁶ neurons in input layer. Oh! This seems a huge number of neurons are required for operation. It is computationally ineffective right. So here comes Convolutional Neural Network or CNN. In simple word what CNN does is, it extract the feature of image and convert it into lower dimension without loosing its characteristics. In the following example you can see that initial the size of the image is 224 x 224 x 3. If you proceed without convolution then you need 224 x 224 x 3 = 100, 352 numbers of neurons in input layer but after applying convolution you input tensor dimension is reduced to 1 x 1 x 1000. It means you only need 1000 neurons in first layer of feedforward neural network.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*V6hPq-srR86AIWYrgFYLfA.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*V6hPq-srR86AIWYrgFYLfA.png">
  <img alt="Simple CNN structure" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*V6hPq-srR86AIWYrgFYLfA.png">
</picture>

## 3. Few Definitions
There are few definitions you should know before understanding CNN
### 3.1 Image Representation
Thinking about images, its easy to understand that it has a height and width, so it would make sense to represent the information contained in it with a two dimensional structure (a matrix) until you remember that images have colors, and to add information about the colors, we need another dimension, and that is when Tensors become particularly helpful.

Images are encoded into color channels, the image data is represented into each color intensity in a color channel at a given point, the most common one being RGB, which means Red, Blue and Green. The information contained into an image is the intensity of each channel color into the width and height of the image, just like this.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*125JKUHmij9bzKcREpq9ew.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*125JKUHmij9bzKcREpq9ew.png">
  <img alt="RGB representation of a image" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*125JKUHmij9bzKcREpq9ew.png">
</picture>

So the intensity of the red channel at each point with width and height can be represented into a matrix, the same goes for the blue and green channels, so we end up having three matrices, and when these are combined they form a tensor.

### 3.2 Edge Detection
Every image has vertical and horizontal edges which actually combining to form a image. Convolution operation is used with some filters for detecting edges. Suppose you have gray scale image with dimension 6 x 6 and filter of dimension 3 x 3(say). When 6 x 6 grey scale image convolve with 3 x 3 filter, we get 4 x 4 image. First of all 3 x 3 filter matrix get multiplied with first 3 x 3 size of our grey scale image, then we shift one column right up to end , after that we shift one row and so on.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*Ekm4QJ1rHE-bJbQllBWLPA.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*Ekm4QJ1rHE-bJbQllBWLPA.png">
  <img alt="Convolution operation" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*Ekm4QJ1rHE-bJbQllBWLPA.png">
</picture>

The convolution operation can be visualized in the following way. Here our image dimension is 4 x 4 and filter is 3 x 3, hence we are getting output after convolution is 2 x 2.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:488/format:webp/1*4h_J0Zpx93_sFHKxWUoHAw.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:488/format:webp/1*4h_J0Zpx93_sFHKxWUoHAw.gif">
  <img alt="Visualization of convolution" src="https://miro.medium.com/v2/resize:fit:488/format:webp/1*4h_J0Zpx93_sFHKxWUoHAw.gif">
</picture>

If we have N x N image size and F x F filter size then after convolution result will be

> [!NOTE]
> (N x N) * (F x F) = (N-F+1)x(N-F+1)(Apply this for above case)

### 3.3 Stride and Padding
Stride denotes how many steps we are moving in each steps in convolution.By default it is one.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*g0OmDI1w9KqN7Rpw6Qo8Xg@2x.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*g0OmDI1w9KqN7Rpw6Qo8Xg@2x.gif">
  <img alt="Convolution with Stride 1" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*g0OmDI1w9KqN7Rpw6Qo8Xg@2x.gif">
</picture>

We can observe that the size of output is smaller that input. To maintain the dimension of output as in input , we use padding. Padding is a process of adding zeros to the input matrix symmetrically. In the following example,the extra grey blocks denote the padding. It is used to make the dimension of output same as input.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*17TNPi4m0pBqOCGrXzU27w.gif">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*17TNPi4m0pBqOCGrXzU27w.gif">
  <img alt="Stride 1 with Padding 1" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*17TNPi4m0pBqOCGrXzU27w.gif">
</picture>

Let say ‘p’ is the padding
Initially(without padding)
(N x N) * (F x F) = (N-F+1)x(N-F+1)---(1)
After applying padding:
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*8VwvmOay_k_0MLTrwqQtEg.png">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:640/format:webp/1*8VwvmOay_k_0MLTrwqQtEg.png">
  <img alt="After applying padding" src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*8VwvmOay_k_0MLTrwqQtEg.png">
</picture>

If we apply filter F x F in (N+2p) x (N+2p) input matrix with padding, then we will get output matrix dimension (N+2p-F+1) x (N+2p-F+1). As we know that after applying padding we will get the same dimension as original input dimension (N x N). Hence we have,
> [!NOTE]
> (N+2p-F+1)x(N+2p-F+1) equivalent to NxN
> N+2p-F+1 = N ---(2)
> p = (F-1)/2 ---(3)

The equation (3) clearly shows that Padding depends on the dimension of filter.

## 4. Layers in CNN
There are six different layers in CNN
- Input layer
- Convo layer (Convo + ReLU)
- Pooling layer
- Fully connected(FC) layer
- Softmax/logistic layer
- Output layer

### 4.1 Input Layer
Input layer in CNN should contain image data. Image data is represented by three dimensional matrix as we saw earlier. You need to reshape it into a single column. Suppose you have image of dimension 28 x 28 =784, you need to convert it into 784 x 1 before feeding into input. If you have “m” training examples then dimension of input will be (784, m).

### 4.2. Convo Layer
Convo layer is sometimes called feature extractor layer because features of the image are get extracted within this layer. First of all, a part of image is connected to Convo layer to perform convolution operation as we saw earlier and calculating the dot product between receptive field(it is a local region of the input image that has the same size as that of filter) and the filter. Result of the operation is single integer of the output volume. Then we slide the filter over the next receptive field of the same input image by a Stride and do the same operation again. We will repeat the same process again and again until we go through the whole image. The output will be the input for the next layer.

Convo layer also contains ReLU activation to make all negative value to zero.

### 4.3. Pooling Layer

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*GksqN5XY8HPpIddm5wzm7A.jpeg">
  <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/v2/resize:fit:720/format:webp/1*GksqN5XY8HPpIddm5wzm7A.jpeg">
  <img alt="Pooling in CNN" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*GksqN5XY8HPpIddm5wzm7A.jpeg">
</picture>

Pooling layer is used to reduce the spatial volume of input image after convolution. It is used between two convolution layer. If we apply FC after Convo layer without applying pooling or max pooling, then it will be computationally expensive and we don’t want it. So, the max pooling is only way to reduce the spatial volume of input image. In the above example, we have applied max pooling in single depth slice with Stride of 2. You can observe the 4 x 4 dimension input is reduce to 2 x 2 dimension.

There is no parameter in pooling layer but it has two hyperparameters — Filter(F) and Stride(S).
In general, if we have input dimension W1 x H1 x D1, then
W2 = (W1−F)/S+1
H2 = (H1−F)/S+1
D2 = D1
Where W2, H2 and D2 are the width, height and depth of output.

### 4.4. Fully Connected Layer(FC)
Fully connected layer involves weights, biases, and neurons. It connects neurons in one layer to neurons in another layer. It is used to classify images between different category by training.

### 4.5. Softmax / Logistic Layer
Softmax or Logistic layer is the last layer of CNN. It resides at the end of FC layer. Logistic is used for binary classification and softmax is for multi-classification.

### 4.6. Output Layer
Output layer contains the label which is in the form of one-hot encoded.

Now you have a good understanding of CNN. Let’s implement a CNN.

