# Generative Adversarial Networks

The objective here is to train a Generative Adversarial Network to generate Indian cars. The model generates new cars from pictures of real Indian cars.
Further, the trained model is deployed on AWS Lambda.

## Model hyperparameters

* Optimizer : Adam
* Loss function : Binary Cross Entropy loss
* Learning Rate : 0.0002
* Batch size : 128
* Epochs : 200

## Model Architecture

GAN model consists of two parts-

1. Generator - Generator maps the latent vector space to data-space. It generates RGB image, equal to our training image's dimension, from the latent vector. Strided transpose convolution performs the upscaling from the feature maps. Each convolution layer is paired with 2d batch norm layer and relu activation. Output of the generator is fed through a tanh function to return it to the input data range of [-1, 1].

2. Discriminator - Discriminator performs the binary classification of the input image into real/fake classes. The input image is processed by series of Conv2d, BatchNorm and LeakyRelu activation. Based on the DCGAN paper, strided convolution is used for downsampling instead of pooling. This is to let the network learn its own pooling function.

## Results

### Real and Generated images

Collage of real images and generated images is shared below.
![](https://github.com/Shashank-Holla/TSAI-EVA4-Phase2/blob/master/06%20-%20Generative%20Adversarial%20Networks/results/realfake.jpg)

### Generated images


### Discriminator/Generator loss during training

![](https://github.com/Shashank-Holla/TSAI-EVA4-Phase2/blob/master/06%20-%20Generative%20Adversarial%20Networks/results/D%26G_loss.jpg)

### Average Discriminator output for real and generated images

Shared below is the average output of the discriminator in classifying real and fake images. 
D(x) is the average discriminator output on classifying training images. Initially the average output is close to 1 as all the training images are correctly classified as real. As the generator trains and produces fake images which is close to the training image distribution, the average output converges to 0.5.

D(G(z)) is the average discriminator output in classifying the fake images. Average output is close to 0 when the training begins and converges to 0.5 as the generator trains.

![](https://github.com/Shashank-Holla/TSAI-EVA4-Phase2/blob/master/06%20-%20Generative%20Adversarial%20Networks/results/avgOP_realfake.jpg)






