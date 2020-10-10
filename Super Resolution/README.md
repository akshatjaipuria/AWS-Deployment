# Image Super Resolution

The objective here is to train SR-GAN (Super Resolution- Generative Adversial Network) on the flying objects dataset to provide Super resolution of the flying object images. The model is then deployed on AWS.

## Model Hyperparameters

- Model : SRGAN
- Batch Size : 64
- Upscale Factor : 2
- Epochs : 10
- Loss function : Content Loss + Adversial Loss
- Optimizer : Adam

## Dataset Details
There are four different classes of images in the dataset:

<table>
<thead>
  <tr>
    <th>Image Class</th>
    <th>No of Images</th>
    <th>Mean</th>
    <th>Std. Dev</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Flying Birds<br></td>
    <td>7781</td>
    <td rowspan="4">[0.3749, 0.4123, 0.4352]</td>
    <td rowspan="4">[0.3326, 0.3393, 0.3740]</td>
  </tr>
  <tr>
    <td>Large QuadCopters</td>
    <td>3609</td>
  </tr>
  <tr>
    <td>Small QuadCopters</td>
    <td>3957</td>
  </tr>
  <tr>
    <td>Winged Drones</td>
    <td>3163</td>
  </tr>
</tbody>
</table>

## Model Architecture

GAN based model includes deep residual network for Generator network which is able to recover photo-realistic textures from heavily downsampled images. 

Model uses perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes the solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, the content loss compares deep features extracted from Super-resolved and High Resolution images with a pre-trained VGG network.

## Results

### Model results
Below are the results from SR-GAN. It consists of HR restored image (restored by Bicubic interpolation of the Low resolution image), actual input HR image (centercropped) and the Super Resolution result from the SR-GAN model.

![](results/sr_result.png)

### Generator/Discriminator Loss Trend versus Epoch

Shared below is the Generator/Discriminator Loss trend during training.

![](results/loss_vs_epoch.png)

### PSNR and SSIM trend

Shared below are the Peak Signal to Noise Ratio and Structural Similarity Index trend during training.

![](results/psnr_ssim.png)

