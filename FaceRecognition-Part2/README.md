
# Face Recognition

The objective of this assignment is to finetune facial recognition classfier on LFW dataset along with some custom data

<h2>Features</h2>

1. Model: Inception Resnet V1
2. Dataset: LFW dataset along with 10 facial images of 10 famous people (sports players, actors)
3. Total no. of parameters: 2,37,99,660
4. Loss: CrossEntropyLoss()
5. Optimizer: SGD
6. Scheduler: StepLR
7. Final Accuracy: 97%
8. Batch size: 8
9. Epochs: 20

<h2>Dataset Stats</h2>
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>No of Images</th>
  </tr>
</thead>
  <tbody>
  <tr>
    <td>LFW Dataset<br></td>
    <td>6733</td>
  </tr>
  <tr>
    <td>Custom Dataset<br></td>
    <td>100</td>
  </tr>
  </tbody>
</table>

<h2>Data Preparation</h2>

- The LFW dataset contains about 5760 folders with images of various famous personalities. Each folder has different number of images varying from 1 to 50 images. For our       training we choose all the folders having greater than 4 images.
  Link to LFW dataset: http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
- The custom dataset we created 10 folders with images of various famous personalities. Each folder has exactly 10 images in it.
  Link to custom dataset: https://drive.google.com/drive/folders/1GQNfGR63a3QzND5jx-8zXdkcjd7fyK7s?usp=sharing
- The images in the above datasets are mostly facial images and each of the images had to be aligned to get the frontal alignment of each face.
- Repetition of folders had to be taken care of and no duplicates are maintained.
- The above prepared data is then split into 70:30:: train:validation sets.

<h2>Model</h2>

- Inception Resnet V1: The Inception Resnet V1 model is pretrained on VGGFace2 where VGGFace2 is a large-scale face recognition dataset developed from Google image searches and “have large variations in pose, age, illumination, ethnicity and profession.”
- Each layer’s weights in the model have an attribute called requires_grad that can be set to True or False. When finetuning the network we freeze all of the layers up through the last convolutional block by setting the requires_grad attributes to False and then only update the weights on the remaining layers.

**NOTE:**

Versions of torch and torchvision in colab and AWS Lambda is incompatible. For this reason, model is trained and deployed on torch, torchvision version 1.5.0 and 0.6.0 respectively.

```
!pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html 
```

<h2>Run Results</h2>
<h3>Model Prediction</h3>
Shared below is a Facial recognition prediction for an input image-
<TABLE>
  <TR>
    <TH>Input Image</TH>
    <TH>FR Prediction</TH>
  </TR>
   <TR>
      <TD><img src="https://github.com/akshatjaipuria/AWS-Deployment/blob/master/FaceRecognition-Part2/images/Rohit_Sharma_0005.jpg" alt="input_image"
	title="inp_img" width="300" height="300" /></TD>
      <TD>Prediction ID: 499<br>
     Prediction name: Rohit Sharma</TD>
   </TR>
</TABLE>

<h3>Training Loss Trend</h3>

![](https://github.com/akshatjaipuria/AWS-Deployment/blob/master/FaceRecognition-Part2/images/LossVsIterations.jpg)
