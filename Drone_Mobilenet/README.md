# Training Mobilenet on Custom Drone Dataset

The objective of the CNN model is to train MobileNet v2 on custom dataset of flying obects consisting of small quadcopters, large quadcopters, Winged drones and flying birds. The model is trained by transfer learning. Also, the trained model is deployed on AWS Lambda.

**HIGHLIGHTS:**

- Model: MobilenetV2
- Total no. of parameters: 2,228,996
- Loss: CrossEntropyLoss()
- Optimizer: SGD
- Scheduler: ReduceLROnPlateau
- Final Accuracy: 88.74%
- Batch size: 128
- Epochs: 20 

<h3>Dataset Stats</h3>
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

<h3>Data Preparation</h3>

- The collected dataset had invalid files such as .txt and webpages. The invalid files have been removed. Also, the dataset contained duplicate images which too have been removed. All this data was stored in 4 different .zip files, which were named in a particular format eg: "flying_birds.zip" , "winged_drones.zip"

- **Resizing Strategy** - The dataset collected contains images of resolution varying from 130x130 pixels to 6000x4000 pixels. So, to reduce the image pre-processing during dataload, the dataset has been resized to 224 x 224 resolution while maintaining the aspect ratio. For cases where the dimensions of the images are less than 224, padding has been added to side.

- The prepared dataset has been split into train:test::70:30 ratio

![Link to the code]

Code for the split:
``` 
for num, image in enumerate(test_imgs):
        file = image.split('/')[-1].split('\\')[-1]
        dest = test_dest+str(num+1)+".jpg"
        resize_save(image, dest) 
```


<h3>Model</h3>

Pre-trained MobileNet v2 model has been considered for this dataset. The pre-trained model is designed for 1000 class predictions. Since, the dataset requires 4 class predictions, the last layer of the model is modified as follows-

```
model = mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, 4)
```

**NOTE:**

Versions of torch and torchvision in colab and AWS Lambda is incompatible. For this reason, model is trained and deployed on torch, torchvision version 1.5.0 and 0.6.0 respectively.

```
!pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html 
```


<h3>Image Gallery</h3>
1) Misclassified Images of Flying Birds:

![Misclassified flying birds](images/fb.jpg)

2) Misclassified Images of Large QuadCopters:

![Misclassified large Quad](images/lq.jpg)

3) Misclassified Images of Small QuadCopters:

![Misclassified small Quad](images/sq.jpg)

4) Misclassified Images of Winged Drones:

![Misclassified winged drones](images/wd.jpg)

5) Accuracy Curve:

![acc curve](images/acc.jpg)

6) Loss Curve:

![Loss curve](images/loss.jpg)
