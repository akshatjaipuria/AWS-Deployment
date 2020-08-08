# Training Mobilenet on Custom Drone Dataset

NOTE: 
```
Older versions of torch and torchvision had to be installed in colab in order to deploy code to AWS Lambda
!pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html 
```
HIGHLIGHTS:

- Model: MobilenetV2
- Custome Dataset: 4 classes
- Total no. of parameters: 2,228,996
- Loss: CrossEntropyLoss()
- Optimizer: SGD
- Scheduler: ReduceLROnPlateau
- Final Accuracy: 88.74%

<h3>Dataset Stats</h3>
These are four different classes of images in the dataset:

1) "Flying Birds" - 7781 images
2) "Large QuadCopters" - 3609 images
3) "Small QuadCopters" - 3957 images
4) "Winged Drones" - 3163 images

<h3>Data Preparation</h3>

- All this data was stored in 4 different .zip files, which were named in a particular format eg: "flying_birds.zip" , "winged_drones.zip"
- The input images were all of different sizes and had to be transformed into 224x224 size images:

- Then the data was split into train:test::70:30 ratio

Code for the split:
``` 
for num, image in enumerate(test_imgs):
        file = image.split('/')[-1].split('\\')[-1]
        dest = test_dest+str(num+1)+".jpg"
        resize_save(image, dest) 
```
- The code after splitting into training and validation datasets are stored into 2 different folders: "train" and "val"
- These folders are loaded into colab during execution to save memory and time

<h3>Model</h3>
We are using the MobilenetV2 pretrained model, but modifying the last layer:

Total no. of parameters: 2,228,996
Final val accuracy: 88.74%

<h3>Image Galary</h3>
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
