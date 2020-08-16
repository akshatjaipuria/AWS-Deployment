# Face Recognition

The objective of this assignment is to create a static webpage on AWS with the following features-

1. ResNet 34 pre-trained model for object classification.
2. MobileNet v2 model trained on flying objects to classify Flying birds, Winged Drones, Small and Large quadcopters.
3. Face alignment model based on dlib toolkit.
4. Face swap model based on dlib toolkit.

## Webpage features

The webpage created and hosted on AWS contains the following features-

### Pre-trained ResNet-34 model

One of the feature is ResNet-34 model, which is trained on Imagenet dataset. 


### MobileNet v2 model

Second feature is MobileNet v2 trained on custom dataset to classify flying objects. 


### Face Alignment model

The third feature is Face Alignment. Face alignment is an important pre-step for facial recognition. Based on the facial landmarks (position of the eye, nose), Face Alignment tries to obtain a normalized rotation, translation and scale representation of the face. The model uses dlib's pre-trained landmarks model shape_predictor_68_face_landmarks.dat. Based on the landmark points detected by dlib's model, the input image is normalized such that left corner of the left eye and right corner of the right eye are at specified position in the output image.   


### Face Swap model

The fourth feature is an attempt to swap face of the source image onto target input image. Using dlib's landmark model- shape_predictor_68_face_landmarks.dat, landmark points are detected on the input target image. Swapped face image is found by applying affine transform and seamless cloning on triangulation points obtained from the landmark points.

## Results

### Object classification using ResNet-34 model.


### Flying object classification using Mobilenet v2 model.


### Face Alignment


### Face Swap
