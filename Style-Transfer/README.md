# Neural Style Transfer

Neural Style Transfer is used to compose an image in the style of another image. It takes an input image and reproduces it with a new artistic style. The algorithm takes three images, an input image, a content-image, and a style-image, and changes the input to resemble the content of the content-image and the artistic style of the style-image.

<h3>Description</h3>
- Model: VGG19 pre-trained model 
- Loss functions: For Content Loss: 'conv_4' layer and for Style Loss: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] layers 
- num-steps used: 250

<h3>Results</h3>
