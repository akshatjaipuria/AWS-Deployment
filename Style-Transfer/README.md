# Neural Style Transfer

Neural Style Transfer is used to compose an image in the style of another image. It takes an input image and reproduces it with a new artistic style. The algorithm takes three images, an input image, a content-image, and a style-image, and changes the input to resemble the content of the content-image and the artistic style of the style-image.

<h3>Description</h3>

- Model: VGG19 pre-trained model 
- Loss functions: For Content Loss: 'conv_4' layer and for Style Loss: ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] layers 
- num-steps used: 250

<h3>Results</h3>
<TABLE>
  <TR>
    <TH>Style Image</TH>
    <TH>Content Image</TH>
    <TH>Output Image</TH>
  </TR>
   <TR>
      <TD><img src="https://github.com/akshatjaipuria/AWS-Deployment/blob/master/Style-Transfer/Images/image1.jpg" alt="style_image"
	title="Style Image" width="256" height="256" /></TD>
      <TD><img src="https://github.com/akshatjaipuria/AWS-Deployment/blob/master/Style-Transfer/Images/download1.jpg" alt="content_image"
	title="Content Image" width="256" height="256" /></TD>
      <TD><img src="https://github.com/akshatjaipuria/AWS-Deployment/blob/master/Style-Transfer/Images/download.png" alt="output_image"
	 width="256" height="256" /></TD>
   </TR>
</TABLE>
