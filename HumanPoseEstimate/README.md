# Human Pose Estimation

The objective of this assignment is to implement Simple Baseline Pose Estimation to estimate Human pose in an image and deploy the model on AWS.


## Results

Shared below is the Human Pose estimate from the model.

<TABLE>
  <TR>
    <TH>Input Image</TH>
    <TH>Human Pose Estimate</TH>
  </TR>
   <TR>
      <TD><img src="https://github.com/akshatjaipuria/AWS-Deployment/blob/master/HumanPoseEstimate/images/charlie_sheen.jpg" alt="input_image"
	title="inp_img" width="300" height="400" /></TD>
      <TD><img src="https://github.com/akshatjaipuria/AWS-Deployment/blob/master/HumanPoseEstimate/images/charlie_with_pose.jpg" alt="input_image"
	title="pose_img" width="300" height="400" /></TD>
   </TR>
</TABLE>

## Some Fun with Gestures

We have used this Human Pose Estimation mapping to track the gesture of the person and invoke certain functionalities according to the movement of the body parts. For now, your implementation supports playing and stopping music. To play the song, the person needs to raise his/her left hand once, and the music starts. Similarly, to stop the misic the person has to raise the right hand up. THe demo video can be found on the following link.
[Demo](https://drive.google.com/file/d/1h3Ka9SY4qQrd-6-rsW3T6xyQgE1gVUdE/view?usp=sharing)
