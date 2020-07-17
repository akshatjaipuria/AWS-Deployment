# Deploying over AWS

This is to deploy MobileNet v2 model on AWS Lambda.

Serverless web framework is used to build the application. The application is available as an API service hosted on AWS Lambda.

## Project Setup

This project is created on Oracle VM box with Ubuntu OS.

### Install Docker 

1. Download and install the following Docker files - 

```
- docker-ce_19.03.9~3-0~ubuntu-bionic_amd64.deb
- docker-ce-cli_19.03.9~3-0~ubuntu-bionic_amd64.deb
- containerd.io_1.2.13-2_amd64.deb

```

Provide necessary permission to docker-

```
sudo chmod 666 /var/run/docker.sock
```

2. Install Node.js using the following commands-

```
curl -sL https://deb.nodesource.com/setup_10.x (Links to an external site.) -o nodesource_setup.sh
sudo bash nodesource_setup.sh
sudo apt-get install -y nodejs
```

3. Install serverless framework using command- 

```
sudo npm install -g serverless
```

4. Configure AWS

Create a new user using Identity and Access Management app on AWS console. Collect the Access key and secret key which is required to configure credentials for Serverless.

Configuring the credentials for Serverless can be done by-

```
sls config credentials --provider aws --key <Access key> --secret <Secret key>
```

5. Download pre-trained Mobilenet V2 model

```
import torch

from torchvision.models import mobilenet
model = mobilenet.mobilenet_v2(pretrained=True)

model.eval()

traced_model = torch.jit.trace(model, trorch.randn(1,3,224,224))
traced_model.save('mobilenetv2.pt')
```

## Results

DNN model can be tested to classify images using any REST client such as Insomnia. Few such iamges classified by the model are shared below.

Please note, the first API call might timeout as the service requires cold start. Please resenf the API request again.

1. Test 1

Image to be classified
![](https://github.com/akshatjaipuria/AWS-Deployment/blob/master/images/Yellow-Labrador-Retriever.jpg)

Model result on Insomnia
![](https://github.com/akshatjaipuria/AWS-Deployment/blob/master/images/response_dog.png)


2. Test 2

Image to be classified

![](https://github.com/akshatjaipuria/AWS-Deployment/blob/master/images/car.jpg)

Model result on Insomnia
![](https://github.com/akshatjaipuria/AWS-Deployment/blob/master/images/response_car.png)




