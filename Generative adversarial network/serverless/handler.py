try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import boto3
import os
import io
import base64
import json
from requests_toolbelt.multipart import decoder
print("Import end...")

S3_BUCKET = 'motley-dcgan-indiancars'
MODEL_PATH = 'DCGAN_IndianCars.pt'

print("Downloading model...")

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model loaded")
except Exception as e:
    print(repr(e))
    raise(e)

print("Model download completed.")

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)

def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def dcgan_indiancars_generate(event, context):
    try:
        fixed_noise = torch.randn(1, 100, 1, 1)
        img_verify = netG(fixed_noise.detach())
        img_verify = denormalize(img_verify, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        print("Shape of generated image-", img_verify.shape)
        img_verify = img_verify.squeeze().permute(1,2,0).numpy()

        img_verify = Image.fromarray(img_verify.astype(np.uint8))

        #Prepare output to return
        print("Loading output to buffer")
        buffer = io.BytesIO()
        img_verify.save(buffer, format="JPEG")
        img_verify_bytes = base64.b64encode(buffer.getvalue())

        return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                "body": json.dumps({'data': img_verify_bytes.decode('ascii')})
            }
    
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }

        


    