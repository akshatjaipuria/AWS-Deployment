try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from PIL import Image
import numpy as np

import boto3
import os
import io
import base64
import json
from requests_toolbelt.multipart import decoder
print("Import end...")

# Configuration
CROP_SIZE = 88
UPSCALE_FACTOR = 2

S3_BUCKET = 'motley-superres-flyingobjects'
MODEL_PATH = 'srgan_flyingobjects.pt'

print("Downloading model...")

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        netG = torch.jit.load(bytestream)
        print("Model loaded")
except Exception as e:
    print(repr(e))
    raise(e)

print("Model download completed.")

# Provide valid crop size. Returns multiple of upscale_factor
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# Provides set of Low resolution images, High resolution images restored by Bicubic interpolation and the actual input HR image (centercropped.)
def image_transform_LR_HR_Restored(HR_image, crop_size, upscale_factor):
    LR_scale = Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    HR_scale = Resize(crop_size, interpolation=Image.BICUBIC)

    HR_image = CenterCrop(crop_size)(HR_image)
    LR_image = LR_scale(HR_image)
    HR_restored_image = HR_scale(LR_image)
    return ToTensor()(LR_image), ToTensor()(HR_restored_image), ToTensor()(HR_image)


def super_resolution_flyingobject(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print("Image body in classify_image loaded")
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        input_image = Image.open(io.BytesIO(picture.content))
        w, h = input_image.size
        crop_size = calculate_valid_crop_size(min(w, h), UPSCALE_FACTOR)
        val_lr, val_hr_restored, val_hr = image_transform_LR_HR_Restored(input_image, crop_size, UPSCALE_FACTOR)

        # Image inference
        with torch.no_grad():
            val_lr = val_lr.to(torch.device("cpu")).unsqueeze(0)
            sr = netG(val_lr)
        
        # Prepare val_hr_restored, val_hr, sr- bring it to (H,W,C) format and numpy format.
        val_hr_restored = val_hr_restored.permute(1,2,0).numpy()
        val_hr_restored = Image.fromarray((val_hr_restored*255).astype(np.uint8))

        val_hr = val_hr.permute(1,2,0).numpy()
        val_hr = Image.fromarray((val_hr*255).astype(np.uint8))

        sr = sr.detach().squeeze().permute(1,2,0).numpy()
        sr = Image.fromarray((sr*255).astype(np.uint8))

        #Prepare output to return
        print("Loading output to buffer")
        buffer_hr_restored = io.BytesIO()
        val_hr_restored.save(buffer_hr_restored, format="JPEG")
        hr_restored_bytes = base64.b64encode(buffer_hr_restored.getvalue())

        buffer_hr = io.BytesIO()
        val_hr.save(buffer_hr, format="JPEG")
        hr_bytes = base64.b64encode(buffer_hr.getvalue())

        buffer_sr = io.BytesIO()
        sr.save(buffer_sr, format="JPEG")
        sr_bytes = base64.b64encode(buffer_sr.getvalue())

        return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                "body": json.dumps({'file': filename.replace('"', ''),'hr_restored': hr_restored_bytes.decode('ascii'),
                            'hr': hr_bytes.decode('ascii'), 'sr': sr_bytes.decode('ascii')})
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


    
