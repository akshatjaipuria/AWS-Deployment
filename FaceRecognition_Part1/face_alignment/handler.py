try:
    import unzip_requirements
except ImportError:
    pass
import boto3
import os
import io
import json
import base64
import dlib
import cv2
import numpy as np
import faceBlendCommon as fbc
from PIL import Image
from requests_toolbelt.multipart import decoder
print("Import End...")

print("Downloading predictor file...Not really!")

try:
        landmark_detector = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
        print("Predictor file Loaded...")

except Exception as e:
    print(repr(e))
    raise(e)

def get_aligned_image(im):
    try:
        face_detector = dlib.get_frontal_face_detector()
        points = fbc.getLandmarks(face_detector, landmark_detector, im)
        points = np.array(points)
        im = np.float32(im)/255.0

        print("Inside the get_aligned_face function.")

        h = 600
        w = 600

        imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), im, points)
        imNorm = np.uint8(imNorm*255)
        return imNorm

        return 
    except Exception as e:
        print(repr(e))
        raise(e)


def align_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        im_arr = np.frombuffer(picture.content, dtype=np.uint8)
        im = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        print(filename)
        face_detector = dlib.get_frontal_face_detector()
        face_rects = face_detector(im, 0)
        if len(face_rects) !=1 :
            return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": "Incorrect number of faces in the uploaded image. Expected 1!"})
            }

        img = get_aligned_image(im)

        print("Image aligned, will be preparing it to send.")


        img = img[:, :, ::-1]
        img = Image.fromarray(img.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())

        print("Image processing complete, sending...")

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'aligned': str(img_base64)})
        }
    except Exception as e:
        print(repr(e))
        return{
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            "body": json.dumps({"error": repr(e)})
        }
