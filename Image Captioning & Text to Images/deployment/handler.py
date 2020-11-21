try:
    import unzip_requirements
except ImportError:
    pass

from PIL import Image
import numpy as np

import boto3
import os
import io
import base64
import json
from requests_toolbelt.multipart import decoder

from image_caption import load_model, load_word_map, caption_image
print("Import end...")

S3_BUCKET = 'motley-imagecaption-flickr30k'
MODEL_PATH = 'imageCaption_flickr30k_checkpoint.pth.tar'
word_map_file = 'WORDMAP.json'

s3 = boto3.client('s3')

# Load checkpoint file from S3. Load encoder, decoder model from the checkpoint using image_caption.load_model
try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket = S3_BUCKET, Key = MODEL_PATH)
        print("Downloading model...")
        checkpoint = io.BytesIO(obj['Body'].read())
        print("Checkpoint loaded")
        encoder_model, decoder_model = load_model(checkpoint)
        print("Encoder-Decoder model loaded")

except Exception as e:
    print(repr(e))
    raise(e)

# Load word_map and rev_word_map dictionaries from word_map_file json.
word_map, rev_word_map = load_word_map(word_map_file)
print("Word map dict loaded")

# Load input image from event.
def load_input_image(event):
    # Returns PIL image and uploaded filename.
    content_type_header = event['headers']['content-type']
    body = base64.b64decode(event["body"])
    print("Image body in caption_image loaded")
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
    if len(filename) < 4:
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
    print("picture object from body- ",picture)
    print("picture object content from body- ", picture.content)
    input_image = Image.open(io.BytesIO(picture.content))
    return input_image, filename


def caption_this(event, context):
    try:
        image, filename = load_input_image(event)

        caption = caption_image(encoder_model, decoder_model, image, word_map, rev_word_map)

        return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                "body": json.dumps({'file': filename.replace('"', ''), 'caption': caption})
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
