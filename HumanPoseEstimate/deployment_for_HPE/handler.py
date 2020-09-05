try:
    import unzip_requirements
except ImportError:
    pass

import boto3
import os
import io
import base64
import json
import albumentations
import numpy as np
import cv2
from PIL import Image
from operator import itemgetter
from requests_toolbelt.multipart import decoder


# Import package
from simple_pose_estimate import human_pose_estimate

POSE_PAIRS = [[9, 8],[8, 7],[7, 6],[6, 2],[2, 1],[1, 0],[6, 3],[3, 4],[4, 5],[7, 12],[12, 11],[11, 10],[7, 13],[13, 14],[14, 15]]
JOINTS = ['r-ankle', 'r-knee', 'r-hip', 'l-hip', 'l-knee', 'l-ankle', 'pelvis', 'thorax', 'upper-neck', 'head-top', 'r-wrist', 'r-elbow', 'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist']
MODEL_PATH = 'simple_pose_estimation.quantized.onnx'

def load_image_from_array(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)
    print("handler.py - Image loaded from image array")
    return img

def transform_image(image):
    transform = albumentations.Compose([
        albumentations.Resize(256, 256, always_apply=True),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229,0.224,0.225), always_apply=True),
    ])
    return transform(image=np.array(image, dtype=np.float32))['image']

def transpose_unsqueeze(image):
    # Convert from H,W,C to C,H,W format and add dimension for batch_size to get - N,C,H,W
    return np.expand_dims(np.transpose(image, (2,0,1)), axis=0)

get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])


def draw_pose(image_p, key_points, OUT_SHAPE):
    THRESHOLD = 0.6    
    
    is_joint_plotted = [False for i in range(len(JOINTS))]
    for pose_pair in POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

        from_x_j, to_x_j = from_x_j * IMG_WIDTH / OUT_SHAPE[0], to_x_j * IMG_WIDTH / OUT_SHAPE[0]
        from_y_j, to_y_j = from_y_j * IMG_HEIGHT / OUT_SHAPE[1], to_y_j * IMG_HEIGHT / OUT_SHAPE[1]

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > THRESHOLD and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > THRESHOLD and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > THRESHOLD and to_thr > THRESHOLD:
            # this is a joint connection, plot a line
            cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)

    return Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_RGB2BGR))



def human_pose_estimate(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print("handler.py - Image body in human_pose_estimate loaded")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        print(filename)

        # Fetch image from given image_bytes
        image = load_image_from_array(picture.content)

        #Transform image
        trans_image = transform_image(image)
        trans_image = transpose_unsqueeze(trans_image)

        #Load model
        model = human_pose_estimate(MODEL_PATH)

        #Fetch pose layers from the model
        pose_layers = model(trans_image)
        OUT_SHAPE = pose_layers.shape[1:]

        #Extract keypoints from pose layers
        key_points = list(get_keypoints(pose_layers=pose_layers))

        #Draw the extracted pose
        pose_estimate = draw_pose(np.array(image), key_points, OUT_SHAPE))

        #Prepare output to return
        print("Loading output to buffer")
        buffer = io.BytesIO()
        pose_estimate.save(buffer, format="JPEG")
        pose_estimate_bytes = base64.b64encode(buffer.getvalue())

        return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                "body": json.dumps({'file': filename.replace('"', ''), 'data': pose_estimate_bytes.decode('ascii')})
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