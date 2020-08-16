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


try:
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("Predictor file Loaded...")

except Exception as e:
    print(repr(e))
    raise(e)

def get_swapped_image(img1, img2):
    try:
        im1Display = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        im2Display = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1Warped = np.copy(img2)

        detector = dlib.get_frontal_face_detector()

        # Read array of corresponding points
        points1 = fbc.getLandmarks(detector, predictor, img1)
        points2 = fbc.getLandmarks(detector, predictor, img2)

        # Find convex hull
        hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

        # Create convex hull lists
        hull1 = []
        hull2 = []
        for i in range(0, len(hullIndex)):
            hull1.append(points1[hullIndex[i][0]])
            hull2.append(points2[hullIndex[i][0]])

        # Calculate Mask for Seamless cloning
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(img2.shape, dtype=img2.dtype) 
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        # Find Centroid
        m = cv2.moments(mask[:,:,1])
        center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

        # Find Delaunay traingulation for convex hull points
        sizeImg2 = img2.shape    
        rect = (0, 0, sizeImg2[1], sizeImg2[0])

        dt = fbc.calculateDelaunayTriangles(rect, hull2)

        # If no Delaunay Triangles were found, quit
        if len(dt) == 0:
            quit()

        imTemp1 = im1Display.copy()
        imTemp2 = im2Display.copy()

        tris1 = []
        tris2 = []
        for i in range(0, len(dt)):
            tri1 = []
            tri2 = []
            for j in range(0, 3):
                tri1.append(hull1[dt[i][j]])
                tri2.append(hull2[dt[i][j]])

            tris1.append(tri1)
            tris2.append(tri2)

        cv2.polylines(imTemp1,np.array(tris1),True,(0,0,255),2);
        cv2.polylines(imTemp2,np.array(tris2),True,(0,0,255),2);

        # Simple Alpha Blending
        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(tris1)):
            fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])

        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

        return output

        return 
    except Exception as e:
        print(repr(e))
        raise(e)


def face_swap(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        im_arr = np.frombuffer(picture.content, dtype=np.uint8)
        im1 = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        print("First image loaded!")

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        print(filename)


        picture = decoder.MultipartDecoder(body, content_type_header).parts[1]
        im_arr = np.frombuffer(picture.content, dtype=np.uint8)
        im2 = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        print("Second image loaded!")


        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        print(filename)


        face_detector = dlib.get_frontal_face_detector()

        #both images should have 1 faces each, checking...
        face_rects = face_detector(im1, 0)
        if len(face_rects) !=1 :
            return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": "Incorrect number of faces in the 1st uploaded image. Expected 1!"})
            }


        face_rects = face_detector(im2, 0)
        if len(face_rects) !=1 :
            return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": "Incorrect number of faces in the 2nd uploaded image. Expected 1!"})
            }


        img = get_swapped_image(im2, im1)

        print("Face Swapped, will be preparing it to send.")

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
            "body": json.dumps({'file': "swapped_img.jpg", 'swapped': str(img_base64)})
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
