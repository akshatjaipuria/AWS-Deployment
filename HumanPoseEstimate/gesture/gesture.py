import cv2
import onnxruntime
import numpy as np
import os
import copy
import sys
from operator import itemgetter
from PIL import Image
import glob
from pygame import mixer
mixer.init()
mixer.music.load('nothing_else_matters.mp3')
play = False
stop = False



def pose_plot(ort_outs, image_p):
    pose_layers = ort_outs
    key_points = list(get_keypoints(pose_layers=pose_layers))
    is_joint_plotted = [False for i in range(len(JOINTS))]
    IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape
    thr, (thorax_x, thorax_y) = key_points[7]
    thorax_x, thorax_y = thorax_x * IMG_WIDTH / OUT_SHAPE[0], thorax_y * IMG_HEIGHT / OUT_SHAPE[1]
    thorax = (thorax_x, thorax_y)

    thr, (l_wrist_x, l_wrist_y) = key_points[15]
    l_wrist_x, l_wrist_y = l_wrist_x * IMG_WIDTH / OUT_SHAPE[0], l_wrist_y * IMG_HEIGHT / OUT_SHAPE[1]
    l_wrist = (l_wrist_x, l_wrist_y)

    thr, (r_wrist_x, r_wrist_y) = key_points[10]
    r_wrist_x, r_wrist_y = r_wrist_x * IMG_WIDTH / OUT_SHAPE[0], r_wrist_y * IMG_HEIGHT / OUT_SHAPE[1]
    r_wrist = (r_wrist_x, r_wrist_y)

    return thorax, l_wrist, r_wrist


ort_session = onnxruntime.InferenceSession("simple_pose_estimation.quantized.onnx")


OUT_SHAPE = (64, 64)
THRESHOLD = 0.5

get_detached = lambda x: copy.deepcopy(x.cpu().detach().numpy())

POSE_PAIRS = [[9, 8],[8, 7],[7, 6],[6, 2],[2, 1],[1, 0],[6, 3],[3, 4],[4, 5],[7, 12],[12, 11],[11, 10],[7, 13],[13, 14],[14, 15]]
get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])
JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']


cap = cv2.VideoCapture(0)

while True: 
    ret, img = cap.read()
    img_1 = cv2.resize(img, (256, 256))
    # print(img_1.shape)
    img_1 = (img_1 - np.min(img_1))/np.ptp(img_1)

    img_1 = np.expand_dims(img_1, axis=0)
    img_1 = np.transpose( img_1, (0, 3, 1, 2))
    img_1 = np.float32(img_1)
    # print(img_1.shape)
    # print(img_1.dtype)


    ort_inputs = {ort_session.get_inputs()[0].name: img_1}
    ort_outs = ort_session.run(None, ort_inputs)

    ort_outs = np.array(ort_outs[0][0])

    thorax, l_wrist, r_wrist = pose_plot(ort_outs, img)

    if l_wrist[1] < thorax[1] and not play:
        mixer.music.play()
        play = True
        stop = False

    if r_wrist[1] < thorax[1] and not stop:
        mixer.music.stop()
        play = False
        stop = True


    cv2.imshow('Video', img)

    if(cv2.waitKey(10) & 0xFF == ord('x')):
        break
