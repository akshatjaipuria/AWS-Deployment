import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize
from onnxruntime.quantization import QuantizationMode

class human_pose_estimate:
    """
    Class to calculate 16-keypoints of human pose.
    Input:
        model_path - path to quantized ONXX Simple Human Pose estimation model
        image - Numpy array in NxCxHxW format. 

    Output:
        feature_map with 16 key points.

    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = onnxruntime.InferenceSession(self.model_path)
    
    def __call__(self, image):
        ort_inputs = {self.ort_session.get_inputs()[0].name: image}
        ort_outs = self.ort_session.run(None, ort_inputs)[0][0]
        return ort_outs
