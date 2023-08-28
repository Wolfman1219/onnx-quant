import os
import urllib.request
import time
import tqdm
import numpy
import onnx
import pandas
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from onnxruntime.quantization.quantize import quantize_dynamic, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process


model_name = "yolov8n.onnx"
# url_name = ("https://github.com/onnx/models/raw/main/vision/"
                # "classification/mobilenet/model")


sess_full = InferenceSession(model_name, providers=["CPUExecutionProvider"])

for i in sess_full.get_inputs():
    print(f"input {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    input_name = i.name
    input_shape = list(i.shape)
    if input_shape[0] in [None, "batch_size", "N"]:
        input_shape[0] = 1

output_name = None
for i in sess_full.get_outputs():
    print(f"output {i}, name={i.name!r}, type={i.type}, shape={i.shape}")
    if output_name is None:
        output_name = i.name

print(f"input_name={input_name!r}, output_name={output_name!r}")

maxN = 50

imgs = [numpy.random.rand(*input_shape).astype(numpy.float32)

        for i in range(maxN)]

experiments = []

class DataReader(CalibrationDataReader):
    def __init__(self, input_name, imgs):
        self.input_name = input_name
        self.data = imgs
        self.pos = -1

    def get_next(self):
        if self.pos >= len(self.data) - 1:
            return None
        self.pos += 1
        return {self.input_name: self.data[self.pos]}

    def rewind(self):
        self.pos = -1


preprocessed_name = model_name + ".pre.onnx"

quantize_name = model_name + ".qdq.dyn.onnx"

quantize_dynamic(preprocessed_name, quantize_name,
                 weight_type=QuantType.QUInt8)
                 
