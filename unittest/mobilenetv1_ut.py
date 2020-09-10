import numpy as np
import torch as tr
from framework.pytorch.backbone.mobilenetv1 import MobileNetV1

if __name__ == "__main__":
    input_data = np.random.randint(0, 255, [1, 3, 224, 224])  # pytorch数据格式为NCHW
    input_tensor = tr.Tensor(input_data)
    model = MobileNetV1()
    cmax = model(input_tensor)
    print(cmax)