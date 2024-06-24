import numpy as np
import cv2
# import model wrapper class
from openvino.model_zoo.model_api.models import SSD
# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
# load openvino inference engine
from openvino.inference_engine import IECore

def load_openvino_model(load_model_path:str, batch_size=1):
    """_summary_

    Args:
        load_model_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    ie = IECore()
    net = ie.read_network(model=load_model_path)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = batch_size  # batch size
    input_shape = net.input_info[input_blob].input_data.shape
    print("input_blob: ", input_blob, "output_blob: ", out_blob)
    exec_net = ie.load_network(network=net, device_name='CPU')
    transpose = (0, 1, 2, 3)
    # wether necessary to transpose input or not in the function: openvino_inference
    if input_shape[1] == 3:
        transpose = (0, 3, 1, 2)
    return (exec_net, input_blob, transpose)

def openvino_inference(model:tuple, input):
    """_summary_

    Args:
        model (tuple): _description_
        input (np.ndarray): NHWC format

    Returns:
        _type_: _description_
    """
    if len(input.shape) == 3:
        input = np.expand_dims(input, axis=0)  # add batch dim, works well, no warning, no error because of [broadcast]
    exec_net, input_blob, transpose = model
    nchw_input = input.transpose(transpose)
    infer = exec_net.infer(inputs={input_blob: nchw_input})
    return [infer[key] for key in infer.keys()]  # [np.ndarray, ...]

if __name__ == '__main__':
    print(" load image and configure model path ")
    # read input image using opencv
    input_data = cv2.imread("./data/SF_sample.mp4.png")
    # define the path to mobilenet-ssd model in IR format
    model_path = "./model/public/mobilenet-ssd/FP32/mobilenet-ssd.xml"  # .onnx also works

    print(" using model from openvino zoo to execute inference ")
    # create adapter for OpenVINO™ runtime, pass the model path
    model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")
    # create model API wrapper for SSD architecture
    # preload=True loads the model on CPU inside the adapter
    ssd_model = SSD(model_adapter, preload=True)
    # apply input preprocessing, sync inference, model output postprocessing
    results = ssd_model(input_data)
    print(results)

    print(" using custom model to execute inference ")
    model = load_openvino_model(model_path, batch_size=1)
    results = openvino_inference(model, cv2.resize(input_data, (300, 300)))
    print(results)