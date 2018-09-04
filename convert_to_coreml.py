import os
import sys
sys.path.append("model/")
from ops import *

import argparse
import coremltools

def get_modelpaths(model_dir):
    '''
    model directory에서 
    모델 파라미터 정보가 담겨진 h5파일과 
    모델 구조 정보가 담겨진 json파일을 
    가져오는 함수
    '''
    h5_path = None
    json_path = None
    for filename in os.listdir(model_dir):
        file_extension = os.path.splitext(filename)[1]
        if file_extension == '.h5':
            h5_path = os.path.join(model_dir, filename)
        elif file_extension == '.json':
            json_path = os.path.join(model_dir, filename)

    if h5_path is None: 
        raise Exception("There is no json file")
    if json_path is None:
        raise Exception("There is no json file")
    
    return json_path, h5_path

def load_model(json_path, h5_path):
    '''
    모델 구조와 모델 파라미터 정보를 바탕으로
    모델을 로드하는 함수
    '''
    from keras.models import model_from_json
    with open(json_path) as f:
        model = model_from_json(f.read())
    model.load_weights(h5_path)
    return model

def check_existence_instance_normalization_in_layer(model):
    for layer in model.layers:
        if "instance_normalization" in layer.name:
            return True
    return False

def convert_lambda(layer):
    from coremltools.proto import NeuralNetwork_pb2
    params = NeuralNetwork_pb2.CustomLayerParams()
    params.className = "InstanceNormalization"
    return params

def main(model_dir):
	model = load_model(*get_modelpaths(model_dir))

	if check_existence_instance_normalization_in_layer(model):
	    coreml_model = coremltools.converters.keras.convert(
	    model,
	    input_names='image',
	    image_input_names='image',
	    output_names='output',
	    image_scale=1/255.0,
	    add_custom_layers=True,
	    custom_conversion_functions=
	    {"InstanceNormalization": convert_lambda})
	else:
	    coreml_model = coremltools.converters.keras.convert(
	    model,
	    input_names='image',
	    image_input_names='image',
	    output_names='output',
	    image_scale=1/255.0)

	save_path = os.path.join(model_dir,"coreml.mlmodel")    

	coreml_model.author = "Sang Jae Kang"
	coreml_model.license = "Private Domain"
	coreml_model.short_description = "Human Body Segmentation"

	coreml_model.input_description['image'] = "RGB 256x256 input image"
	coreml_model.output_description['output'] = "Body Profile Mask"

	coreml_model.save(save_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("model_path", help='the model directory to convert to Coreml.mlmodel')
	args = parser.parse_args()
	main(args.model_path)


