import tensorflow as tf
import os
import json
import keras
from keras.models import model_from_json
from keras import backend as K


def convert_model(json_path, weights_path, output_path):
    # Set the backend to tensorflow
    K.set_image_data_format('channels_last')

    # Load the model architecture from JSON
    with open(json_path, 'r') as f:
        model_json = f.read()

    # Create model from JSON
    model = model_from_json(model_json)

    # Load weights
    model.load_weights(weights_path)

    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted {json_path} and {weights_path} to {output_path}")


# Convert all models
model_dir = 'model'
models = [
    ('model-bw.json', 'model-bw.h5', 'model-bw.tflite'),
    ('model-bw_dru.json', 'model-bw_dru.h5', 'model-bw_dru.tflite'),
    ('model-bw_tkdi.json', 'model-bw_tkdi.h5', 'model-bw_tkdi.tflite'),
    ('model-bw_smn.json', 'model-bw_smn.h5', 'model-bw_smn.tflite')
]

for json_file, h5_file, tflite_file in models:
    json_path = os.path.join(model_dir, json_file)
    weights_path = os.path.join(model_dir, h5_file)
    output_path = os.path.join(model_dir, tflite_file)
    convert_model(json_path, weights_path, output_path)
