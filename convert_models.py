import tensorflow as tf
import os
import json
import numpy as np


def create_model_from_config(json_path):
    with open(json_path, 'r') as f:
        model_config = json.load(f)

    # Create model from config
    model = tf.keras.models.model_from_config(model_config)
    return model


def convert_model(json_path, weights_path, output_path):
    try:
        print(f"Loading model from {json_path}")
        model = create_model_from_config(json_path)
        print("Model architecture loaded")

        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path)
        print("Weights loaded")

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        print("Model compiled")

        # Convert the model to TFLite format
        print("Converting to TFLite format")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Set basic conversion options
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float32]

        # Convert
        tflite_model = converter.convert()

        # Save the TFLite model
        print(f"Saving TFLite model to {output_path}")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully converted {json_path} to {output_path}")

        # Verify the model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        print("Model verification successful")

    except Exception as e:
        print(f"Error converting {json_path}: {str(e)}")
        print("Full error:", e)


# Convert all models
model_dir = 'model'
models = [
    ('model-bw.json', 'model-bw.h5', 'model-bw.tflite'),
    ('model-bw_dru.json', 'model-bw_dru.h5', 'model-bw_dru.tflite'),
    ('model-bw_tkdi.json', 'model-bw_tkdi.h5', 'model-bw_tkdi.tflite'),
    ('model-bw_smn.json', 'model-bw_smn.h5', 'model-bw_smn.tflite')
]

for json_file, h5_file, tflite_file in models:
    print(f"\nProcessing {json_file}...")
    json_path = os.path.join(model_dir, json_file)
    weights_path = os.path.join(model_dir, h5_file)
    output_path = os.path.join(model_dir, tflite_file)
    convert_model(json_path, weights_path, output_path)
