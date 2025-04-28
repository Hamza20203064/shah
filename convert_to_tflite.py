import tensorflow as tf
import os


def convert_model(model_path, output_path):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted {model_path} to {output_path}")


# Convert all models
model_dir = 'model'
models = [
    ('model-bw.h5', 'model-bw.tflite'),
    ('model-bw_dru.h5', 'model-bw_dru.tflite'),
    ('model-bw_tkdi.h5', 'model-bw_tkdi.tflite'),
    ('model-bw_smn.h5', 'model-bw_smn.tflite')
]

for h5_file, tflite_file in models:
    input_path = os.path.join(model_dir, h5_file)
    output_path = os.path.join(model_dir, tflite_file)
    convert_model(input_path, output_path)
