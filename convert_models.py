import tensorflow as tf
import os


def convert_to_tflite(h5_model_path, tflite_model_path):
    # Load the H5 model
    model = tf.keras.models.load_model(h5_model_path)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted {h5_model_path} to {tflite_model_path}")


# List of models to convert
models = [
    ('model-bw.h5', 'model-bw.tflite'),
    ('model-bw_dru.h5', 'model-bw_dru.tflite'),
    ('model-bw_tkdi.h5', 'model-bw_tkdi.tflite'),
    ('model-bw_smn.h5', 'model-bw_smn.tflite')
]

# Convert each model
for h5_file, tflite_file in models:
    h5_path = os.path.join('model', h5_file)
    tflite_path = os.path.join('model', tflite_file)
    if os.path.exists(h5_path):
        convert_to_tflite(h5_path, tflite_path)
    else:
        print(f"Warning: {h5_path} not found")
