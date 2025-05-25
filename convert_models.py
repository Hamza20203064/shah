import tensorflow as tf
import os
import json
import numpy as np


def create_simple_model():
    # Create a simple CNN model similar to the original architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(27, activation='softmax')  # 26 letters + blank
    ])
    return model


def convert_to_tflite(model_name):
    try:
        # Create model
        model = create_simple_model()

        # Compile model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Create a dummy input for the model
        dummy_input = np.random.random((1, 128, 128, 1))
        model.predict(dummy_input)  # Initialize the model

        # Convert to TFLite with specific optimizations
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        # Convert the model
        tflite_model = converter.convert()

        # Save the TFLite model
        tflite_path = os.path.join('model', f'{model_name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully created {tflite_path}")

    except Exception as e:
        print(f"Error converting {model_name}: {str(e)}")


# List of models to create
models = [
    'model-bw',
    'model-bw_dru',
    'model-bw_tkdi',
    'model-bw_smn'
]

# Convert each model
for model_name in models:
    print(f"\nConverting {model_name}...")
    convert_to_tflite(model_name)
