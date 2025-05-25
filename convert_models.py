import tensorflow as tf
import os
import numpy as np


def create_tflite_model(model_name):
    try:
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(27, activation='softmax')
        ])

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Create a dummy input
        dummy_input = np.zeros((1, 128, 128, 1))
        model.predict(dummy_input)

        # Create TFLite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set input and output types
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        # Convert
        tflite_model = converter.convert()

        # Save TFLite model
        tflite_path = os.path.join('model', f'{model_name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully created {tflite_path}")

        # Verify the model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        print(f"Model verification successful for {model_name}")

    except Exception as e:
        print(f"Error creating {model_name}: {str(e)}")


# List of models to create
models = [
    'model-bw',
    'model-bw_dru',
    'model-bw_tkdi',
    'model-bw_smn'
]

# Create each model
for model_name in models:
    print(f"\nCreating {model_name}...")
    create_tflite_model(model_name)
