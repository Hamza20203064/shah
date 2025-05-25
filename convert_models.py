import tensorflow as tf
import os
import numpy as np


def create_simple_model():
    # Create a very simple model for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(27, activation='softmax')  # 26 letters + blank
    ])
    return model


def convert_to_tflite(model_name):
    try:
        # Create and compile model
        model = create_simple_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Create a dummy input
        dummy_input = np.zeros((1, 128, 128, 1))
        model.predict(dummy_input)

        # Save as SavedModel first
        saved_model_path = os.path.join('model', f'{model_name}_saved')
        model.save(saved_model_path)

        # Convert using SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert
        tflite_model = converter.convert()

        # Save TFLite model
        tflite_path = os.path.join('model', f'{model_name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully created {tflite_path}")

        # Clean up saved model directory
        import shutil
        shutil.rmtree(saved_model_path)

    except Exception as e:
        print(f"Error converting {model_name}: {str(e)}")
        if os.path.exists(saved_model_path):
            shutil.rmtree(saved_model_path)


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
