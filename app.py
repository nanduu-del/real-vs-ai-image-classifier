import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Settings
IMAGE_SIZE = (224, 224)
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "real_vs_ai_model.keras"

# Dataset categories
CLASS_NAMES = ["ai", "real"]


# Build the AI model
def build_model():

    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Load or train the model
def load_model():

    if MODEL_PATH.exists():
        print("Loading saved model...")
        return tf.keras.models.load_model(MODEL_PATH)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation"

    for folder in [
        train_dir / "ai",
        train_dir / "real",
        val_dir / "ai",
        val_dir / "real"
    ]:
        if not folder.is_dir() or not any(
            p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            for p in folder.iterdir() if p.is_file()
        ):
            raise FileNotFoundError(
                f"Missing images in {folder}"
            )

    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=16,
        label_mode="binary"
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=16,
        label_mode="binary",
        shuffle=False
    )

    model = build_model()

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=8
    )

    model.save(MODEL_PATH)

    return model


# Classify an uploaded image
def classify_image(image):

    if image is None:
        return "Please upload an image."

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    real_score = prediction * 100
    ai_score = (1 - prediction) * 100

    if prediction >= 0.5:
        result = "REAL IMAGE"
        confidence = real_score
    else:
        result = "AI-GENERATED IMAGE"
        confidence = ai_score

    return f"""
# Classification Result

## {result}

**Confidence: {confidence:.2f}%**

### Prediction scores

- Real image: {real_score:.2f}%
- AI-generated image: {ai_score:.2f}%

*This is an AI prediction and may not always be correct.*
"""


# Start the model
print("Loading AI model...")
model = load_model()
print("Model is ready!")


# Create application
with gr.Blocks(
    title="Real vs AI Image Classifier",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("""
    # REAL vs AI IMAGE CLASSIFIER

    ### Class 12 Artificial Intelligence Project

    Upload an image to identify whether it is
    a real photograph or AI-generated.
    """)

    with gr.Row():

        image_input = gr.Image(
            label="Upload Image",
            type="pil"
        )

        output = gr.Markdown(
            "Your result will appear here."
        )

    with gr.Row():

        classify = gr.Button(
            "CLASSIFY IMAGE",
            variant="primary"
        )

        clear = gr.ClearButton(
            [image_input, output],
            value="CLEAR"
        )

    classify.click(
        classify_image,
        inputs=image_input,
        outputs=output
    )

    gr.Markdown("""
    ### Technologies Used

    Python | TensorFlow | EfficientNetB0 |
    Gradio | NumPy | Pillow
    """)


if __name__ == "__main__":
    app.launch(
        pwa=True,
        inbrowser=True
)
