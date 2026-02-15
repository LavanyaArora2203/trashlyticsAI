import os
import numpy as np
from PIL import Image
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optional TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False


class GarbageClassifier:

    def __init__(self, use_pretrained=True):
        self.img_size = 224
        self.use_pretrained = use_pretrained and TF_AVAILABLE

        if self.use_pretrained:
            self.model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights="imagenet",
                pooling="avg"
            )
        else:
            self.model = None

    def _analyze_image_features(self, img_array):
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)

        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])

        color_variance = np.std([r_mean, g_mean, b_mean])

        score = 0.5

        if mean_brightness < 120:
            score += 0.15
        elif mean_brightness > 180:
            score -= 0.15

        if color_variance > 20:
            score += 0.2

        if std_brightness > 60:
            score += 0.15

        score = np.clip(score, 0.0, 1.0)
        return float(score)

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)

        probability = self._analyze_image_features(img_array)

        severity = self._calculate_severity(probability)

        return {
            "garbage_probability": probability,
            "severity_score": severity,
            "classification": "Garbage" if probability > 0.5 else "Not Garbage",
        }

    def _calculate_severity(self, probability):
        if probability > 0.8:
            severity = 0.8 + (probability - 0.8)
        elif probability > 0.5:
            severity = 0.4 + (probability - 0.5) * 1.33
        else:
            severity = probability * 0.8

        return float(np.clip(severity, 0.0, 1.0))

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
