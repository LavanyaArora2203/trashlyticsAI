import os
from models.garbage_classifier import GarbageClassifier

os.makedirs("saved_models", exist_ok=True)

classifier = GarbageClassifier(use_pretrained=False)

classifier.save("saved_models/garbage_classifier.pkl")

print("✅ Model saved successfully!")
