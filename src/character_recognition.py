import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

class CharacterRecognizer:
    def __init__(self, model_path, labels_path='./models/labels_dict.json', img_size=(28, 28)):
        # Inference-only load: skip optimizer/training state to reduce memory usage.
        self.model = load_model(model_path, compile=False)
        self.img_size = img_size
        with open(labels_path, 'r') as f:
            self.class_labels = json.load(f)

    def preprocess_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def recognize_characters(self, segmented_images, return_probs=False):
        results = []
        for img in segmented_images:
            if img is None or img.size == 0:
                continue
            preprocessed_img = self.preprocess_image(img)
            prediction = self.model.predict(preprocessed_img, verbose=0)
            prob = np.max(prediction, axis=1)[0]
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_character = self.class_labels[str(predicted_class)]
            if return_probs:
                results.append((predicted_character, prob))
            else:
                results.append(predicted_character)
        if not return_probs:
            return "".join(results)
        return results
