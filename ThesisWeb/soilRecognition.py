from PIL import Image
import tempfile
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras


image_size = (128, 128)
isSoil = False

soilRecog_model = keras.models.load_model('soil_detection_model.h5')

class SoilRecogClass:
    def soilDetection(self, image):
        # Check if the input is a file path
        if isinstance(image, str):
            # Load the image directly using the file path
            detection_image = tf.keras.preprocessing.image.load_img(image, target_size=image_size)
            detection_image = tf.keras.preprocessing.image.img_to_array(detection_image)
            detection_image = np.expand_dims(detection_image, axis=0)  # Add batch dimension
            detection_image = detection_image / 255.0  # Normalize pixel values to [0, 1]
        else:  # Assume the input is a NumPy array
            # Ensure the image is resized to the desired size
            detection_image = tf.image.resize(image, image_size)
            detection_image = detection_image.numpy() / 255.0
            detection_image = np.expand_dims(detection_image, axis=0)

        # Perform soil detection
        soilRecognition = soilRecog_model.predict(detection_image)
        print(soilRecognition[0][0])

        # Assuming 0.5 as the threshold, you can adjust as needed
        threshold = 0.5
        isSoil = soilRecognition[0][0] >= threshold

        return isSoil

