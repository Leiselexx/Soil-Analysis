import joblib
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class pH_Level:
    def __init__(self, model_path='pH_model-v6.3.pkl', scaler_path='scaler-2.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)  # Load pre-fitted StandardScaler

    def pHRecognition(self, image):
        dominant_color = self.dominantColorChecker(image)
        pH_Level = self.pHLevelModel(dominant_color)
        print(pH_Level, dominant_color)
        return pH_Level

    def pHLevelModel(self, rgb_color):
        #scaled_color = self.scaler.transform([rgb_color])  # Transform using pre-fitted scaler
        rgb_color_reshaped = np.array(rgb_color).reshape(1, -1)
        scaled_color = self.scaler.transform(rgb_color_reshaped)
        print(rgb_color_reshaped)
        print("SCALED:", scaled_color)
        pH_prediction = self.model.predict(scaled_color)
        return round(pH_prediction[0], 2)

    def dominantColorChecker(self, image):
        # Assuming 'image' is already an image array
        kernel_size = 9
        sigma = 1.5 - 3.0
        # Load the image

        if image is None:
            print("Image could not be loaded.")
        else:
            ######################################

            height, width, _ = image.shape

            # Define the size of the crop (centered)
            crop_width = 350  # Example crop width
            crop_height = 350  # Example crop height

            # Calculate the coordinates for the top-left and bottom-right corners of the centered crop
            x1 = (width - crop_width) // 2
            y1 = (height - crop_height) // 2
            x2 = x1 + crop_width
            y2 = y1 + crop_height

            # Crop the image to the specified centered region
            centered_cropped_image = image[y1:y2, x1:x2]
            centered_cropped_image = cv2.GaussianBlur(centered_cropped_image, (kernel_size, kernel_size), sigma)

            ######################################

            # Define a brightness factor (adjust this as needed)
            brightness_factor = 20.0  # You can increase or decrease this value

            # Convert the image to a float32 data type
            image_float = centered_cropped_image.astype(np.float32)

            # Create a brightness factor array with the same data type
            brightness_array = np.array([brightness_factor, brightness_factor, brightness_factor], dtype=np.float32)

            # Add the brightness factor to each channel of the image using NumPy
            brightened_image = image_float + brightness_array

            # Make sure pixel values are in the valid range [0, 255]
            brightened_image = np.clip(brightened_image, 0, 255)

            # Convert the brightened image back to uint8 data type
            brightened_image = brightened_image.astype(np.uint8)

            ######################################

            # Reshape the image to a 2D array of pixels and 3 color channels
            pixels = brightened_image.reshape(-1, 3)

            # Perform k-means clustering to find dominant color
            num_clusters = 1
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(pixels)

            # Get the centroid (dominant color)
            dominant_color = kmeans.cluster_centers_[0]

            dominant_color = np.uint8([[[dominant_color[0], dominant_color[1], dominant_color[2]]]])

            # Convert to BGR to RGB
            dominant_color = cv2.cvtColor(dominant_color, cv2.COLOR_BGR2RGB)

            # Convert to integer RGB values (0-255)
            dominant_color_rgb = np.uint8(dominant_color)

            red, green, blue = dominant_color_rgb[0][0]

            # red, green, blue = dominant_color_rgb
            return [red, green, blue]




