from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from ph_color import pH_Level
from soilRecognition import SoilRecogClass
from cropRecommendation import RecommendCrops
import tempfile

app = Flask(__name__)

# Initialize models and classes
ph_level_predictor = pH_Level()
recognize_Soil = SoilRecogClass()
cropRecommend = RecommendCrops()

# Load the soil classification model and crop data
soil_model = tf.keras.models.load_model('soil_classification_model_v3.h5')
df = pd.read_excel('cropsData.xlsx')


@app.route('/readiness_check')
def readiness_check():
    # Perform checks to determine if the app is ready
    return 'OK', 200

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    if request.method == 'POST':
        # Get the uploaded image file and save it temporarily
        file = request.files['image']
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            image_path = temp.name

        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return "Error: The image could not be loaded."

        check_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        check_img = tf.keras.preprocessing.image.img_to_array(check_img)
        check_img = np.expand_dims(check_img, axis=0)  # Add batch dimension
        check_img = check_img / 255.0  # Normalize pixel values to [0, 1]
        # Check for soil recognition
        is_soil_recognized = recognize_Soil.soilDetection(image_path)

        if is_soil_recognized:
            # Soil is recognized, proceed with pH and soil classification
            pH_level = ph_level_predictor.pHRecognition(img)  # Pass the resized image for pH recognition

            # Make a prediction using the soil classification model
            soil_classification_prediction = soil_model.predict(check_img)
            predicted_class = np.argmax(soil_classification_prediction)
            crop_recommendation = cropRecommend.recommend_crops_for_pH(pH_level, df)

            # Define soil classes
            classes = ['Clay', 'Loam', 'Sandy']

            # Return the combined results
            return render_template('index.html',
                                   soil_result=f'Predicted Soil Class: {classes[predicted_class]}',
                                   pH_result=f'Predicted pH Level: {pH_level:.2f}',
                                   crop_result=f'Recommended Crops: {crop_recommendation}')
        else:
            # Soil is not recognized
            return render_template('index.html', soil_result='Soil not recognized. Unable to proceed with pH and soil classification.')

if __name__ == '__main__':
    app.run(debug=True)
