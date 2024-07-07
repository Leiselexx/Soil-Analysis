from flask import Flask, render_template, request, url_for,  send_from_directory
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from ph_color import pH_Level
from soilRecognition import SoilRecogClass
from cropRecommendation import RecommendCrops
import tempfile
from werkzeug.utils import secure_filename
from pathlib import Path

import os
import shutil

#testtttttt
app = Flask(__name__)

# Initialize models and classes
ph_level_predictor = pH_Level()
recognize_Soil = SoilRecogClass()
cropRecommend = RecommendCrops()

# Load the soil classification model and crop data
soil_model = tf.keras.models.load_model('soil_classification_model_v3.h5')
df = pd.read_excel('cropsData.xlsx')
app.config['ENV'] = 'production' 


@app.route('/readiness_check')
def readiness_check():
    # Perform checks to determine if the app is ready
    return 'OK', 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/crop')
def contacts():
    return render_template('crop.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('/tmp', filename)



@app.route('/predict_soil', methods=['POST'])
def predict_soil():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # Save the image in the static/uploads directory
            filename = secure_filename('test')
            if app.config.get('ENV', 'production')== 'production':
                filepath = os.path.join('/tmp', filename)

            else:
                filepathlocal = os.path.join(app.root_path, 'static', 'uploads', filename)

            file.save(filepath)
            

            # Read the image using OpenCV
            img = cv2.imread(filepath)
            if img is None:
                return "Error: The image could not be loaded."

            check_img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            check_img = tf.keras.preprocessing.image.img_to_array(check_img)
            check_img = np.expand_dims(check_img, axis=0)  # Add batch dimension
            check_img = check_img / 255.0  # Normalize pixel values to [0, 1]
            # Check for soil recognition
            is_soil_recognized = recognize_Soil.soilDetection(filepath)

            if is_soil_recognized:
                # Soil is recognized, proceed with pH and soil classification
                pH_level = ph_level_predictor.pHRecognition(img)  # Pass the resized image for pH recognition
                colors = ph_level_predictor.dominantColorChecker(img)
                values_list = colors[0].tolist()
                red, green, blue = colors

                # Make a prediction using the soil classification model
                soil_classification_prediction = soil_model.predict(check_img)
                predicted_class = np.argmax(soil_classification_prediction)
                crop_recommendation = cropRecommend.recommend_crops_for_pH(pH_level, df)

                # Define soil classes
                classes = ['CLAY', 'LOAM', 'SANDY']

                soil_definitions = {
                    'CLAY': 'Clay soil is characterized by fine particles. It retains water well but can become compacted.',
                    'LOAM': 'Loam soil is a balanced mixture of sand, silt, and clay. It is ideal for plant growth.',
                    'SANDY': 'Sandy soil has larger particles and drains quickly. It may require more frequent watering.'
                }

                #This is to only show the predicted soil type definition.
                soil_definition = soil_definitions.get(f'{classes[predicted_class]}', 'No definition available.')

                # Create a URL for the image
                if app.config.get('ENV', 'production')== 'production':
                    image_url = url_for('uploaded_file', filename=filename)
                else:
                    image_url = url_for('static', filename=os.path.join('uploads', filename))
                

                # Return the combined results
                return render_template('results.html',
                                       soil_result=f'{classes[predicted_class]}',
                                       soil_definition=soil_definition,
                                       pH_result=f'{pH_level:.2f}',
                                       redMeter=(red/255)*100,
                                       greenMeter=(green / 255) * 100,
                                       blueMeter=(blue / 255) * 100,
                                       crop_result=crop_recommendation,
                                       soilImg=image_url)
            else:
                # Soil is not recognized
                return render_template('notRecognized.html', soil_result='Soil not recognized. Unable to proceed with pH and soil classification.')

        else:
            # No file was uploaded
            return render_template('index.html', error="No file uploaded.")

@app.route('/get_crop_recommendations', methods=['POST'])
def get_crop_recommendations():
    if request.method == 'POST':
        pH_level = float(request.form['pH_level'])  # Extract pH level from the form
        crop_recommendations = cropRecommend.recommend_crops_for_pH(pH_level, df)
        return render_template('crop.html', crop_recommendations=crop_recommendations)
            
if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

    

