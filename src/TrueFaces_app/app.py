import os
import numpy as np
from flask import Flask, render_template, request, url_for, redirect, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Lambda, Add
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
from preprocessing.patch_generator import smash_n_reconstruct
import preprocessing.filters as f
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Define the directory where uploaded files will be saved
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Custom Activation Function
@tf.function
def hard_tanh(x):
    return tf.maximum(tf.minimum(x, 1), -1)

# Define the feature extraction layer with residual block
class featureExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.bn2 = BatchNormalization()

        # Residual Block
        self.res_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.res_bn = BatchNormalization()

        # Activation
        self.activation = Lambda(hard_tanh)

    def call(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Residual Block
        res = self.res_conv(x)
        res = self.res_bn(res)
        x = Add()([x, res])  # Adding the input to the output of the residual block

        x = self.activation(x)
        return x

# Load the saved model
model = load_model('useful_files/best_model.h5', 
                   custom_objects={
                       'featureExtractionLayer': featureExtractionLayer, 
                       'hard_tanh': hard_tanh, 
                       'Adam': Adam  # Register Adam as a custom object
                   })

# Function to preprocess a single image
def preprocess_single_image(image_path):
    rt, pt = smash_n_reconstruct(image_path)
    frt = tf.cast(tf.expand_dims(f.apply_all_filters(rt), axis=-1), dtype=tf.float64)
    fpt = tf.cast(tf.expand_dims(f.apply_all_filters(pt), axis=-1), dtype=tf.float64)
    return frt, fpt

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    
    if imagefile and allowed_file(imagefile.filename):
        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(image_path)

        try:
            # Verify that the file is an image
            img = Image.open(image_path)
            img.verify()  # PIL will throw an error if it's not a valid image

            # Preprocess the image
            frt, fpt = preprocess_single_image(image_path)
            frt = tf.expand_dims(frt, axis=0)  # Add batch dimension
            fpt = tf.expand_dims(fpt, axis=0)  # Add batch dimension

            # Make the prediction
            prediction = model.predict({'rich_texture': frt, 'poor_texture': fpt})
            confidence = prediction[0].squeeze()
            predicted_label = 'Fake' if confidence > 0.5 else 'Real'
            confidence_percent = (1 - confidence if predicted_label == 'Real' else confidence) * 100

            # Redirect to the same page with the anchor
            return redirect(url_for('index', 
                                    filename=filename, 
                                    prediction=predicted_label, 
                                    confidence=confidence_percent) + '#upload')

        except (IOError, Image.UnidentifiedImageError):
            flash("Invalid image file. Please upload a valid image in PNG or JPG format.")
            return redirect(url_for('index'))
    else:
        flash("Allowed image types are - png, jpg, jpeg")
        return redirect(url_for('index'))

@app.route('/', methods=['GET'])
def index():
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    return render_template('index.html', 
                           filename=filename, 
                           prediction=prediction, 
                           confidence=confidence)

if __name__ == "__main__":
    app.run(port=8080, debug=True)
