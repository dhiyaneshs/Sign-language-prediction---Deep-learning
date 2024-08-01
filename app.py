from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model('hand_gesture_model.h5')

# Define the class labels (Modify this according to your model's labels)
class_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((64, 64))  # Resize to match the model's expected input
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the gesture
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
