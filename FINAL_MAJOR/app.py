from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
NUM_FRAMES = 16

# Load the video classification model
model = tf.keras.models.load_model("model.h5")

# Initialize the Flask app
app = Flask(__name__)

# Define a function to preprocess the video frames
def preprocess_frame(frame):
    # Resize the frame to the input shape of the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalize the pixel values to the range [0, 1]
    normalized_frame = rgb_frame / 255.0
    # Add a batch dimension to the frame
    batched_frame = np.expand_dims(normalized_frame, axis=0)
    return batched_frame

# Define a route for the webpage
@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('new1.html')

# Define a route to handle the video classification
@app.route('/classify_video', methods=['POST'])
def classify_video():
    # Get the video file from the request
    file = request.files['file']

    # Read the video using OpenCV
    cap = cv2.VideoCapture(file)
    frames = []

    # Extract frames from the video and preprocess them
    while len(frames) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame (e.g. resize and normalize)
        frame = cv2.resize(frame, (224, 224))
        frame = (frame / 255.0) - 0.5
        frames.append(frame)
    cap.release()

    # Pad or truncate the frames to make sure they all have the same length
    frames = np.array(frames)
    if frames.shape[0] < NUM_FRAMES:
        padding = np.zeros((NUM_FRAMES - frames.shape[0], 224, 224, 3))
        frames = np.concatenate((frames, padding))
    else:
        frames = frames[:NUM_FRAMES]

    # Run the frames through the video classification model
    preds = model.predict(np.array([frames]))
    classes = np.argmax(preds, axis=1)

    # Map the class IDs to class names
    class_names = ["OutdoorLaunchpad", "Satellite", "Vehicle", "PersonCloseUp", "OutdoorAntenna", "Mountain", "Sky", "Speech", "Traffic", "Logo",
                "IndoorLab", "Crowd", "Graphics"]
    results = [{'class': class_names[i], 'confidence': float(preds[0][i])} for i in range(len(class_names))]
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # Return the classification results as a JSON response
    return jsonify(results)
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)