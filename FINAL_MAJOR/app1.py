import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained video classification model
model = load_model('model.h5')

# Define the classes for the model
classes = ["OutdoorLaunchpad", "Satellite", "Vehicle", "PersonCloseUp", "OutdoorAntenna", "Mountain", "Sky", "Speech", "Traffic", "Logo",
                "IndoorLab", "Crowd", "Graphics"]

# Define a function to preprocess the video frames
def preprocess_frame(frame):
    # Resize the frame to the size required by the model
    frame = cv2.resize(frame, (224, 224))
    # Convert the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize the pixel values
    frame = frame / 255.0
    # Expand the dimensions of the frame to match the input shape of the model
    frame = np.expand_dims(frame, axis=0)
    return frame

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Video Classification App')
    
    # Create a file uploader to upload a video file
    uploaded_file = st.file_uploader('Upload a video file', type=['mp4'])
    
    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the video file using OpenCV
        video = cv2.VideoCapture(uploaded_file)
        
        # Loop over each frame of the video
        while True:
            # Read a frame from the video
            ret, frame = video.read()
            
            # If there are no more frames, break out of the loop
            if not ret:
                break
            
            # Preprocess the frame
            processed_frame = preprocess_frame(frame)
            
            # Make a prediction using the model
            prediction = model.predict(processed_frame)
            
            # Get the index of the predicted class
            predicted_class = np.argmax(prediction[0])
            
            # Get the name of the predicted class
            class_name = classes[predicted_class]
            
            # Display the frame and the predicted class
            st.image(frame, channels='BGR', caption=class_name, use_column_width=True)
    
    # If no file was uploaded, display a message
    else:
        st.write('Please upload a video file')

if __name__ == '__main__':
    main()
