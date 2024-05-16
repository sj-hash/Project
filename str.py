import streamlit as st
import cv2
import face_recognition as frg
import yaml 
import os
import pickle

st.set_page_config(layout="wide")

# Function to recognize faces
def recognize(image, tolerance, encodings, names, red_list):
    # Load face encodings from the pickle file
    with open(encodings, 'rb') as f:
        data_encoding = pickle.load(f)
        list_encodings = data_encoding["encodings"]
        list_names = data_encoding["names"]

    # Perform face recognition
    face_locations = frg.face_locations(image)
    face_encodings = frg.face_encodings(image, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = frg.compare_faces(list_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = list_names[first_match_index]

        # Check if the name is in the red list
        if name in red_list:
            color = (0, 0, 255)  # Red color for rectangle
        else:
            color = (0, 255, 0)  # Green color for rectangle
        
        # Draw rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), color, 3)
        
        # Write the name on the image
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    
    return image

# Load configuration from YAML file
cfg = yaml.load(open('config.yaml','r'), Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']
VIDEO_PROMPT = cfg['INFO']['VIDEO_PROMPT']

# Load red list from the specified file
red_list_path = "red_list.txt"
if os.path.exists(red_list_path):
    with open(red_list_path, "r") as f:
        red_list = [line.strip() for line in f.readlines()]
else:
    red_list = []

st.sidebar.title("Settings")

# Create a menu bar
menu = ["Picture", "Webcam", "Video"]
choice = st.sidebar.selectbox("Input type", menu)

# Put slide to adjust tolerance
TOLERANCE = st.sidebar.slider("Tolerance", 0.0, 1.0, 0.5, 0.01)
st.sidebar.info("Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")

# Information section 
st.sidebar.title("Student Information")
name_container = st.sidebar.empty()
name_container.info('Name: Unknown')

if choice == "Picture":
    st.title("Face Recognition App")
    st.write(PICTURE_PROMPT)
    uploaded_images = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if len(uploaded_images) != 0:
        # Read uploaded image with face_recognition
        for image in uploaded_images:
            image = frg.load_image_file(image)
            image = recognize(image, TOLERANCE, "face_encodings_custom.pickle", "face_names_custom.pickle", red_list) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, channels="RGB")
    else: 
        st.info("Please upload an image")
    
elif choice == "Webcam":
    st.title("Face Recognition App")
    st.write(WEBCAM_PROMPT)

    # Initialize webcam
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Display webcam feed
    FRAME_WINDOW = st.image([])
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            st.info("Please turn off any other app that is using the camera and restart the app")
            break

        # Recognize faces
        image = recognize(frame, TOLERANCE, "face_encodings_custom.pickle", "face_names_custom.pickle", red_list)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the frame
        FRAME_WINDOW.image(image, channels="RGB")


elif choice == "Video":
    st.title("Face Recognition App")
    st.write(VIDEO_PROMPT)
    uploaded_video = st.file_uploader("Upload video", type=["mp4"])
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        st.video(video_bytes, format='video/mp4')

with st.sidebar.form(key='my_form'):
    st.title("Developer Section")
    submit_button = st.form_submit_button(label='REBUILD DATASET')
    if submit_button:
        with st.spinner("Rebuilding dataset..."):
            build_dataset()
        st.success("Dataset has been reset")
