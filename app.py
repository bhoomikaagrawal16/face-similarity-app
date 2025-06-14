import streamlit as st
import dlib
import numpy as np
import zipfile
import tempfile
import os
from PIL import Image

# Load dlib models once
@st.cache_resource
def load_models():
    face_detector = dlib.get_frontal_face_detector()
    pose_predictor = dlib.shape_predictor(".\models\posePredictor.dat")
    face_encoder = dlib.face_recognition_model_v1(".\models\dlib_face_recognition_resnet_model_v1.dat")
    return face_detector, pose_predictor, face_encoder

face_detector, pose_predictor, face_encoder = load_models()

# Encode face from image
def encodeFace(image):
    image = np.array(image.convert("RGB"))
    faces = face_detector(image, 1)
    if len(faces) == 0:
        return None
    face_location = faces[0]
    face_landmarks = pose_predictor(image, face_location)
    aligned_face = dlib.get_face_chip(image, face_landmarks)
    encodings = np.array(face_encoder.compute_face_descriptor(aligned_face))
    return encodings

# Compute similarity
def getSimilarity(enc1, enc2):
    return np.linalg.norm(enc1 - enc2)

# Streamlit UI
st.title("🔍 Face Similarity Finder")

# Threshold input box
threshold = st.number_input(
    "Set similarity threshold (lower = stricter match):",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01
)

target_img_file = st.file_uploader("Upload your face image", type=["jpg", "jpeg", "png"])
zip_file = st.file_uploader("Upload a zip of photos to search from", type=["zip"])

if target_img_file and zip_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save target image
        target_img_path = os.path.join(temp_dir, "target.jpg")
        with open(target_img_path, "wb") as f:
            f.write(target_img_file.read())
        target_img = Image.open(target_img_path)
        target_enc = encodeFace(target_img)

        if target_enc is None:
            st.error("No face found in the uploaded target image.")
            st.stop()

        # Extract zip
        zip_path = os.path.join(temp_dir, "images.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        extract_path = os.path.join(temp_dir, "gallery")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Gather images recursively
        gallery_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    gallery_files.append(os.path.join(root, file))

        if not gallery_files:
            st.error("Zip file is empty or contains no readable images.")
            st.stop()

        results = []

        for img_path in gallery_files:
            try:
                img = Image.open(img_path)
                enc = encodeFace(img)
                if enc is not None:
                    dist = getSimilarity(target_enc, enc)
                    results.append((img, os.path.basename(img_path), dist))
            except:
                continue

        # Filter by threshold
        filtered_results = [r for r in results if r[2] <= threshold]

        if filtered_results:
            filtered_results.sort(key=lambda x: x[2])
            st.subheader("Top Matches:")
            for img, name, dist in filtered_results[:5]:
                st.image(img, caption=f"{name} | Distance: {round(dist, 4)}", width=250)
        else:
            st.warning("No matching faces found below the given threshold.")
