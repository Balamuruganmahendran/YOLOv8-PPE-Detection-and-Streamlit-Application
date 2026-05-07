import streamlit as st
import os
from ultralytics import YOLO
import shutil # Import shutil for directory operations

st.set_page_config(layout="wide", page_title="YOLOv8 PPE Detector")

st.title('👷 YOLOv8 PPE Detection & Visualization')
st.markdown("--- ")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = '/content/runs/ppe_model/weights/best.pt'
    return YOLO(model_path)

model = load_model()

# --- Sidebar for Upload and Settings ---
st.sidebar.header('Upload & Settings')
uploaded_file = st.sidebar.file_uploader("Choose an image or video file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider('IOU Threshold', 0.0, 1.0, 0.7, 0.05)

# --- Main Content Area ---

if uploaded_file is not None:
    st.subheader(f"Processing: {uploaded_file.name}")
    # Create a temporary directory to save the uploaded file
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_type = uploaded_file.type

    if file_type.startswith('image'):
        st.image(file_path, caption='Original Image', use_column_width=True)
    elif file_type.startswith('video'):
        st.video(file_path, format=file_type)

    st.markdown("### Running Prediction...")
    with st.spinner('Detecting objects...'):
        try:
            # Clear previous prediction results to avoid confusion
            predict_base_dir = '/content/runs/detect'
            if os.path.exists(predict_base_dir):
                for item in os.listdir(predict_base_dir):
                    item_path = os.path.join(predict_base_dir, item)
                    if os.path.isdir(item_path) and item.startswith('predict'):
                        shutil.rmtree(item_path)

            results = model.predict(source=file_path, save=True, conf=confidence_threshold, iou=iou_threshold)

            # Get the path to the latest prediction folder
            # YOLO saves results in a new folder each time (predict, predict2, etc.)
            subdirs = [os.path.join(predict_base_dir, d) for d in os.listdir(predict_base_dir) if os.path.isdir(os.path.join(predict_base_dir, d)) and d.startswith('predict')]
            subdirs.sort(key=os.path.getmtime, reverse=True)

            if subdirs:
                prediction_output_dir = subdirs[0]
                st.subheader(f'Prediction Results for {uploaded_file.name}:')

                output_files = [f for f in os.listdir(prediction_output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4'))]

                if output_files:
                    # Prioritize displaying video if it's a video prediction
                    video_files = [f for f in output_files if f.lower().endswith('.mp4')]
                    if video_files:
                        video_path = os.path.join(prediction_output_dir, video_files[0])
                        st.video(video_path)
                        with open(video_path, "rb") as file:
                            st.download_button(
                                label="Download Predicted Video",
                                data=file.read(),
                                file_name=f"predicted_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                    else:
                        # Display images in columns
                        image_files_to_display = sorted([f for f in output_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        if image_files_to_display:
                            for image_file in image_files_to_display:
                                image_path = os.path.join(prediction_output_dir, image_file)
                                st.image(image_path, caption=f'Predicted: {image_file}', use_column_width=True)
                                with open(image_path, "rb") as file:
                                    st.download_button(
                                        label=f"Download {image_file}",
                                        data=file.read(),
                                        file_name=f"predicted_{image_file}",
                                        mime="image/jpeg" # Assuming mostly JPEGs
                                    )
                        else:
                            st.warning("No displayable output files (images/video) found in the prediction results.")
                else:
                    st.warning("No output files found in the prediction results.")
            else:
                st.error("Could not find the prediction output directory.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
        finally:
            # Clean up the temporary uploaded file and directory
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
else:
    st.info("Upload an image or video using the sidebar to get started!")

st.markdown('---')
st.markdown('**Instructions:** Use the sidebar to upload your media file and adjust detection parameters. The model will then process your input and display the detected objects.')
