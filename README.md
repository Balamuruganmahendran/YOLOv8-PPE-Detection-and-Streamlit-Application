# YOLOv8 PPE Detection and Streamlit Application

## Project Overview
This project implements a Personal Protective Equipment (PPE) detection system using the YOLOv8 object detection model. It includes data ingestion, organization, model training, performance evaluation, and a user-friendly Streamlit web application for real-time inference.

## Features
-   **Automated Data Ingestion**: Downloads the PPE dataset from KaggleHub.
-   **Data Organization**: Structures the dataset for YOLOv8 training.
-   **YOLOv8 Model Training**: Trains a YOLOv8 nano model on the PPE dataset to detect various safety equipment and personnel.
-   **Model Performance Evaluation**: Reports key metrics like Precision, Recall, mAP50, and mAP50-95 for overall and per-class performance.
-   **Streamlit Web Application**: 
    -   Interactive UI for uploading images or videos.
    -   Real-time object detection with the trained YOLOv8 model.
    -   Adjustable confidence and IOU thresholds.
    -   Displays original and predicted media side-by-side.
    -   Download options for predicted outputs.

## Setup and Installation

To run this project, you'll need a Google Colab environment or a similar Python environment with GPU support. Follow these steps:

1.  **Clone the Repository** (if applicable, otherwise follow the Colab notebook steps).
2.  **Install Dependencies**:
    The Colab notebook cells handle most installations, but ensure you have `ultralytics`, `kagglehub`, `pyngrok`, and `streamlit` installed.
    ```bash
    !pip install ultralytics kagglehub pyngrok streamlit
    ```
3.  **Kaggle API Key (Optional)**: If directly downloading from Kaggle without `kagglehub`, ensure your Kaggle API key is configured.
4.  **ngrok Authentication Token**: For deploying the Streamlit app publicly, you need an ngrok authentication token. Obtain one from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) and set it in your environment or directly in the Colab cell:
    ```python
    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
    ```

## Usage

### 1. Data Ingestion & Preparation
Run the initial Colab cells to download and organize the PPE dataset.

### 2. Model Training
Execute the model training cells. The `model.train()` function will train a YOLOv8n model and save the best weights to `/content/runs/ppe_model/weights/best.pt`.

### 3. Model Evaluation
Run the evaluation cell to see the precision, recall, and mAP scores for the trained model.

### 4. Run the Streamlit Application
Once the model is trained and `app.py` is created (or updated by the agent):

1.  Execute the Streamlit deployment cell (`!streamlit run app.py ...`).
2.  Click the ngrok public URL provided in the output to access your application.

#### Streamlit App Interaction:
-   **Upload**: Use the sidebar to upload an image (`.jpg`, `.png`) or a video (`.mp4`, `.avi`, `.mov`).
-   **Adjust Settings**: Use the sliders in the sidebar to modify the detection confidence threshold and IOU (Intersection Over Union) threshold.
-   **View Results**: The application will display your original media and the predicted output with bounding boxes.
-   **Download**: Download buttons are provided for the processed images/videos.

## Project Structure (Key Files)

-   `app.py`: The Streamlit application code for inference and visualization.
-   `data.yaml`: YOLOv8 configuration file specifying dataset paths and class names.
-   `/content/runs/ppe_model/weights/best.pt`: The trained YOLOv8 model weights.
-   `/content/datasets/ppe/`: Directory containing the downloaded and organized dataset.

## Model Details
-   **Model Architecture**: YOLOv8n (nano version).
-   **Dataset**: Construction Site Safety Image Dataset (from Roboflow via KaggleHub).
-   **Classes**: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle.

## Results
The model achieves the following approximate performance metrics on the validation set:
-   **Overall Precision (P)**: 0.892
-   **Overall Recall (R)**: 0.687
-   **Overall mAP50**: 0.779
-   **Overall mAP50-95**: 0.461

## Output Screenshots

### Screenshot 1
![App UI](<img width="1920" height="1080" alt="Screenshot (2)" src="https://github.com/user-attachments/assets/c6cfeb16-4ce7-45c5-833f-31aeeaa7737f" />
)

### Screenshot 2 
![Actual Image Uploading ](<img width="1920" height="1080" alt="Screenshot (3)" src="https://github.com/user-attachments/assets/cd2dac4b-54f4-4f7e-8e26-1216fc4486ee" />
)


### Screenshot 3 
![Prediction Image](<img width="1920" height="1080" alt="Screenshot (5)" src="https://github.com/user-attachments/assets/c13db184-1e8d-4cbf-9016-6615155669b4" />
)

## Author

**Balamurugan Mahendran**
