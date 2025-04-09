# Face Recognition Alarm System

This repository contains a face recognition alarm system that integrates Python and C code to detect faces, recognize individuals, and send alerts via Telegram. The system uses a combination of machine learning models, ONNX runtime, and a web server to process live video feeds and trigger alarms for unrecognized faces.

## Features

- **Live Face Detection**: Captures live video feed and detects faces using Haar cascades and MTCNN.
- **Face Recognition**: Compares detected faces against a reference embedding using cosine similarity.
- **Telegram Alerts**: Sends alerts with the detected face image to a specified Telegram chat for unrecognized faces.
- **ONNX Runtime Integration**: Uses ONNX models for efficient inference in C.
- **HTTP Trigger**: A lightweight HTTP server in C processes triggers from the Python application.

## Repository Structure

### File Descriptions

1. **`face_model_training.py`**:
   - Trains a face recognition model using a pre-trained ONNX backbone.
   - Fine-tunes the model with a custom dataset and exports it to ONNX format.
   - Generates a reference embedding for face recognition.

2. **`live_feed_detect_face.py`**:
   - Captures live video feed using Picamera2.
   - Detects faces using Haar cascades and crops them using MTCNN.
   - Sends an HTTP trigger to the C program when a face is detected.

3. **`face_recognition_alarm.c`**:
   - Runs an HTTP server to process triggers from the Python script.
   - Loads the ONNX model and performs face recognition using cosine similarity.
   - Sends Telegram alerts for unrecognized faces.

## Setup Instructions

### Prerequisites

- **Python**:
  - Install Python 3.7 or higher.
  - Install required Python packages:
    ```bash
    pip install opencv-python flask picamera2 mtcnn torch torchvision onnx onnx2pytorch insightface
    ```

- **C**:
  - Install ONNX Runtime C API.
  - Install `libmicrohttpd` for the HTTP server.
  - Install `libcurl` for Telegram alerts.

### Configuration

1. **Telegram Bot Setup**:
   - Create a Telegram bot using [BotFather](https://core.telegram.org/bots#botfather).
   - Replace `TELEGRAM_BOT_TOKEN` and `CHAT_ID` in `face_recognition_alarm.c` with your bot token and chat ID.

2. **Model and Dataset**:
   - Update paths in `face_model_training.py` to point to your dataset and save locations.
   - Train the model and export it to ONNX format.

3. **Reference Embedding**:
   - Use a clear image of yourself to generate a reference embedding using `face_model_training.py`.

4. **Directory Setup**:
   - Ensure the directory `/home/olafu/LV/face_detect` exists for saving detected face images.

### Running the System

1. **Train the Model**:
   - Run `face_model_training.py` to train and export the model:
     ```bash
     python face_model_training.py
     ```

2. **Start the C HTTP Server**:
   - Compile and run `face_recognition_alarm.c`:
     ```bash
     gcc face_recognition_alarm.c -o face_recognition_alarm -lm -lonnxruntime -lmicrohttpd -lcurl
     ./face_recognition_alarm
     ```

3. **Start the Python Face Detection**:
   - Run `live_feed_detect_face.py` to start the live feed and face detection:
     ```bash
     python live_feed_detect_face.py
     ```

4. **Access the Video Feed**:
   - Open a browser and navigate to `http://<your-device-ip>:5000/video_feed` to view the live video feed.

### Alerts

- When an unrecognized face is detected, the system sends a Telegram alert with the detected face image.

## Troubleshooting

- **ONNX Runtime Errors**: Ensure the ONNX Runtime C API is installed and the model path is correct.
- **Telegram Alerts Not Sent**: Verify the bot token, chat ID, and internet connectivity.
- **Face Detection Issues**: Ensure the camera is properly configured and the Haar cascade/MTCNN models are working.

## Acknowledgments

- [ONNX Runtime](https://onnxruntime.ai/)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [libmicrohttpd](https://www.gnu.org/software/libmicrohttpd/)
- [stb_image](https://github.com/nothings/stb)
