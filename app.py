import os
from flask import Flask, request, redirect, render_template, flash, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import boto3

app = Flask(__name__, static_url_path="/static")
UPLOAD_FOLDER = "static/uploads/"
DOWNLOAD_FOLDER = "static/downloads/"
ALLOWED_EXTENSIONS = {"jpg", "png", "jpeg"}

# Global variables for the model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = None  # Global variable for the model

# Initialize the S3 client
s3 = boto3.client('s3')

sagemaker_client = boto3.client('runtime.sagemaker')
sagemaker_endpoint_name = 'xgboost-2024-09-27-15-28-12-384'

# Temporary directory to store the downloaded files
TMP_DIR = "/tmp"


def download_from_s3(bucket_name, s3_key, local_file_path):
    """Downloads a file from an S3 bucket to a local path."""
    s3.download_file(bucket_name, s3_key, local_file_path)


def load_model():
    """Loads the pre-trained SSD model."""
    global net

    # S3 bucket information
    bucket_name = "open-cv-model-repo"
    model_s3_key = "ssd/MobileNetSSD_deploy.caffemodel"
    prototxt_s3_key = "ssd/MobileNetSSD_deploy.prototxt.txt"

    # Paths to save the downloaded files locally
    prototxt_local_path = os.path.join(TMP_DIR, "MobileNetSSD_deploy.prototxt.txt")
    model_local_path = os.path.join(TMP_DIR, "MobileNetSSD_deploy.caffemodel")

    download_from_s3(bucket_name, prototxt_s3_key, prototxt_local_path)
    download_from_s3(bucket_name, model_s3_key, model_local_path)

    # Load the pre-trained model using OpenCV's DNN module
    net = cv2.dnn.readNetFromCaffe(prototxt_local_path, model_local_path)


# Ensure the model is loaded only once when the app starts
@app.before_request
def ensure_model_loaded():
    global net
    if net is None:
        load_model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(path, filename):
    detect_object(path, filename)


def detect_object(path, filename):
    """Detects objects in the provided image using the preloaded model."""
    global net  # Use the globally loaded model
    # Load the input image and get its dimensions
    image = cv2.imread(path)
    (h, w) = image.shape[:2]

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform forward pass to get the detections
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.60:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)

            # Adjust the position for the label to not overlap with the box
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Save the processed image
    cv2.imwrite(f"{DOWNLOAD_FOLDER}{filename}", image)


@app.route('/iris', methods=['PUT'])
def iris_infer():
    request_json = request.json
    model_request_parameters = [
        request_json['sepalLength'],
        request_json['sepalWidth'],
        request_json['petalLength'],
        request_json['petalWidth']
    ]

    sagemaker_body = ','.join(map(str, model_request_parameters))

    response = sagemaker_client.invoke_endpoint(
        EndpointName=sagemaker_endpoint_name,
        ContentType='text/csv',
        Body=sagemaker_body,
    )

    # Parse the response from SageMaker
    result = response['Body'].read().decode('utf-8')

    # Return the prediction result as JSON
    return jsonify({'prediction': result})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the 'file' is in the request
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)

        file = request.files['file']

        # Check if no file was selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Check if the file is allowed and save it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            process_file(os.path.join(UPLOAD_FOLDER, filename), filename)

            processed_image_path = os.path.join(DOWNLOAD_FOLDER, filename)
            upload_bucket = "open-cv-upload-bucket"
            s3.upload_file(processed_image_path, upload_bucket, f"processed/{filename}")

            data = {
                "processed_img": 'static/downloads/' + filename,
                "uploaded_img": 'static/uploads/' + filename
            }

            return render_template("index.html", data=data)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
