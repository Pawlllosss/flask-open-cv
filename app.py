import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2

app = Flask(__name__, static_url_path="/static")
UPLOAD_FOLDER = "static/uploads/"
DOWNLOAD_FOLDER = "static/downloads/"
ALLOWED_EXTENSIONS = {"jpg", "png", "jpeg"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(path, filename):
    detect_object(path, filename)


def detect_object(path, filename):
    # List of class labels for object detection
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Generate random colors for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Paths to the SSD model files
    prototxt = "ssd/MobileNetSSD_deploy.prototxt.txt"
    model = "ssd/MobileNetSSD_deploy.caffemodel"

    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

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

            data = {
                "processed_img": 'static/downloads/' + filename,
                "uploaded_img": 'static/uploads/' + filename
            }

            return render_template("index.html", data=data)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
