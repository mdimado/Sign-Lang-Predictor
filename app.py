from flask import Flask, request, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json

app = Flask(__name__)
model = YOLO("/Users/mohammedimaduddin/Desktop/Developer/signlangpred/best.pt")  # Update this path to your model's location

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        image_bytes = file.read()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        results = model(image)
        detections = []
        for result in results:
            boxes = result.boxes
            boxes = boxes.cpu()
            # Assuming 'cls' is the predicted class index
            detected_index = int(boxes.cls.numpy()[0])  # This gets the detected class index
            detected_letter = chr(65 + detected_index)  # Convert index to letter (65 is ASCII for 'A')
            detections.append(detected_letter)
        detections_json = json.dumps(detections)  # Convert list to JSON string to pass to HTML
        return render_template('result.html', detections=detections_json)


if __name__ == '__main__':
    app.run(debug=True)
