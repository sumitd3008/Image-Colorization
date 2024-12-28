from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from flask_cors import CORS


app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/*": {
    "origins": "*"}
})

# Directory paths
DIR = "D:/Projects_Sumit/colorize"

# Paths for the model
PROTOTXT = "./colorization_deploy_v2.prototxt"
MODEL = "./colorization_release_v2.caffemodel"
POINTS = "./pts_in_hull.npy"

# Load the model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


@app.route('/colorize', methods=['POST'])
def colorize_image():

    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Save the uploaded file with a unique name
    file = request.files['image']

    # Read the file as a byte array
    file_bytes = np.frombuffer(file.read(), np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    h, w = image.shape[:2]
    desired_width = 800
    aspect_ratio = w / h
    desired_height = int(desired_width / aspect_ratio)
    image_resized = cv2.resize(image, (desired_width, desired_height))

    scaled = image_resized.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image_resized.shape[1], image_resized.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    _, buffer = cv2.imencode('.png', colorized)
    colorized_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return the full URL to the colorized image
    return jsonify({
        "output": f'data:image/png;base64,{colorized_base64}'
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
