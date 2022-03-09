from flask import Flask, request, jsonify

# from torch_utils import transform_image, get_prediction  # development
from app.torch_utils import transform_image, get_prediction  # production

app = Flask(__name__)

@app.route('/')
def hello():
    result = 'Hello there'
    return result

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')

        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            predicted_class_name = get_prediction(tensor)
            data = {'predicted class name': predicted_class_name}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})
