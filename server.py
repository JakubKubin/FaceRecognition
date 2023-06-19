import logging
from flask import Flask, request
from face_finding import controller

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

@app.route("/get_dict_emotions", methods=['POST', 'GET'])
def get_emotions():
    data = request.get_json()
    directory = data['path']
    return controller.get_dict_emotions(directory)


@app.route("/get_dict_backgrounds", methods=['POST', 'GET'])
def get_backgrounds():
    data = request.get_json()
    directory = data['path']
    return controller.get_dict_backgrounds(directory)

@app.route("/get_recognized_faces", methods=['POST', 'GET'])
def get_recognized_faces():
    data = request.get_json()
    directory = data['path1']
    chosen_image = data['path2']
    return controller.get_recognized_faces(directory, chosen_image)

@app.route("/get_recognized_faces2", methods=['POST', 'GET'])
def get_recognized_faces2():
    data = request.get_json()
    directory = data['path']
    chosen_faces = data['faces']
    return controller.get_recognized_faces2(directory, chosen_faces)

@app.route("/get_filtered_images", methods=['POST', 'GET'])
def get_filtered_images():
    data = request.get_json()
    paths = data['paths']
    backgrounds = data['backgrounds']
    emotion = data['emotion']

    return controller.get_filtered_images(paths, backgrounds, emotion)

def run_server_api():
    app.run(host='0.0.0.0', port=8080)

if __name__ == "__main__":
    run_server_api()