import cv2
from deepface import DeepFace
import os
import base64
import numpy as np

from retinaface.src.retinaface import RetinaFace

detector = RetinaFace()

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string

def decode_base64_to_image(baseimage):
    decoded_data = base64.b64decode(baseimage)
    np_array = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

def import_imgs(paths):
    filtered_paths = [path for path in paths if (".png" in path or ".jpg" in path)]
    if not filtered_paths:
        print("Error: No pictures provided!")
    return paths

def import_imgs_dir(directory):
    try:
        paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        paths = [path for path in paths if (".png" in path or ".jpg" in path)]
        if not paths:
            print(f"Error: No pictures in {directory} directory")
            exit(1)
        return paths
    except:
        return [directory]

def crop_faces(paths):
    array_of_faces_with_path = []
    for image_path in paths:
        try:
            image = detector.read(image_path)
            image_rgb = cv2.imread(image_path)
            faces = detector.predict(image)
        except:
            faces = detector.predict(image_path)

        for face in faces:
            x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
            face_image = image_rgb[y1:y2, x1:x2]
            array_of_faces_with_path.append([image_path, face_image])
    return array_of_faces_with_path

def recognize(faces, faces_of_chosen_image):
    dictionary = {}
    i = 0
    for chosen_picture_face in faces_of_chosen_image:
        for face in faces:
            obj = DeepFace.verify(face[1], chosen_picture_face[1], model_name='ArcFace', detector_backend='opencv', enforce_detection=False)
            if obj["verified"]:

                data = {}

                data['path1'] = chosen_picture_face[0]
                data['face1'] = encode_image_to_base64(chosen_picture_face[1])

                data['path2'] = face[0]
                data['face2'] = encode_image_to_base64(face[1])

                dictionary[i] = data
                i+=1

    return dictionary


def recognize2(faces, chosen_faces):
    paths = []
    for chosen_face in chosen_faces:
        for face in faces:
            obj = DeepFace.verify(face[1], chosen_face, model_name='ArcFace', detector_backend='opencv', enforce_detection=False)
            if obj["verified"]:
                paths.append(face[0])
    return {"paths": paths}

def detect_emotions(faces):
    dictionary = {}
    i = 0
    for face in faces:
        emotion = DeepFace.analyze(
            face[1], actions=("emotion"), enforce_detection=False
        )

        data = {}
        data['path'] = face[0]
        data['face'] = encode_image_to_base64(face[1])
        data['emotion'] = emotion[0]["dominant_emotion"]

        dictionary[i] = data
        i +=1

    return dictionary

def detect_emotions2(faces, emotion):
    result = []

    for face in faces:
        if face[0] in result:
            continue
        analyzed_emotion = DeepFace.analyze(
            face[1], actions=("emotion"), enforce_detection=False
        )

        if(emotion == analyzed_emotion[0]["dominant_emotion"]):
            result.append(face[0])
    return result



def detect_backgrounds(paths):
    results = {}
    for image_path in paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image], [1], None, [256], [0, 256])
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(gray, 50, 150)
        texture = cv2.mean(edges)[0]

        if hist[100] > 5000 and texture < 50:
            results[image_path] = "outdoor"
        else:
            results[image_path] = "indoor"
    return results