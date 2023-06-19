from face_finding import ai

def get_dict_emotions(directory):
    paths = ai.import_imgs_dir(directory)
    faces = ai.crop_faces(paths)
    return ai.detect_emotions(faces)

def get_dict_backgrounds(directory):
    paths = ai.import_imgs_dir(directory)
    return ai.detect_backgrounds(paths)

def get_recognized_faces2(directory, chosen_faces):
    paths = ai.import_imgs_dir(directory)
    faces = ai.crop_faces(paths)
    return ai.recognize2(faces, chosen_faces)

def get_filtered_images(paths, backgrounds, emotion):
    filtered_paths = ai.import_imgs(paths)
    results = []
    output = {}

    if(len(backgrounds) > 0):
        bg_result = ai.detect_backgrounds(filtered_paths)
        for key, value in bg_result.items():
            if value in backgrounds:
                results.append(key)
    if(len(backgrounds) == 0):
        results = paths

    if(emotion == "default"):
        output = {
        "emotion": "default",
        "paths": results
        }
        return output


    faces = ai.crop_faces(results)

    output = {
        "emotion": emotion,
        "paths": ai.detect_emotions2(faces, emotion)
    }

    return output


def get_recognized_faces(directory, chosen_image):
    chosen_image = ai.import_imgs_dir(chosen_image)
    faces_of_chosen_image = ai.crop_faces(chosen_image)
    paths = ai.import_imgs(directory)
    faces = ai.crop_faces(paths)
    return ai.recognize(faces, faces_of_chosen_image)