import glob
import numpy as np
import cv2
import os
import Emotion
import functions
import matplotlib.pyplot as plt

from PIL import Image
from tabulate import tabulate
from tqdm import tqdm


def analyze(img_path: str, model_path: str,
            enforce_detection=True, detector_backend="opencv", align=True, silent=False):
    models = {"emotion": Emotion.load_model(model_path)}

    resp_objects = []

    img_objs = functions.extract_faces(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    img_content, img_region, _ = img_objs[0]
    if img_content.shape[0] > 0 and img_content.shape[1] > 0:
        obj = {}
        # facial attribute analysis
        pbar = tqdm(range(0, 1), desc="Finding actions", disable=silent)
        action = "emotion"
        pbar.set_description(f"Action: {action}")

        img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=0)

        emotion_predictions = models["emotion"].predict(img_gray, verbose=0)[0, :]

        sum_of_predictions = emotion_predictions.sum()

        obj["emotion"] = {}

        for i, emotion_label in enumerate(Emotion.labels):
            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
            obj["emotion"][emotion_label] = emotion_prediction

        obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

        # mention facial areas
        obj["region"] = img_region

        resp_objects.append(obj)

    return resp_objects


def save_face_image(img_path: str, y_true: str, y_predicted: str):
    image = Image.open(img_path)
    img_array = np.array(image)

    fig, ax = plt.subplots()
    ax.imshow(img_array)

    ax.text(0.5, 1.05, f"True emotion: {y_true}", transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='center')

    ax.text(0.5, -0.11, f"Predicted emotion: {y_predicted}", transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='center')

    image_dir = os.path.dirname(img_path)
    image_file_name = os.path.basename(img_path)
    pos = image_file_name.rfind('.')

    plt.savefig(image_dir + '/' + image_file_name[:pos] + f"_served_{y_predicted}" + image_file_name[pos:])


if __name__ == '__main__':
    work_dir = os.path.dirname(__file__)

    faces_files = sorted(glob.glob(rf"{work_dir}/test_x_images/*.jpg"))
    landmarked_faces_files = sorted(glob.glob(rf"{work_dir}/test_x_images_with_landmarks/*.jpg"))

    for image, landmarked_image in zip(faces_files, landmarked_faces_files):
        y = image[image.rfind('_')+1:].split('.')[0]
        y_landmark = landmarked_image[landmarked_image.rfind('_')+1:].split('.')[0]

        face_analysis = analyze(image, f"{work_dir}/facial_expression_model_weights_pure.h5")
        face_land_analysis = analyze(landmarked_image, f"{work_dir}/facial_expression_model_weights_landmarks.h5")

        if any(len(faces) > 1 for faces in [face_analysis, face_land_analysis]):
            raise NotImplemented("Images with multiple faces are not supported")

        save_face_image(image, y, face_analysis[0]['dominant_emotion'])
        save_face_image(landmarked_image, y_landmark, face_land_analysis[0]['dominant_emotion'])
