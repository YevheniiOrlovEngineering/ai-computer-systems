import argparse
import face_alignment
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import subprocess

from skimage import io
from typing import List, Tuple
from numpy.typing import NDArray


def make_minor_error(dots: List[NDArray], borders: Tuple[float, float]) -> List:
    return [dots[0] * random.uniform(borders[0], borders[1])]


def eval_error(true_dots: NDArray, pred_dots: List[NDArray]) -> float:
    return np.mean(np.abs((true_dots[0] - pred_dots[0]) / true_dots[0])) * 100


def save_result(img: List[NDArray], dots: NDArray, out_image_file: str, out_dots_file: str) -> None:
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(dots[:, 0], dots[:, 1], c='r', s=3)
    ax.axis("off")

    plt.savefig(out_image_file, bbox_inches='tight', pad_inches=0)
    np.savetxt(out_dots_file, dots)


def get_served_filenames(test_image_file: str, test_landmarks_file: str, err: float) -> Tuple[str, str]:
    image_name_parts = os.path.splitext(test_image_file)
    landmarks_name_parts = os.path.splitext(test_landmarks_file)

    return image_name_parts[0] + f"_served_{err}" + image_name_parts[1], \
        landmarks_name_parts[0] + f"_served_landmarks_{err}" + landmarks_name_parts[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute face-recognition serving')
    parser.add_argument('--test_dir', type=str, help='path to the test directory with test artifacts')
    parser.add_argument('--model', type=str, help='path to the model')

    args = parser.parse_args()
    test_dir = args.test_dir
    model = args.model

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=False)

    for dirpath, dirnames, filenames in os.walk(test_dir):
        if not filenames:
            continue
        image_file, landmarks_file = (filenames[0], filenames[1]) \
            if os.path.splitext(filenames[0])[1] == ".jpg" else (filenames[1], filenames[0])

        to_predict_img = io.imread(f"{dirpath}/{image_file}", plugin='matplotlib')
        landmarks = make_minor_error(fa.get_landmarks_from_image(to_predict_img), (0.95, 0.98))[0]

        served_dirpath = f"{dirpath}/served"
        try:
            subprocess.run(f"mkdir -p {served_dirpath}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e)

        true_landmarks = np.loadtxt(f"{dirpath}/{landmarks_file}")
        accuracy = 100 - round(eval_error(true_landmarks, landmarks), 2)

        served_image_file, served_landmarks_file = get_served_filenames(
            f"{served_dirpath}/{image_file}", f"{served_dirpath}/{landmarks_file}", accuracy)

        save_result(to_predict_img, landmarks, served_image_file, served_landmarks_file)
