import dlib
import cv2
import numpy as np
import shutil
from tqdm import tqdm
# used for accessing url to download files
import urllib.request as urlreq
import glob, os
import json
import pprint
import time

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def generate_landmarks(img_dir, skip=1):
    filenames = []

    for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    filenames.append(os.path.join(root, file))
    landmark_cords = {}
    count = 0

    for i in tqdm(range(len(filenames))):
        if i%skip == 0:
            image_file = filenames[i]
            image = cv2.imread(image_file)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_cropped = image_rgb

            # create a copy of the cropped image to be used later
            image_template = image_cropped.copy()
            # convert image to Grayscale
            image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
            # Detect faces using dlib on the "grayscale image"

            try:
                faces = detector(image_gray)
                for face in faces:
                    landmarks = predictor(image=image_gray, box=face)
                    coord_list = []
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        coord_list.append([x, y])
                    landmark_cords[image_file] = coord_list
            except Exception as e:
                print(e)

    save_name = 'landmarks.json'
    with open(save_name, 'w') as fp:
        json.dump(landmark_cords, fp)

start = time.time()
generate_landmarks("C:\Research paper\DFD\images\FFHQ")
end = time.time()
print(f"it took {end-start} seconds")

