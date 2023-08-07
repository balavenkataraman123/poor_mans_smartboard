import copy
import time
import argparse

import cv2 as cv
from pupil_apriltags import Detector

def main():
    cap_device = 4
    cap_width = 1280
    cap_height = 720


    nthreads = 6
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    at_detector = Detector(
        nthreads=nthreads,
    )
    elapsed_time = 0
    while True:
        start_time = time.time()
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )
        print(tags)
    cap.release()

if __name__ == '__main__':
    main()