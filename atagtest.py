import cv2
import numpy as np
from pupil_apriltags import Detector

at_detector = Detector(
    nthreads=18,
)

image = cv2.imread("view.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tags = at_detector.detect(
    image,
    estimate_tag_pose=False,
    camera_params=None,
    tag_size=None,
)
print(tags)
