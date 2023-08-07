import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math
mp_hands = mp.solutions.hands

class NN_Processor():
    modelpath = ""
    gestures = []
    pointer = 0
    def __init__(self, mfilepath, conf_file, pointer=8):
        self.model = tf.keras.models.load_model(mfilepath)
        f = open(conf_file, "r")
        self.gestures = [i.strip() for i in f.readlines()]
        f.close()
        self.hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
        self.pointer = pointer
    def processs(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for handno, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if 1 == 1:
                    xcoords = [handmark.x for handmark in hand_landmarks.landmark]
                    ycoords = [handmark.y for handmark in hand_landmarks.landmark]
                    minx = min(xcoords)
                    maxx = max(xcoords)
                    miny = min(ycoords)
                    maxy = max(ycoords)
                    xcoords1 = [(i - minx) / (maxx-minx) for i in xcoords]
                    ycoords1 = [(i - miny) / (maxy-miny) for i in ycoords]
                    temp = []
                    for i in range(21):
                        temp.append(xcoords1[i])
                        temp.append(ycoords1[i])
                        temp.append(hand_landmarks.landmark[i].z)
                    ans = list(self.model.predict([temp])[0])
                    if max(ans) > 0.85:        
                        ans_gesture = self.gestures[ans.index(max(ans))]
                        return ((xcoords[self.pointer], ycoords[self.pointer]), ans_gesture)
#class Hand_location_tracker():
#class Cosine_Similarity_Processor():
    #self.vector_base
