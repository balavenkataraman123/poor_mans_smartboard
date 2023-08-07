import cv2
import numpy as np
import airtec_processor
from pupil_apriltags import Detector

cameraname = "/dev/video0"
camw = 1280
camh = 720
cap = cv2.VideoCapture(cameraname)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camh)

cxc = []
cyc = []
tocalibrate = 1
gesture_buffer = []
height = 1360
width = 2560

drawing = np.zeros((height,width,3), np.uint8)
caliator = cv2.imread("calibrator.png")

at_detector = Detector(
    nthreads=18,
)

hg_detector = airtec_processor.NN_Processor("model.h5", "config.txt")
#pts_srcl = [[113.62127756, 246.069322  ],
# [572.38721893, 335.42686363],
# [ 80.47372266, 497.09077005],
# [565.83282172, 515.96108497]]

hcalc = 0
homographies = []
while True:
    while tocalibrate == 1:
        txc = [-69,-69,-69,-69]
        tyc = [-69,-69,-69,-69]        
        _, image = cap.read()
        if not _:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )
        print(tags)
        for i in tags:
            id = i.tag_id
            cx = i.center[0]
            cy = i.center[1]
            txc[id] = cx
            tyc[id] = cy
        if min(txc) != -69:
            cxc = txc
            cyc = tyc
            tocalibrate = 0
            cv2.destroyAllWindows()
            cv2.imwrite("calibrated.png", image)
            hcalc = 0
            break            
        cv2.imshow("deez", caliator)        
        cv2.imshow("nuts", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()            
            break
        
    while tocalibrate == 0:
        if hcalc == 0:
            np.set_printoptions(suppress=True)
            pts_src = np.array([[cxc[i],cyc[i]] for i in range(4)])
            #pts_src = np.array(pts_srcl)
            pts_dst = np.array([[150,150], [width-150,150],[150,height-150],[width-150,height-150]])    
            h, status = cv2.findHomography(pts_src, pts_dst)
            homographies.append(h)

            hcalc = 1

        _, frame = cap.read()
            

        handresult = hg_detector.processs(frame)
        if not handresult is None:
            coords = handresult[0]
            print(handresult)
            print(coords)
            if handresult[1] == "Pen Up" or handresult[1] == "Pen Down":
                print("yeeter")
                x = coords[0] * camw
                y = coords[1] * camh
                newcoords = homographies[0].dot(np.array([x,y,1]))
                newcoords /= newcoords[2]
                drawing = cv2.circle(drawing, (int(newcoords[0]),int(newcoords[1])), 5, (255,255,255),-1)    
                print(newcoords)
            
        cv2.imshow("deez" ,drawing)
        cv2.imshow("nuts" , frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()            
            break
    break

cap.release()
