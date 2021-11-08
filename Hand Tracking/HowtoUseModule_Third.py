import cv2
import mediapipe as mp
import time

import HandTrackingModule_Second as HTM

pTime = 0
cTime = 0
# define a video capture object
cap = cv2.VideoCapture(0)
DetectorObject = HTM.HandDetector()
while True:
    # Capture the video frame by frame
    success, img = cap.read() 

    img = DetectorObject.Findhands(img)
    # you can change the draw value to True
    lmList = DetectorObject.FindPosition(img, draw=False)
    if len(lmList) !=0:
        # Thumb position
        print(lmList[4])
    cTime = time.time()
    fbs = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fbs)), (10,70), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)

    # Display the resulting frame
    cv2.imshow('frame', img)      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break