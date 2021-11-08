import cv2
import math
import time
import HandTrackingModule_Second as HTM
import numpy as np
# to control the volume: copy from https://github.com/AndreMiras/pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Params
wCam, hCam = 640, 480
pTime = 0
vol = 0

# video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Create an object
HandObject = HTM.HandDetector(detectionCon=0.7)

# to control the volume: copy from https://github.com/AndreMiras/pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# if we print this we can see the range of our volume is (-65.25, 0.0, 0.03125)
# 0.0 is the max and 65 is the min
# print(volume.GetVolumeRange())
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# if you put the value to 0.0 it will be max in your pc
# volume.SetMasterVolumeLevel(-65.0, None)

while True:
    # read the video
    success, img = cap.read()
    
    # now we can find the hands in the image
    img = HandObject.Findhands(img)

    # to get the positions of the all 21 landmarks on the hand
    lmList = HandObject.FindPosition(img, draw=False)
    if len(lmList) != 0:
        # 4 = thump_tip, 8 = index_finger_tip
        # print(lmList[4], lmList[8])
        # the format of the lmList [4, x_position, y_posisiotn]
        x_4, y_4 = lmList[4][1], lmList[4][2]
        x_8, y_8 = lmList[8][1], lmList[8][2]
        cv2.circle(img, (x_4, y_4), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x_8, y_8), 15, (255, 0, 0), cv2.FILLED)
        # draw a line between them
        cv2.line(img, (x_4, y_4), (x_8, y_8), (255, 0, 0), 3)
        # get the center of this line
        cx, cy = (x_4+x_8)//2, (y_4+y_8)//2
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        # we need to know the length of this line
        length = math.hypot(x_8-x_4, y_8-y_4)
        # print(length) the max=300, min=50  almost
        # just change the color of circle when we are lower than min
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # lets control the volume: pip install pycaw
        # Hand range: 50 - 300,   Vol range: -65.25 - 0.0
        # first we need to convert the hand range ----> vol range
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        # print(vol)
        # send it to the master volume level:
        volume.SetMasterVolumeLevel(vol, None)

    # lets add a volume bar to the img
    # cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0))
    # cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), cv2.FILLED)

    # calculate the FBS
    cTime = time.time()
    fbs = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FBS: {int(fbs)}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 0), 1)
 
    # show the image
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
