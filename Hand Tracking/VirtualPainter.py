# i did not compelet this, but i complete the handtracking module
import cv2
import HandTrackingModule_Second as HTM
import time
import numpy as np
import os

# import images
FolderPath = 'Painter'
ImagesList = []
for imgPath in os.listdir(FolderPath):
    image = cv2.imread(f'{FolderPath}/{imgPath}')
    ImagesList.append(image)

# run webcam
header = ImagesList[0]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# params
pTime = 0
drawColor = (255, 0, 255)
BrushThickness = 15
EraserThickness = 50
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Hand Object
HandObject = HTM.HandDetector(detectionCon=0.5)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # 1. Find hand landmarks
    img = HandObject.Findhands(img)
    landmarkList = HandObject.FindPosition(img, draw=False)
    if len(landmarkList) != 0:
        # index and middle finger tip
        x1, y1 = landmarkList[8][1:]
        x2, y2 = landmarkList[12][1:]
        # 2. Check which fingers are up
        FingersNum = HandObject.FingersUp()
        # print(FingersNum)

        # 3.selection mode (2 index and middle finger tips are up)
        if FingersNum[1] and FingersNum[2]:
            xp, yp = 0, 0
            # print('Selection mode')
            # if we are at the top of the picture
            # if we are in the header
            if y1 < 125:
                if 250 < x1 < 450:
                    header = ImagesList[0]
                    drawColor = (255, 0, 0)
                elif 650 < x1 < 800:
                    header = ImagesList[1]
                    drawColor = (255, 255, 255)
                elif 950 < x1 < 1200:
                    header = ImagesList[2]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

        # 4. if one finger up we have the drawing
        if FingersNum[1] and FingersNum[2] == False:
            # print('Draw mode')
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # draw
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, EraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, EraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, BrushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, BrushThickness)

            xp, yp = x1, y1

        # setting the header image
        img[0:125, 0:1280] = header



    cTime = time.time()
    fbs = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the resulting frame
    # combining two images of img and imgCanvas together
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow('frame', img)
    #cv2.imshow('frame2', imgCanvas)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break