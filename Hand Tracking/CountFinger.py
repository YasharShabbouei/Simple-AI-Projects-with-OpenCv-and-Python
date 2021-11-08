import cv2
import HandTrackingModule_Second as HTM
import time
import os

# Params
wCam, hCam = 680, 420
pTime = 0

# video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# import images
path = 'FingerImages'
# myList = os.listdir(path)
imageList = []
for imPath in os.listdir(path):
    # FingerImages/1.jpg , ...
    image = cv2.imread(f'{path}/{imPath}')
    imageList.append(image)

# create an object
HandObject = HTM.HandDetector(detectionCon=0.7)

# finger tips id
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = HandObject.Findhands(img)
    lmList = HandObject.FindPosition(img, draw=False)

    if len(lmList) != 0:
        FingersPosition = []
        # [...-1][]: 1 landmark below the thumb tip (#3 landmark)
        # this is for the left hand. for the right hand it is vice versa
        # if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
        #     # if lmList[8][2] < lmList[6][2]:
        #     # print('index finger is open')
        #     FingersPosition.append(1)
        # else:
        #     FingersPosition.append(0)
        for id in range(0, 5):
            # for thumb finger( # 4): [][1]: x_position,
            # [...-1][]: 1 landmark below the thumb tip (#3 landmark)
            # this is for the left hand. for the right hand it is vice versa
            if id == 0:
                if lmList[tipIds[id]][1] < lmList[tipIds[id] - 1][1]:
                    # if lmList[8][2] < lmList[6][2]:
                    # print('index finger is open')
                    FingersPosition.append(1)
                else:
                    FingersPosition.append(0)

            # this is for 4 fingers. For the thumb finger it is different. [][2] here
            # indicates the y position. [..-2][] indicates two landmarks below the current
            # fingertip. based on the finger tips we can say the numbers (4, 8 ,12, 16, 20)
            # if number 8 is blow number 6(8-2), then the finger is closed
            else:
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    FingersPosition.append(1)
                else:
                    FingersPosition.append(0)
        # print(FingersPosition)
        TotalFingers = FingersPosition.count(1)
        # print(TotalFingers)
        # slicing the img
        # 6.jpg is for 0. 0-1=-1 will be the first value in python.
        # that is why 6.jpg is for zero
        h, w, c = imageList[TotalFingers-1].shape
        img[0:h, 0:w] = imageList[TotalFingers-1]

    # calculate the FBS
    cTime = time.time()
    fbs = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FBS: {int(fbs)}', (500, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 255, 255), 1)

    # show the image
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
