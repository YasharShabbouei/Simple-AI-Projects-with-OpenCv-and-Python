import cv2
import numpy as np
import time
import HandTrackingModule_Second as HTM
import pyautogui

# Params
pTime = 0
wCam, hCam = 640, 480
wScr, hScr = pyautogui.size()
FrameLimit = 100  # to limit the frame size

# For smoothening
Smoothening = 5
PreviousLocationX, PreviousLocationY = 0, 0
CurrentLocationX, CurrentLocationY = 0, 0

# define a video capture object
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# create Hand Object
HandObject = HTM.HandDetector(maxHands=1, detectionCon=0.75)

while True:

    # Capture the video frame by frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = HandObject.Findhands(img)
    lmList, img = HandObject.FindPosition(img, draw=True)

    # 2. get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # index tip
        x2, y2 = lmList[12][1:]  # middle tip
        # 3. check fingers are up
        FingersNum = HandObject.FingersUp()
        # however when moving down it has some issues, so we limit the moving range:
        cv2.rectangle(img, (FrameLimit, FrameLimit),
                      (wCam - FrameLimit, hCam - FrameLimit),
                      (255, 0, 255), 2)
        # 4. only index finger: Moving mode
        if FingersNum[1] == 1 and sum(FingersNum) == 1:
            # 5. convert coordinates from webcam to our screen
            x3 = np.interp(x1, (FrameLimit, wCam-FrameLimit), (0, wScr))
            y3 = np.interp(y1, (FrameLimit, hCam-FrameLimit), (0, hScr))

            # 6. Smoothen Values: instead of x3,y3 we will send the smooth values to step 7
            CurrentLocationX = PreviousLocationX + (x3 - PreviousLocationX) / Smoothening
            CurrentLocationY = PreviousLocationY + (y3 - PreviousLocationY) / Smoothening

            # 7. Move Mouse
            pyautogui.moveTo(CurrentLocationX, CurrentLocationY)
            cv2.circle(img, (x1, y1), 20, (255, 0, 0), cv2.FILLED)
            PreviousLocationX, PreviousLocationY = CurrentLocationX, CurrentLocationY
        # 8. Both Index and middle fingers are up: Clicking mode
        if FingersNum[1] == 1 and sum(FingersNum) == 2 and FingersNum[2] == 1:
            # 9. Find distance between fingers
            length, img, LineInfo = HandObject.FindDistance(FingerTip1=8, FingerTip2=12, img=img)
            # 10. click mouse if distance is short
            if length < 45:
                cv2.circle(img, (LineInfo[4], LineInfo[5]), 20, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    # 11. Frame rate
    cTime = time.time()
    fbs = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # 12. Display the resulting frame
    cv2.imshow('frame', img)

    # 13. Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
