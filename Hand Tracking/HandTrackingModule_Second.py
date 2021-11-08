import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    def __init__(self, mode=False, maxHands=2, Complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.Complexity = Complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHandas = mp.solutions.hands
        self.hands = self.mpHandas.Hands(self.mode, self.maxHands, self.Complexity,
                                         self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def Findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        #
        if self.results.multi_hand_landmarks:
            for EachHand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, EachHand, self.mpHandas.HAND_CONNECTIONS)
        return img

    def FindPosition(self, img, HandNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[HandNo]
            for point, joint in enumerate(myHand.landmark):
                hight, width, center = img.shape
                x_position, y_position = int(joint.x * width), int(joint.y * hight)
                self.lmList.append([point, x_position, y_position])
                if draw:
                    cv2.circle(img, (x_position, y_position), 7,
                               (255, 0, 0), cv2.FILLED)
        return self.lmList, img
        # lmList is now internally available for us

    def FingersUp(self):
        FingersNum = []
        for id in range(0, 5):
            # for thumb finger( # 4): [][1]: x_position,
            # [...-1][]: 1 landmark below the thumb tip (#3 landmark)
            # this is for the right hand. for the left hand it is vice versa
            if id == 0:
                if self.lmList[self.tipIds[id]][1] < self.lmList[self.tipIds[id] - 1][1]:
                    # if lmList[8][2] < lmList[6][2]:
                    # print('index finger is open')
                    FingersNum.append(1)
                else:
                    FingersNum.append(0)

            # this is for 4 fingers. For the thumb finger it is different. [][2] here
            # indicates the y position. [..-2][] indicates two landmarks below the current
            # fingertip. based on the finger tips we can say the numbers (4, 8 ,12, 16, 20)
            # if number 8 is blow number 6(8-2), then the finger is closed
            else:
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    FingersNum.append(1)
                else:
                    FingersNum.append(0)
        return FingersNum

    def FindDistance(self, FingerTip1, FingerTip2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[FingerTip1][1:]
        x2, y2 = self.lmList[FingerTip2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 0), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    # define a video capture object
    cap = cv2.VideoCapture(0)
    DetectorObject = HandDetector()
    while True:
        # Capture the video frame by frame
        success, img = cap.read()

        img = DetectorObject.Findhands(img)
        lmList = DetectorObject.FindPosition(img)
        if len(lmList) != 0:
            # Thumb position
            print(lmList[4])
        cTime = time.time()
        fbs = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
