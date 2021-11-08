import cv2
import mediapipe as mp
import time


# define a video capture object
cap = cv2.VideoCapture(0)

# 
mpHandas = mp.solutions.hands
hands = mpHandas.Hands()
mpDraw = mp.solutions.drawing_utils

#
pTime = 0
cTime = 0
while(True):
      
    # Capture the video frame by frame
    success, img = cap.read()
    
    # Change to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    #
    if results.multi_hand_landmarks:
        for EachHand in results.multi_hand_landmarks:
            for point, joint in enumerate(EachHand.landmark):
                hight, width, center = img.shape
                x_position, y_position = int(joint.x * width), int(joint.y * hight)
                if point == 4:
                    cv2.circle(img, (x_position,y_position), 15,
                    (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, EachHand, mpHandas.HAND_CONNECTIONS)
    
    #
    cTime = time.time()
    fbs = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fbs)), (10,70), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)

    # Display the resulting frame
    cv2.imshow('frame', img)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()