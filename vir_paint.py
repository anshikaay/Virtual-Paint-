import cv2
import numpy as np
import mediapipe as mp

#start webcam
cap=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=1)

#loaded mediapipe hand tracking system aur =1 means only track one hand

draw=mp.solutions.drawing_utils

#dots and lines on screen visible 

canvas=None
prev_x,prev_y=0,0
#canvas = where image will get printed, {x,y}=stores fingers last position 
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1) 
     #mirror
    h,w,_=frame.shape  #height,width of camera image
    if canvas is None:
        canvas=np.zeros_like(frame)
    #convert to rgb for mp
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)

    mode="None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
             #index fingertip
            x1=int(hand_landmarks.landmark[8].x*w)     #mp gives x,y as % so multiply with pixels
            y1=int(hand_landmarks.landmark[8].y*h)
            #3middle tip
            x2=int(hand_landmarks.landmark[12].x*w)     
            y2=int(hand_landmarks.landmark[12].y*h)
            #ring tip
            x3=int(hand_landmarks.landmark[16].x*w)     
            y3=int(hand_landmarks.landmark[16].y*h)
            #pinky tip
            x4=int(hand_landmarks.landmark[20].x*w)     
            y4=int(hand_landmarks.landmark[20].y*h)
            
            #checkif up
            y1_base=int(hand_landmarks.landmark[6].y*h)
            y2_base=int(hand_landmarks.landmark[10].y*h)
            y3_base=int(hand_landmarks.landmark[14].y*h)
            y4_base=int(hand_landmarks.landmark[18].y*h)
            
            index_up=y1<y1_base
            middle_up=y2<y2_base
            fingers=[8,12,16,20]
            bases=[6,10,14,18]
            all_up=True
            for tip,base in zip(fingers,bases):
                 tip_y=int(hand_landmarks.landmark[tip].y*h)
                 base_y=int(hand_landmarks.landmark[base].y*h)
                 if tip_y>base_y:
                      all_up=False
                      break
            if all_up:
                 mode="Clear"
                 canvas=np.zeros_like(frame)
            elif index_up and middle_up:
                 mode="Erase"
            elif index_up:
                 mode="Draw"
            else:
                 mode="None"

            if mode in["Draw","Erase"]:
              
                  color=(0,255,0) if mode=="Draw" else (0,0,0)
                  thickness=5 if mode=="Draw" else 30
                 
                  if prev_x==0 and prev_y==0:
                     prev_x,prev_y=x1,y1

                  cv2.line(canvas,(prev_x,prev_y),(x1,y1),color,thickness)
                  prev_x,prev_y=x1,y1

    #update old point to new point 
#raw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            else:
             prev_x,prev_y=0,0
            
            draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0  # No hand detected
    cv2.putText(frame,f"Mode:{mode}",(10,60),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,0,255),3)
     
    combined=cv2.addWeighted(frame,0.5,canvas,0.5,0)
    #blending webcam aur drawing
    cv2.imshow("PAINT WITH ME",combined)
    if cv2.waitKey(1) &0xFF==ord('q'):
                break
cap.release()
cv2.destroyAllWindows()

