import numpy as np
import cv2
from random import randrange


#loading pretrained data from algorithm using opencv (haarcascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#to input a video 
vid = cv2.VideoCapture('test1.mp4')


#iterate over frames
while True:
    
    #read the current frame
    successful_frame_read, frame = vid.read()
    
    #must convert as grayscaled frame
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    
    #draw rectangles around faces with changing colors
    for(x,y,w,h) in face_coordinates:
         cv2.rectangle(frame, (x,y), (x+w , y+h), (randrange(256),randrange(256),randrange(256)), 2)
    
    #displaying the image with detection
    cv2.imshow('Clever Face Detector',frame)
    #waits for a key to be pressed by the user to quit the output program
    key = cv2.waitKey(1)

    #stop if 'q' or 'Q' is pressed 
    if key==81 or key==113:
        break

#release the VideoCapture object
vid.release()


print("Code Completed!")
