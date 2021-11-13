import numpy as np
import cv2
from random import randrange


#loading pretrained data from algorithm using opencv (haarcascade algorithm) 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#choosing an image to detect faces in
img = cv2.imread('test1.jpg')


#must convert as grayscaled image
grayscale_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
    
#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    
#draw rectangles around faces with random color
for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w , y+h), (randrange(256),randrange(256),randrange(256)), 2)
    
#displaying the image with detection
cv2.imshow('Clever Face Detector',img)
#waits for a key to be pressed by the user to quit the output program 
cv2.waitKey()


print("Code Completed!")
